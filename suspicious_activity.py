#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
suspicious_activity.py -- standalone suspicious-activity detector for camera frames.

What it does
------------
Takes 1-2 consecutive image frames from one camera (captured 1 second apart),
sends them as an array of images in a single request to a vision model on
OpenRouter (primary: qwen/qwen3.8-flash) with low reasoning effort, and reports
whether any suspicious activity is visible:

    {
      "suspicious": true,          # boolean verdict
      "significance": "high",      # "low" | "medium" | "high"
      "reason": "Person forcing a window open.",
      "model": "qwen/qwen3.8-flash",
      "frames": 2
    }

Design bias: the model is told to prefer FALSE ALARMS over missed events, so it
over-reports rather than under-reports. In the same spirit, an unparsable model
reply is treated as suspicious=true with significance "low" (never a silent miss).

Resilience: if the primary model is rate-limited, errors, or returns unusable
output, the script automatically falls back through:
    qwen/qwen2.5-vl-72b-instruct  ->  openai/gpt-4o-mini

Requirements
------------
- Python 3.8+ (standard library only, no pip installs)
- OpenRouter API key in the OPENROUTER_API_KEY environment variable
  (get one at https://openrouter.ai/keys)

CLI examples
------------
    python suspicious_activity.py frame_0001.jpg frame_0002.jpg
    python suspicious_activity.py .\\frames --last 2        # newest 2 frames in dir
    echo '["<base64 jpeg>", "<base64 jpeg>"]' | python suspicious_activity.py --stdin-b64
    python suspicious_activity.py f0.jpg f1.jpg --exit-on-suspicious
        # exit code 10 when suspicious (handy in automation)

Library example
---------------
    from suspicious_activity import analyze_frames
    verdict = analyze_frames(["f0.jpg", "f1.jpg"])
    if verdict["suspicious"]:
        alert(verdict["significance"], verdict["reason"])

Model / payload
---------------
Primary model: qwen/qwen3.8-flash (override with --model or OPENROUTER_MODEL).
Fallbacks:     qwen/qwen2.5-vl-72b-instruct, openai/gpt-4o-mini
The payload is sent with "reasoning_effort": "low" (fast/cheap but still reasons);
override with --reasoning-effort or pass reasoning_effort=... to analyze_frames().
Models that don't support reasoning_effort (e.g. gpt-4o-mini) simply omit it.
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.8-flash"
# Fallback chain used when the primary model fails or returns unusable output
# (e.g. upstream rate limits, empty replies, unparsable text). Each entry is
# tried in order until one produces a usable verdict.
FALLBACK_MODELS = [
    "qwen/qwen2.5-vl-72b-instruct",
    "openai/gpt-4o-mini",
]
DEFAULT_REASONING_EFFORT = "low"
# Hard-coded default key (used when OPENROUTER_API_KEY is not set).
# NOTE: Removed hardcoded key to comply with GitHub secret scanning rules.
DEFAULT_API_KEY = ""
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
VALID_SIGNIFICANCE = ("low", "medium", "high")
# Models known NOT to accept a reasoning_effort payload parameter.
REASONING_EFFORT_UNSUPPORTED = {"openai/gpt-4o-mini"}

SYSTEM_PROMPT = (
    "You are a rigorous CCTV security analyst. You receive consecutive camera "
    "frames and output one strict JSON verdict object and nothing else."
)


# --------------------------------------------------------------------------
# prompt building
# --------------------------------------------------------------------------

def build_user_prompt(n_frames):
    plural = "s" if n_frames != 1 else ""
    return (
        f"You are given {n_frames} consecutive image frame{plural} from ONE "
        "camera, in chronological order, captured 1 second apart "
        f"(frame 1 is earliest, frame {n_frames} latest).\n\n"
        "Decide whether the scene shows SUSPICIOUS ACTIVITY, for example:\n"
        "- intrusion / trespass / someone in a restricted or staff-only area\n"
        "- climbing a fence, forcing a door or window, breaking & entering\n"
        "- a visible weapon, fighting, violence, threats, someone being restrained\n"
        "- tampering with cameras, doors, locks, gates, or equipment\n"
        "- loitering, prowling, peeking into windows, hiding from the camera\n"
        "- smoke, fire, sparks, flooding, gas leak\n"
        "- a bag or object moved, dumped, or left behind\n"
        "- a person moving suspiciously fast, at an unusual hour, or deliberately "
        "avoiding being seen\n\n"
        "Rules:\n"
        "1. Reply ONLY with a single JSON object, no commentary, no markdown:\n"
        '   {"suspicious": true|false, "significance": "<low|medium|high>", '
        '"reason": "<one short sentence>"}\n'
        '2. "suspicious" must be a JSON boolean. Ordinary scenes (empty rooms, '
        "normal walking, vehicles driving, pets, weather, shadows, camera noise) "
        "are FALSE.\n"
        '3. "significance" grades the threat when suspicious is true: "low" = '
        'ambiguous/minor, "medium" = clear but limited threat, "high" = crime or '
        'danger in progress. When suspicious is false use "low".\n'
        "4. BIAS TOWARD FALSE ALARMS: if you are uncertain or the evidence is "
        "borderline, answer suspicious=true. False alarms are acceptable; "
        "missed events are NOT.\n"
        "5. If the frames are unreadable or corrupt, answer suspicious=true with "
        'significance "low" and say so in reason.\n\n'
        f"There {'is' if n_frames == 1 else 'are'} {n_frames} frame{plural} in total. Begin."
    )


# --------------------------------------------------------------------------
# image -> data URL helpers
# --------------------------------------------------------------------------

def sniff_mime(data):
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"GIF8":
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] == b"BM":
        return "image/bmp"
    return None


def to_data_url(kind, value):
    """kind: 'file' (path), 'b64' (base64 string), 'bytes' (raw image bytes)."""
    mime = None
    if kind == "file":
        with open(value, "rb") as fh:
            data = fh.read()
        mime = mimetypes.guess_type(value)[0]
    elif kind == "b64":
        b64 = re.sub(r"\s+", "", value)
        if b64.startswith("data:"):           # already a data URL
            return b64
        try:
            data = base64.b64decode(b64)
        except Exception as exc:
            raise ValueError(f"invalid base64 frame: {exc}")
    else:  # bytes
        data = bytes(value)

    if not mime:
        mime = sniff_mime(data)
    if not mime or not mime.startswith("image/"):
        raise ValueError(f"unrecognized image data in {kind} input")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# --------------------------------------------------------------------------
# OpenRouter call
# --------------------------------------------------------------------------

def _build_body(image_urls, model, reasoning_effort, strict_json, provider_routing):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text",
                                          "text": build_user_prompt(len(image_urls))}]
             + [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]},
        ],
        "temperature": 0.1,
        "max_tokens": 400,
    }
    if reasoning_effort and reasoning_effort != "none":
        body["reasoning_effort"] = reasoning_effort
    if provider_routing:
        # Explicitly allow fallback to other OpenRouter providers serving this
        # model when the default upstream is rate-limited. Valid syntax:
        # provider.allow_fallbacks (not a top-level "route" key).
        body["provider"] = {"allow_fallbacks": True}
    if strict_json:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "suspicious_activity_verdict",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "suspicious": {"type": "boolean"},
                        "significance": {"type": "string", "enum": list(VALID_SIGNIFICANCE)},
                        "reason": {"type": "string"},
                    },
                    "required": ["suspicious", "significance", "reason"],
                    "additionalProperties": False,
                },
            },
        }
    return body


def _make_request(body, api_key):
    return urllib.request.Request(
        OPENROUTER_URL, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/suspicious-activity-detector",
            "X-Title": "Suspicious Activity Detector",
        },
    )


def _http_error_detail(exc):
    try:
        return exc.read().decode("utf-8", "replace")[:400]
    except Exception:
        return ""


def call_openrouter(image_urls, model, api_key, reasoning_effort=DEFAULT_REASONING_EFFORT,
                    timeout=90, max_retries=3, strict_json=False,
                    provider_routing=True, debug_log=None):
    """Call ONE OpenRouter vision model with the frames; return raw reply text.

    Retries transient failures (429/5xx, empty content) with backoff. Raises
    RuntimeError if the model can't produce a non-empty reply. The caller
    (analyze_frames) is responsible for trying fallback models. If the model
    rejects reasoning_effort (HTTP 400), it is dropped and the request retried
    once.
    """
    body = _build_body(image_urls, model, reasoning_effort, strict_json,
                       provider_routing)
    req = _make_request(body, api_key)
    re_dropped = False

    last_err = None
    for attempt in range(1, max_retries + 1):
        payload = None
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = _http_error_detail(exc)
            if (exc.code == 400 and not re_dropped
                    and "reasoning_effort" in body
                    and ("reasoning_effort" in detail.lower()
                         or "unrecognized" in detail.lower())):
                # Model rejects reasoning_effort: drop it and retry once
                # (does not consume the retry budget this iteration).
                re_dropped = True
                body.pop("reasoning_effort", None)
                req = _make_request(body, api_key)
                last_err = f"HTTP 400 (retrying without reasoning_effort): {detail}"
                if debug_log:
                    debug_log(f"[{model}] reasoning_effort rejected; retrying without it")
                time.sleep(1)
                continue
            if exc.code in (400, 401, 402, 403):
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}")
            last_err = f"HTTP {exc.code}: {detail}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = str(exc)
        if payload is None:
            if attempt < max_retries:
                time.sleep(2 ** attempt)          # 2s, 4s, 8s backoff
            continue

        api_err = payload.get("error")
        if api_err:
            raise RuntimeError(f"OpenRouter error: {api_err}")
        choice = payload["choices"][0]
        msg = choice.get("message", {})
        raw = msg.get("content")
        if isinstance(raw, list):
            # OpenAI-style multi-part content: join text parts.
            parts = []
            for part in raw:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(str(part.get("text", "")))
                    else:
                        parts.append(json.dumps(part, ensure_ascii=False))
                else:
                    parts.append(str(part))
            raw = "".join(parts)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            # Empty/null reply (e.g. upstream flakiness, reasoning-only
            # responses). Treat as a transient failure and retry.
            fin = choice.get("finish_reason")
            reasoning = msg.get("reasoning")
            last_err = (
                f"empty reply (finish_reason={fin}, "
                f"reasoning={str(reasoning)[:120]!r})"
            )
        else:
            if debug_log:
                debug_log(f"[{model}] {raw}")
            return raw if isinstance(raw, str) else json.dumps(raw)
        if attempt < max_retries:
            time.sleep(2 ** attempt)          # 2s, 4s, 8s backoff
    raise RuntimeError(
        f"model {model} failed after {max_retries} attempts: {last_err}")


# --------------------------------------------------------------------------
# response parsing
# --------------------------------------------------------------------------

def extract_json(text):
    """Pull the first JSON object out of a model reply (tolerates fences)."""
    if not text:
        raise ValueError("empty model response")
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    try:
        obj = json.loads(t)
        # tolerate a JSON array wrapping the verdict object
        if isinstance(obj, list):
            obj = next((x for x in obj if isinstance(x, dict)), None)
        return obj
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    while start != -1:
        depth = 0
        in_str = esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(t[start:i + 1])
                        except json.JSONDecodeError:
                            break
        start = t.find("{", start + 1)
    raise ValueError(f"no JSON object found in model response: {text[:160]!r}")


def normalize_verdict(obj):
    if not isinstance(obj, dict):
        raise ValueError(f"verdict is not a JSON object: {obj!r}")
    suspicious = bool(obj.get("suspicious", False))
    sig = str(obj.get("significance", "")).strip().lower()
    if sig not in VALID_SIGNIFICANCE:
        sig = "medium" if suspicious else "low"      # never under-report severity
    reason = str(obj.get("reason", "")).strip()
    if not reason:
        reason = ("Suspicious activity detected." if suspicious
                  else "No suspicious activity detected.")
    return {"suspicious": suspicious, "significance": sig, "reason": reason}


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def _try_parse_reply(raw, urls, model, api_key, reasoning_effort,
                     timeout, max_retries, strict_json, provider_routing,
                     debug_log):
    """Try to turn one model's raw reply into a verdict; None if unusable."""
    try:
        return normalize_verdict(extract_json(raw))
    except ValueError:
        pass
    # The plain-text JSON request may have produced prose. If the model
    # supports response_format, retry once asking for JSON schema.
    if strict_json:
        return None
    try:
        if debug_log:
            debug_log(f"[parse failed; retrying {model} with response_format json_schema] raw={raw[:400]!r}")
        reply2 = call_openrouter(
            urls, model, api_key,
            reasoning_effort=reasoning_effort,
            timeout=timeout, max_retries=max_retries,
            strict_json=True,
            provider_routing=provider_routing,
            debug_log=debug_log)
        verdict = normalize_verdict(extract_json(reply2))
        verdict["_refined"] = True
        return verdict
    except (ValueError, RuntimeError):
        return None


def analyze_frames(frames, model=None, api_key=None,
                   reasoning_effort=DEFAULT_REASONING_EFFORT,
                   timeout=90, max_retries=3, strict_json=False,
                   provider_routing=True, debug_log=None,
                   fallback_models=None):
    """Analyze consecutive frames (1 frame / second) and return a verdict dict.

    frames: iterable of
      - file paths (str)
      - base64 image strings (str)
      - raw image bytes
      - (bytes, mime) tuples
      - ("file"|"b64"|"bytes"|"url", value) tuples

    Returns: {"suspicious": bool, "significance": "low"|"medium"|"high",
              "reason": str, "model": str, "frames": int}
    """
    model = model or os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or DEFAULT_API_KEY
    if not api_key:
        raise RuntimeError("OpenRouter API key missing: set OPENROUTER_API_KEY or pass api_key=...")

    urls = []
    for f in frames:
        if isinstance(f, tuple) and len(f) == 2:
            k, v = f
            if k == "url":
                urls.append(v)
            elif k in ("file", "b64", "bytes"):
                urls.append(to_data_url(k, v))
            elif isinstance(k, (bytes, bytearray)):      # (bytes, mime)
                mime = v or sniff_mime(bytes(k)) or "image/jpeg"
                if not mime.startswith("image/"):
                    mime = "image/jpeg"
                urls.append(f"data:{mime};base64,{base64.b64encode(bytes(k)).decode('ascii')}")
            else:
                raise ValueError(f"unsupported frame tuple: {k!r}")
        elif isinstance(f, str):
            urls.append(to_data_url("file" if os.path.isfile(f) else "b64", f))
        elif isinstance(f, (bytes, bytearray)):
            urls.append(to_data_url("bytes", f))
        else:
            raise ValueError(f"unsupported frame input: {type(f).__name__}")
    if not urls:
        raise ValueError("no frames provided")

    # Try the primary model, then the fallback chain in order. The chain is
    # used on any failure (HTTP errors, empty replies) or when the reply can't
    # be parsed into a verdict.
    chain = [model]
    if fallback_models is None:
        fallback_models = FALLBACK_MODELS
    chain += [m for m in fallback_models if m != model]

    last_err = None
    verdict = None
    for candidate in chain:
        if debug_log:
            debug_log(f"[trying model] {candidate}")
        try:
            raw = call_openrouter(
                urls, candidate, api_key,
                reasoning_effort=reasoning_effort,
                timeout=timeout, max_retries=max_retries,
                strict_json=strict_json,
                provider_routing=provider_routing,
                debug_log=debug_log)
            v = _try_parse_reply(raw, urls, candidate, api_key,
                                 reasoning_effort, timeout, max_retries,
                                 strict_json, provider_routing, debug_log)
        except RuntimeError as exc:
            last_err = str(exc)
            v = None
        if v is not None:
            verdict = v
            verdict["model"] = candidate
            break
        last_err = f"model {candidate} returned no usable verdict"
        if debug_log:
            debug_log(f"[model {candidate}] no usable verdict; next fallback")

    if verdict is None:
        # Every model in the chain failed. Never under-report: treat as
        # suspicious by default and surface the underlying error.
        verdict = normalize_verdict({
            "suspicious": True,
            "significance": "low",
            "reason": "All models failed; flagged as precaution.",
        })
        verdict["_error"] = last_err
    verdict.update({"frames": len(urls)})
    return verdict


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

EXAMPLES = """examples:
  python suspicious_activity.py frame_0001.jpg frame_0002.jpg
  python suspicious_activity.py .\\frames --last 2
  echo '["<b64>", "<b64>"]' | python suspicious_activity.py --stdin-b64

exit codes:
  0   ran fine
  1   error (bad input, missing key, API failure)
  10  ran fine AND suspicious=true (only with --exit-on-suspicious)
"""


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    p = argparse.ArgumentParser(
        prog="suspicious_activity.py",
        description="Flag suspicious activity in 1-2 consecutive camera frames "
                    "(1 fps) using a Qwen vision model on OpenRouter. "
                    "Prefers false alarms over missed events.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES)
    p.add_argument("frames", nargs="*", metavar="FRAME",
                   help="image file (jpg/png/webp/...) or a directory of frames")
    p.add_argument("--dir", metavar="PATH",
                   help="directory to expand into image frames")
    p.add_argument("--last", type=int, default=2, metavar="N",
                   help="keep only the newest N frames when expanding a directory "
                        "(default 2)")
    p.add_argument("--max-frames", type=int, default=8, metavar="N",
                   help="hard cap on the number of frames sent (default 8)")
    p.add_argument("--stdin-b64", action="store_true",
                   help="read a JSON array of base64 image strings from stdin")
    p.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL,
                   help=f"OpenRouter model id (default {DEFAULT_MODEL})")
    p.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                   choices=["minimal", "low", "medium", "high", "none"],
                   help=f"reasoning effort sent in the payload (default {DEFAULT_REASONING_EFFORT})")
    p.add_argument("--api-key", default=None,
                   help="OpenRouter API key (default: $OPENROUTER_API_KEY)")
    p.add_argument("--timeout", type=float, default=90,
                   help="per-attempt timeout in seconds (default 90)")
    p.add_argument("--retries", type=int, default=3,
                   help="max attempts before giving up (default 3)")
    p.add_argument("--strict-json", action="store_true",
                   help="ask for JSON-schema structured output "
                        "(requires model support for response_format)")
    p.add_argument("--no-fallback", action="store_true",
                   help="disable OpenRouter provider routing/fallback "
                        "(default: allow_fallbacks is sent)")
    p.add_argument("--debug", action="store_true",
                   help="print raw model replies (and the refined re-request) "
                        "to stderr for debugging")
    p.add_argument("--exit-on-suspicious", action="store_true",
                   help="exit with code 10 when suspicious is true")
    args = p.parse_args(argv)

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or DEFAULT_API_KEY
    if not api_key:
        sys.exit("error: OPENROUTER_API_KEY is not set "
                 "(get one at https://openrouter.ai/keys)")

    inputs = []                                   # list of (kind, value) tuples
    paths = list(args.frames)
    if args.dir:
        paths.append(args.dir)
    for path in paths:
        if os.path.isdir(path):
            files = [os.path.join(path, n) for n in sorted(os.listdir(path))
                     if os.path.splitext(n)[1].lower() in IMAGE_EXTS]
            if args.last and len(files) > args.last:
                files = files[-args.last:]
            inputs.extend(("file", f) for f in files)
        elif os.path.isfile(path):
            inputs.append(("file", path))
        else:
            sys.exit(f"error: frame path not found: {path}")

    if args.stdin_b64:
        raw = sys.stdin.read().strip()
        try:
            arr = json.loads(raw)
        except json.JSONDecodeError:
            sys.exit("error: stdin is not valid JSON "
                     "(expected an array of base64 image strings)")
        if isinstance(arr, dict):
            arr = arr.get("frames", [])
        if not isinstance(arr, list):
            sys.exit("error: --stdin-b64 expects a JSON array of base64 strings")
        inputs = [("b64", s) for s in arr]

    if not inputs:
        sys.exit("error: no frames given "
                 "(pass file paths, a directory, or --stdin-b64)")
    if len(inputs) > args.max_frames:
        inputs = inputs[-args.max_frames:]

    print(f"* analyzing {len(inputs)} frame(s) via {args.model} "
          f"(fallbacks: {', '.join(FALLBACK_MODELS)}) "
          f"(reasoning_effort={args.reasoning_effort})", file=sys.stderr)
    debug_log = None
    if args.debug:
        debug_log = lambda text: print(f"[debug] {text}", file=sys.stderr)
    try:
        verdict = analyze_frames(
            inputs, model=args.model, api_key=api_key,
            reasoning_effort=args.reasoning_effort,
            timeout=args.timeout, max_retries=args.retries,
            strict_json=args.strict_json,
            provider_routing=not args.no_fallback,
            debug_log=debug_log)
    except (RuntimeError, ValueError) as exc:
        sys.exit(f"error: {exc}")

    verdict.pop("_refined", None)
    err = verdict.pop("_error", None)
    if err and args.debug:
        print(f"[debug] all models failed: {err}", file=sys.stderr)
    print(json.dumps(verdict, indent=2))
    if args.exit_on_suspicious and verdict["suspicious"]:
        sys.exit(10)


if __name__ == "__main__":
    main()