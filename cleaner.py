import os
import io
import tokenize
import re


def clean_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return

    # 1. Remove comments using tokenize
    result = []
    try:
        tokens = tokenize.tokenize(io.BytesIO(source.encode("utf-8")).readline)
    except tokenize.TokenError:
        print(f"Token error in {filepath}")
        return

    last_lineno = -1
    last_col = 0

    for tok in tokens:
        token_type = tok.type
        token_string = tok.string
        start_line, start_col = tok.start
        end_line, end_col = tok.end

        # Keep track of formatting
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            result.append(" " * (start_col - last_col))

        # Filter comments
        if token_type == tokenize.COMMENT:
            # We want to keep some comments if they are genuinely required.
            lower_comment = token_string.lower()
            if "todo" in lower_comment or "fixme" in lower_comment:
                result.append(token_string)
            elif (
                "type: ignore" in lower_comment
                or "noqa" in lower_comment
                or "pylint:" in lower_comment
            ):
                result.append(token_string)
            elif (
                "!" in token_string or "?" in token_string
            ):  # heuristic for "important"
                # Actually, no, user wants all unnecessary removed.
                pass
            else:
                # Remove it
                pass
        else:
            result.append(token_string)

        last_lineno = end_line
        last_col = end_col

    cleaned_source = "".join(result)

    # 2. Remove print debugs and docstrings.
    # To remove simple print(...) debugging, we can use regex on lines.
    # Since parsing AST and unparsing drops all formatting (like empty lines).
    lines = cleaned_source.splitlines()
    new_lines = []

    # We will also remove AI section banners like # =========================
    for line in lines:
        stripped = line.strip()
        # if the line is just a simple print statement with no assignment
        # like: print("Starting") or print(var)
        if re.match(r"^print\s*\([^)]*\)\s*$", stripped):
            if (
                "error" not in stripped.lower()
                and "exception" not in stripped.lower()
                and "critical" not in stripped.lower()
            ):
                continue  # Skip simple prints

        # if it's a completely empty line and the previous line was also empty, skip it to avoid huge gaps
        if not stripped and new_lines and not new_lines[-1].strip():
            # but allow up to 2 empty lines
            if (
                len(new_lines) >= 2
                and not new_lines[-1].strip()
                and not new_lines[-2].strip()
            ):
                continue

        new_lines.append(line)

    final_source = "\n".join(new_lines)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(final_source)
    print(f"Cleaned {filepath}")


if __name__ == "__main__":
    for root, dirs, files in os.walk(r"D:\VLM"):
        if (
            ".venv" in root
            or ".git" in root
            or "node_modules" in root
            or "CCTV" in root
            or "datasets" in root
        ):
            # wait, should we clean CCTV? "Perform a complete CLEAN CODE PASS on this project."
            # yes, we should clean everything except .venv and .git
            pass

        # Actually filter out specific directories to not traverse
        dirs[:] = [
            d for d in dirs if d not in [".venv", ".git", "node_modules", "__pycache__"]
        ]

        for file in files:
            if file.endswith(".py") and file != "cleaner.py":
                clean_file(os.path.join(root, file))
