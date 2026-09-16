"""
L4/components.py
----------------
All Streamlit rendering helpers for the AI CCTV Surveillance dashboard.
"""
import streamlit as st
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .header-wrap {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #1a2332 100%);
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 1.8rem 2rem 1.4rem;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }
        .header-wrap::before {
            content: "";
            position: absolute;
            top: -40%;
            left: -10%;
            width: 60%;
            height: 200%;
            background: radial-gradient(ellipse, rgba(0,122,255,0.12) 0%, transparent 70%);
            pointer-events: none;
        }
        .header-title {
            color: #ffffff;
            font-size: 2.2rem;
            font-weight: 800;
            text-align: center;
            margin: 0;
            letter-spacing: 3px;
            text-shadow: 0 0 30px rgba(0,122,255,0.4);
        }
        .header-sub {
            color: #8b949e;
            text-align: center;
            font-size: 0.9rem;
            margin-top: 0.4rem;
            letter-spacing: 1px;
        }
        .layer-pill {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.72rem;
            font-weight: 600;
            margin: 0 3px;
            letter-spacing: 0.5px;
        }
        .pill-ready  { background: #21262d; color: #58a6ff; border: 1px solid #30363d; }
        .pill-run    { background: #0d2818; color: #3fb950; border: 1px solid #238636; }
        .pill-done   { background: #1a2040; color: #79c0ff; border: 1px solid #1f6feb; }
        .pill-error  { background: #2d1214; color: #f85149; border: 1px solid #6e1c1e; }
        .pill-init   { background: #2d2208; color: #e3b341; border: 1px solid #9e6a03; }
        </style>
        <div class="header-wrap">
            <h1 class="header-title">🎯 AI CCTV SURVEILLANCE</h1>
            <div class="header-sub">
                Real-time · L1 Motion · L2 Detection · L3 Behaviour · VLM Reasoning
            </div>
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM STATUS BAR
# ─────────────────────────────────────────────────────────────────────────────

def render_system_status(layer_statuses: dict):
    """
    layer_statuses: dict mapping layer name to one of
    READY | INITIALIZING | RUNNING | DONE | ERROR | OFFLINE
    """
    cls_map = {
        "READY":        "pill-ready",
        "INITIALIZING": "pill-init",
        "RUNNING":      "pill-run",
        "DONE":         "pill-done",
        "ERROR":        "pill-error",
        "OFFLINE":      "pill-ready",
    }
    icon_map = {
        "READY":        "🔵",
        "INITIALIZING": "🟡",
        "RUNNING":      "🟢",
        "DONE":         "✅",
        "ERROR":        "🔴",
        "OFFLINE":      "⚫",
    }

    pills_html = ""
    for name, status in layer_statuses.items():
        cls = cls_map.get(status, "pill-ready")
        icon = icon_map.get(status, "⚫")
        pills_html += f'<span class="layer-pill {cls}">{icon} {name}: {status}</span>'

    st.markdown(
        f'<div style="text-align:center;margin:0.6rem 0 1.2rem;">{pills_html}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD PANEL
# ─────────────────────────────────────────────────────────────────────────────

def render_upload_panel(videos_dir: Path):
    """
    Returns the absolute path (str) of the saved uploaded video,
    or None if nothing is uploaded.
    """
    st.markdown("### 📂 Upload Video")
    uploaded = st.file_uploader(
        "Upload a video file (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov"],
        key="video_uploader",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        videos_dir.mkdir(parents=True, exist_ok=True)
        save_path = videos_dir / f"uploaded_{uploaded.name}"
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"✅ Saved: `{save_path.name}` ({uploaded.size // 1024} KB)")
        return str(save_path)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATED VIDEO OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def render_video(video_path: Path):
    st.markdown("#### 🎥 Annotated Output")
    if video_path and Path(video_path).exists():
        try:
            with open(video_path, "rb") as vf:
                st.video(vf.read())
        except Exception as e:
            st.error(f"Error loading video: {e}")
    else:
        st.markdown(
            '<div style="height:320px;background:#0d1117;border:1px solid #30363d;'
            'border-radius:10px;display:flex;align-items:center;justify-content:center;'
            'color:#484f58;font-size:1rem;">No annotated video yet — start the pipeline</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# L1 MOTION CHART
# ─────────────────────────────────────────────────────────────────────────────

def render_l1_chart(l1_results: list):
    st.markdown("#### 📊 L1 — Frame Motion %")
    if not l1_results:
        st.info("Waiting for L1 motion data…")
        return

    df = pd.DataFrame(l1_results)
    if "timestamp_sec" not in df.columns or "change_percentage" not in df.columns:
        st.warning("L1 data format unexpected.")
        return

    # Keep last 300 data points for performance
    df = df.tail(300)
    df = df.rename(columns={"timestamp_sec": "Time (s)", "change_percentage": "Change %"})

    # Highlight high-motion frames
    threshold = 5.0
    df["Alert"] = df["Change %"] > threshold

    st.line_chart(df.set_index("Time (s)")["Change %"], height=200, use_container_width=True)

    high = df[df["Alert"]]
    if not high.empty:
        st.warning(f"⚡ {len(high)} high-motion frames detected (>{threshold}%)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Frames Analysed", len(df))
    col2.metric("Avg Motion %", f"{df['Change %'].mean():.2f}")
    col3.metric("Max Motion %", f"{df['Change %'].max():.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# L2 LIVE DETECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def render_live_detections(stats_list: list):
    st.markdown("#### 🔍 L2 — Live Detections")
    if not stats_list:
        st.info("Waiting for detection data…")
        return

    latest = stats_list[-1]
    tracks = latest.get("tracks", {})

    col1, col2, col3 = st.columns(3)
    persons  = sum(1 for t in tracks.values() if t.get("class", "") == "person")
    vehicles = sum(1 for t in tracks.values()
                   if t.get("class", "") in ["car", "truck", "bus", "motorcycle"])
    col1.metric("Active Tracks", latest.get("active_tracks_count", 0))
    col2.metric("Persons", persons)
    col3.metric("Vehicles", vehicles)

    if tracks:
        rows = []
        for tid, t in list(tracks.items())[-30:]:  # show last 30
            rows.append({
                "Track ID":   tid,
                "Class":      t.get("class", "?"),
                "Confidence": round(t.get("confidence", 0), 2),
                "State":      t.get("movement_state", "—"),
                "Frame":      t.get("last_seen_frame", "—"),
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No active tracks in the latest frame.")


# ─────────────────────────────────────────────────────────────────────────────
# L3 EVENTS
# ─────────────────────────────────────────────────────────────────────────────

_SEV_COLORS = {
    "critical": "#f85149",
    "high":     "#e3b341",
    "medium":   "#58a6ff",
    "low":      "#3fb950",
}

def render_events_live(events: list):
    st.markdown("#### ⚠️ L3 — Suspicious Events")
    if not events:
        st.success("No suspicious events detected.")
        return

    st.markdown(f"**{len(events)} event(s) logged**")

    for ev in reversed(events[-50:]):   # newest first, max 50
        sev   = str(ev.get("severity", "low")).lower()
        color = _SEV_COLORS.get(sev, "#8b949e")
        etype = ev.get("event_type", "unknown").replace("_", " ").title()
        tid   = ev.get("track_id", "?")
        ts    = ev.get("timestamp", "?")
        dur   = ev.get("duration", 0)
        obj   = ev.get("object_type", "?")

        st.markdown(
            f"""
            <div style="border-left:4px solid {color};padding:8px 14px;
                        margin-bottom:8px;background:#161b22;border-radius:0 8px 8px 0;">
                <span style="color:{color};font-weight:700;font-size:0.85rem;">
                    {etype.upper()} · {sev.upper()}
                </span><br/>
                <span style="color:#8b949e;font-size:0.8rem;">
                    Track {tid} · {obj} · {dur:.1f}s · @ {ts:.1f}s
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# VLM ANALYSIS CARDS
# ─────────────────────────────────────────────────────────────────────────────

def render_vlm_cards(vlm_results: list):
    st.markdown("#### 🧠 VLM — AI Analysis")
    if not vlm_results:
        st.info("VLM analysis will appear here after a suspicious event is detected…")
        return

    for item in reversed(vlm_results[-20:]):
        etype = str(item.get("event_type", "Event")).replace("_", " ").title()
        eid   = str(item.get("event_id", ""))[:8]
        tid   = item.get("track_id", "?")
        nf    = item.get("frames_used", "?")
        model = item.get("model", "VLM")
        text  = item.get("vlm_analysis", "No analysis available.")

        with st.expander(f"🔎 {etype}  ·  Track {tid}  ·  event `{eid}…`"):
            st.markdown(
                f'<p style="color:#8b949e;font-size:0.78rem;">'
                f'Model: {model} · Frames: {nf}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#0d1117;padding:12px;border-radius:8px;'
                f'border:1px solid #30363d;color:#e6edf3;line-height:1.6;">'
                f'{text}</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY HELPERS (kept for compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def render_stats(stats_dict: dict):
    st.markdown("#### 📈 Detection Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Tracks",    stats_dict.get("active_tracks", 0))
    col2.metric("Persons Detected", stats_dict.get("persons", 0))
    col3.metric("Vehicles Detected",stats_dict.get("vehicles", 0))


def render_alert_table(events_df: pd.DataFrame):
    st.markdown("#### 🚨 Suspicious Events & VLM Analysis")

    if events_df.empty:
        st.success("No suspicious events detected.")
        return

    f1, f2, f3 = st.columns(3)
    with f1:
        sel_type = st.selectbox("Event Type",  ["All"] + list(events_df["event_type"].unique()))
    with f2:
        sel_obj  = st.selectbox("Object Type", ["All"] + list(events_df["object_type"].unique()))
    with f3:
        sel_sev  = st.selectbox("Severity",    ["All"] + list(events_df["severity"].unique()))

    fdf = events_df.copy()
    if sel_type != "All": fdf = fdf[fdf["event_type"]  == sel_type]
    if sel_obj  != "All": fdf = fdf[fdf["object_type"] == sel_obj]
    if sel_sev  != "All": fdf = fdf[fdf["severity"]    == sel_sev]

    st.caption(f"Showing **{len(fdf)}** events.")
    st.dataframe(
        fdf[["event_id","timestamp","track_id","object_type",
             "event_type","duration","severity","vlm_analysis"]],
        use_container_width=True,
        hide_index=True,
    )
