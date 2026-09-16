"""
L4/dashboard.py
---------------
Main Streamlit entry-point for the AI CCTV Surveillance platform.

Run with:
    streamlit run L4/dashboard.py
"""
import sys
import time
import threading
from pathlib import Path

import streamlit as st

# ── Project root on sys.path ───────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from L4.data_loader import DataLoader
from L4.pipeline_runner import PipelineRunner, get_runner, set_runner
from L4.components import (
    render_header,
    render_system_status,
    render_upload_panel,
    render_video,
    render_l1_chart,
    render_live_detections,
    render_events_live,
    render_vlm_cards,
    render_alert_table,
    render_stats,
)

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI CCTV Surveillance",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CONFIG_PATH   = project_root / "config" / "settings.yaml"
VIDEOS_DIR    = project_root / "videos"
OUTPUT_VIDEO  = project_root / "outputs" / "detections" / "tracking_annotated.webm"

# ── Session State Bootstrap ────────────────────────────────────────────────
_defaults = {
    "video_path":     None,     # str: path to the uploaded video
    "processing":     False,    # bool
    "vlm_enabled":    True,     # bool: toggle in sidebar
    "auto_refresh":   True,     # bool
    # Live data queues (lists, appended by pipeline callbacks)
    "q_stats":        [],
    "q_l1":           [],
    "q_events":       [],
    "q_vlm":          [],
    "q_status":       [],
    "q_error":        [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ────────────────────────────────────────────────────────────────

def _reset_queues():
    for key in ("q_stats", "q_l1", "q_events", "q_vlm", "q_status", "q_error"):
        st.session_state[key] = []


def _derive_layer_statuses() -> dict:
    status_list = st.session_state.q_status
    errors      = st.session_state.q_error
    processing  = st.session_state.processing

    if errors:
        unified = "ERROR"
    elif not status_list:
        unified = "READY"
    else:
        unified = status_list[-1]

    layers = ["L1", "L2", "Tracking", "L3", "VLM"]

    if unified == "RUNNING":
        stats  = st.session_state.q_stats
        l1     = st.session_state.q_l1
        events = st.session_state.q_events
        vlm    = st.session_state.q_vlm
        return {
            "L1":       "RUNNING" if l1     else "INITIALIZING",
            "L2":       "RUNNING" if stats  else "INITIALIZING",
            "Tracking": "RUNNING" if stats  else "INITIALIZING",
            "L3":       "RUNNING" if events else "RUNNING",
            "VLM":      "RUNNING" if vlm    else ("RUNNING" if st.session_state.vlm_enabled else "OFFLINE"),
        }
    elif unified == "DONE":
        return {l: "DONE" for l in layers}
    elif unified == "ERROR":
        return {l: "ERROR" for l in layers}
    elif unified == "INITIALIZING":
        return {l: "INITIALIZING" for l in layers}
    else:
        return {l: "READY" for l in layers}


def _start_pipeline():
    if st.session_state.processing:
        return
    if not st.session_state.video_path:
        st.session_state.q_error.append("No video/webcam selected. Please upload a video or use webcam first.")
        return

    _reset_queues()

    state_queues = {
        "stats":  st.session_state.q_stats,
        "l1":     st.session_state.q_l1,
        "events": st.session_state.q_events,
        "vlm":    st.session_state.q_vlm,
        "status": st.session_state.q_status,
        "error":  st.session_state.q_error,
    }

    runner = PipelineRunner(
        CONFIG_PATH,
        st.session_state.video_path,
        state_queues,
        vlm_enabled=st.session_state.vlm_enabled,
    )
    runner.start()

    set_runner(runner)
    st.session_state.processing = True


def _stop_pipeline():
    runner: PipelineRunner | None = get_runner()
    if runner and runner.is_alive():
        runner.stop()
    st.session_state.processing = False
    set_runner(None)
    st.session_state.q_status.append("DONE")


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("**VLM Analysis**")
    vlm_on = st.toggle(
        "Enable VLM (AI visual reasoning)",
        value=st.session_state.vlm_enabled,
        help="Disable to speed up processing. VLM loads a large model on first run.",
    )
    st.session_state.vlm_enabled = vlm_on

    st.markdown("---")
    st.markdown("**Auto-Refresh**")
    auto = st.toggle("Auto-refresh UI while running", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto

    st.markdown("---")
    st.markdown("**Config**")
    st.caption(f"`{CONFIG_PATH}`")

    if st.session_state.q_error:
        st.markdown("---")
        st.error("**Errors**")
        for err in st.session_state.q_error:
            st.code(err)

    st.markdown("---")
    st.markdown(
        "<div style='color:#484f58;font-size:0.75rem;text-align:center;'>"
        "L1→L2→L3→VLM→L4<br/>VLM Surveillance Platform</div>",
        unsafe_allow_html=True,
    )


# ── Main Layout ────────────────────────────────────────────────────────────

render_header()

# ── Layer status bar ──────────────────────────────────────────────────────
render_system_status(_derive_layer_statuses())

# ── Upload + Controls ─────────────────────────────────────────────────────
up_col, ctrl_col = st.columns([2, 1])

with up_col:
    saved_path = render_upload_panel(VIDEOS_DIR)
    if saved_path:
        st.session_state.video_path = saved_path

with ctrl_col:
    st.markdown("### ▶ Controls")

    use_webcam = st.checkbox("🎥 Use Live WebCam", disabled=st.session_state.processing)
    if use_webcam:
        st.session_state.video_path = "0"
        st.caption("📼 Ready: `Live WebCam`")
    elif st.session_state.video_path == "0":
        st.session_state.video_path = None

    if not use_webcam and st.session_state.video_path:
        vname = Path(st.session_state.video_path).name
        st.caption(f"📼 Ready: `{vname}`")
    elif not use_webcam:
        st.caption("📼 No video loaded")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.button(
            "▶ Start",
            on_click=_start_pipeline,
            type="primary",
            disabled=st.session_state.processing or not st.session_state.video_path,
            use_container_width=True,
        )
    with btn_col2:
        st.button(
            "⏹ Stop",
            on_click=_stop_pipeline,
            type="secondary",
            disabled=not st.session_state.processing,
            use_container_width=True,
        )

    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    # Pipeline alive check
    runner: PipelineRunner | None = get_runner()
    if runner and not runner.is_alive() and st.session_state.processing:
        st.session_state.processing = False
        set_runner(None)

st.markdown("---")

# ── Results Tabs ───────────────────────────────────────────────────────────
tab_video, tab_l1, tab_l2, tab_l3, tab_vlm, tab_history = st.tabs([
    "📹 Annotated Video",
    "📊 L1 Motion",
    "🔍 L2 Detections",
    "⚠️ L3 Events",
    "🧠 VLM Analysis",
    "📋 Full History",
])

with tab_video:
    if st.session_state.processing:
        live_img_path = project_root / "outputs" / "live_frame.jpg"
        if live_img_path.exists():
            import time
            # Cache buster to force Streamlit to refresh the image
            st.image(str(live_img_path), caption="Live Streaming (1 FPS)", use_container_width=True)
        else:
            st.info("Starting live stream... please wait.")
    else:
        render_video(OUTPUT_VIDEO)

with tab_l1:
    render_l1_chart(st.session_state.q_l1)

with tab_l2:
    render_live_detections(st.session_state.q_stats)

with tab_l3:
    render_events_live(st.session_state.q_events)

with tab_vlm:
    if not st.session_state.vlm_enabled:
        st.info("VLM is disabled. Enable it in the sidebar to see AI analysis.")
    else:
        render_vlm_cards(st.session_state.q_vlm)

with tab_history:
    st.markdown("#### 📋 Historical Events (from saved files)")
    loader = DataLoader(project_root)
    events_df = loader.get_events()
    render_alert_table(events_df)
    st.markdown("---")
    render_stats(loader.get_stats())

# ── Auto-refresh ───────────────────────────────────────────────────────────
if st.session_state.auto_refresh and st.session_state.processing:
    time.sleep(1.5)
    st.rerun()
