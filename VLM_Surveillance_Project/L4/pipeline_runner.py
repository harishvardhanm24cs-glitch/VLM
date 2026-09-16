"""
L4/pipeline_runner.py
---------------------
Runs the SurveillancePipeline in a background daemon thread and funnels
all callback data into thread-safe lists stored in Streamlit's session state.
"""
import sys
import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Global state to prevent Streamlit from deepcopying unpicklable objects (like Models/Threads)
_global_state = {}

def set_runner(runner):
    _global_state["runner"] = runner

def get_runner():
    return _global_state.get("runner")

# Ensure project root is importable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class PipelineRunner:
    """
    Wraps SurveillancePipeline so it can be driven from Streamlit.

    Parameters
    ----------
    config_path : Path
        Path to config/settings.yaml.
    video_path : str
        Absolute path to the video file to process (uploaded by the user).
    state_queues : dict
        Shared dict with lists that receive streaming data:
            - state_queues["stats"]   : list of per-frame stats dicts
            - state_queues["l1"]      : list of L1 motion result dicts
            - state_queues["events"]  : list of L3 event dicts
            - state_queues["vlm"]     : list of VLM analysis dicts
            - state_queues["status"]  : list of status strings ("RUNNING"/"DONE"/"ERROR")
            - state_queues["error"]   : list of error message strings
    """

    def __init__(self, config_path: Path, video_path: str, state_queues: dict,
                 vlm_enabled: bool = True):
        self.config_path  = config_path
        self.video_path   = video_path
        self.state_queues = state_queues
        self.vlm_enabled  = vlm_enabled
        self._stop_event  = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Spin up the pipeline thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="SurveillancePipeline"
        )
        self._thread.start()
        logger.info(f"Pipeline thread started for video: {self.video_path}")

    def stop(self):
        """Signal the pipeline to stop gracefully."""
        self._stop_event.set()
        logger.info("Stop signal sent to pipeline.")

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _push(self, key: str, value):
        """Thread-safe append into a session-state queue."""
        q = self.state_queues.get(key)
        if q is not None:
            q.append(value)

    def _run(self):
        try:
            # Late import so Streamlit doesn't need to import heavy models
            # at page-load time.
            from main import SurveillancePipeline
            import yaml

            # --- Inject runtime settings (e.g. vlm toggle from UI) ---
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            cfg.setdefault("VLM", {})["enabled"] = self.vlm_enabled

            # Write a temporary in-memory override by subclassing
            self._push("status", "INITIALIZING")
            pipeline = SurveillancePipeline.__new__(SurveillancePipeline)
            pipeline.project_root = self.config_path.parent.parent
            pipeline.config = cfg
            pipeline._setup_directories()
            pipeline.l1_results = []
            pipeline.frame_cache = {}
            pipeline._init_subsystems()

            self._push("status", "RUNNING")
            
            # Delete old output video so it doesn't show up
            output_video_path = Path(pipeline.project_root) / "outputs" / "detections" / "tracking_annotated.webm"
            if output_video_path.exists():
                try:
                    output_video_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete old output video: {e}")

            # Monkey-patch _init_subsystems to intercept L1 results
            original_save_l1 = pipeline.save_l1_results

            def patched_save_l1():
                original_save_l1()
                for rec in pipeline.l1_results:
                    self._push("l1", rec)

            pipeline.save_l1_results = patched_save_l1

            pipeline.run(
                on_stats=lambda s: self._push("stats", s),
                on_event=lambda e: self._push("events", e),
                on_vlm=lambda v: self._push("vlm", v),
                video_path_override=self.video_path,
                stop_event=self._stop_event,
            )

            self._push("status", "DONE")
            logger.info("Pipeline finished successfully.")
            
            # Delete the uploaded video after task finishes
            if self.video_path and str(self.video_path) != "0":
                in_path = Path(self.video_path)
                if in_path.exists() and "uploaded_" in in_path.name:
                    try:
                        in_path.unlink()
                        logger.info(f"Deleted uploaded video: {in_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete uploaded video: {e}")

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"Pipeline crashed: {error_msg}", exc_info=True)
            self._push("error", error_msg)
            self._push("status", "ERROR")
