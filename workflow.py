"""Safe local video-upload workflow for BoxingTracker AI.

No Streamlit dependency. The existing v3 detection engine remains in main.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import shutil
import subprocess
import sys
import uuid

import cv2

MAX_UPLOAD_BYTES = 150 * 1024 * 1024
MAX_DURATION_SECONDS = 60.0
REQUIRED_RESULTS = ("punch_events.csv", "punch_debug.csv", "summary.csv", "annotated_boxing.mp4")


@dataclass(frozen=True)
class ClipInfo:
    duration: float
    fps: float
    frames: int
    width: int
    height: int


@dataclass(frozen=True)
class ProcessResult:
    ok: bool
    output_dir: Path
    message: str
    log: str


def inspect_clip(path: Path, max_duration: float = MAX_DURATION_SECONDS) -> ClipInfo:
    """Fail quickly for empty, corrupt, excessively long, or unrealistic videos."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("The uploaded video is empty or missing.")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("Video exceeds the 150 MB upload limit.")
    if path.suffix.lower() != ".mp4":
        raise ValueError("Please upload an MP4 video.")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("OpenCV cannot read this MP4. Try re-encoding it as H.264.")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        can_read, first_frame = cap.read()
    finally:
        cap.release()

    if not (math.isfinite(fps) and 1 <= fps <= 240 and frames > 0
            and width >= 160 and height >= 120 and can_read and first_frame is not None):
        raise ValueError("Invalid or unsupported video: cannot decode frames, dimensions, or frame rate.")
    duration = frames / fps
    if duration <= 0 or duration > max_duration:
        raise ValueError(f"Clip is {duration:.1f}s long. Please choose a clip up to {int(max_duration)} seconds.")
    return ClipInfo(duration, fps, frames, width, height)


def _manifest(run_dir: Path, **changes) -> None:
    path = run_dir / "run.json"
    values = {}
    if path.is_file():
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            values = {}
    values.update(changes)
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")


def save_upload(upload, runs_root: Path) -> tuple[Path, ClipInfo]:
    """Copy a Streamlit UploadedFile (or a named binary file-like object) safely."""
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    run_dir = runs_root / run_id
    run_dir.mkdir(exist_ok=False)
    dest = run_dir / "input.mp4"  # Never use the untrusted uploaded filename as a path.
    try:
        upload.seek(0)
        with dest.open("wb") as file_out:
            shutil.copyfileobj(upload, file_out, length=1024 * 1024)
        info = inspect_clip(dest)
        _manifest(run_dir, run_id=run_id, original_filename=Path(getattr(upload, "name", "video.mp4")).name,
                  status="uploaded", created_at=datetime.now(timezone.utc).isoformat(),
                  duration_seconds=round(info.duration, 2), fps=round(info.fps, 2),
                  dimensions=f"{info.width}x{info.height}")
        return run_dir, info
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise


def process_run(run_dir: Path, project_root: Path, inference_size: int = 960,
                min_travel: float = 0.34, timeout_seconds: int = 1800,
                python_exe: str | None = None) -> ProcessResult:
    """Run the unchanged punch detector through a shell-free subprocess."""
    run_dir = Path(run_dir)
    project_root = Path(project_root)
    if inference_size not in (640, 960, 1280):
        raise ValueError("Inference size must be 640, 960, or 1280.")
    if not 0.20 <= min_travel <= 0.60:
        raise ValueError("Minimum travel must be within 0.20 and 0.60.")
    video_path = run_dir / "input.mp4"
    inspect_clip(video_path)
    output_dir = run_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    cmd = [python_exe or sys.executable, str(project_root / "main.py"),
           "--source", str(video_path), "--output-dir", str(output_dir),
           "--imgsz", str(inference_size), "--min-travel", str(min_travel)]
    _manifest(run_dir, status="processing", inference_size=inference_size, min_travel=min_travel)
    try:
        result = subprocess.run(cmd, cwd=str(project_root), capture_output=True,
                                text=True, timeout=timeout_seconds, shell=False)
        log = (result.stdout or "") + "\n" + (result.stderr or "")
        complete = all((output_dir / file).is_file() for file in REQUIRED_RESULTS)
        if result.returncode != 0:
            message = "Detection failed. See the process log for details."
        elif not complete:
            message = "Detection returned success but did not produce all expected results."
        else:
            message = "Processing completed successfully."
        ok = result.returncode == 0 and complete
        _manifest(run_dir, status="complete" if ok else "failed")
        return ProcessResult(ok, output_dir, message, log[-8000:])
    except subprocess.TimeoutExpired:
        _manifest(run_dir, status="failed")
        return ProcessResult(False, output_dir, "Processing timed out. Try a shorter clip or lower inference size.", "Timeout")
    except OSError as exc:
        _manifest(run_dir, status="failed")
        return ProcessResult(False, output_dir, "Could not start the Python detection process.", str(exc))


def recent_analyses(runs_root: Path, limit: int = 15) -> list[tuple[str, Path]]:
    """Find successful saved runs across Streamlit restarts (newest first)."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    options = []
    for run in sorted(runs_root.iterdir(), key=lambda p: p.name, reverse=True):
        out = run / "outputs"
        if not run.is_dir() or not all((out / name).is_file() for name in REQUIRED_RESULTS):
            continue
        title = run.name
        try:
            meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
            friendly = str(meta.get("original_filename", "video.mp4"))
            if len(friendly) > 21:
                friendly = friendly[:18] + "..."
            title = f"{run.name[:13]} · {friendly}"
        except (OSError, ValueError):
            title = f"Run · {run.name[:18]}"
        options.append((title, out))
        if len(options) >= limit:
            break
    return options
