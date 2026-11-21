"""Browser-compatible video with safe space for Streamlit's player controls."""
from __future__ import annotations
from pathlib import Path
import shutil
import subprocess
import cv2


def video_info(path: Path) -> tuple[float, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0, 0, 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return (frames / fps if fps > 0 else 0.0, frames, fps)
    finally:
        cap.release()


def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, FileNotFoundError):
        return None


def ensure_player_video(path: Path) -> tuple[Path, str | None]:
    """Make a cached H.264 copy with bottom padding for native playback controls.

    This changes ONLY browser playback, not the v3 model, punch counts, or original export.
    """
    path = Path(path)
    if path.name.endswith("_player.mp4"):
        return path, None
    converted = path.with_name(path.stem.replace("_h264", "") + "_player.mp4")
    if (converted.is_file() and converted.stat().st_size > 0
            and converted.stat().st_mtime >= path.stat().st_mtime):
        return converted, None
    exe = find_ffmpeg()
    if not exe:
        return path, "FFmpeg not found; video may not play in your browser. Install imageio-ffmpeg."
    # Controls cover the BOTTOM of native video elements. Preserve our bottom HUD
    # by placing 88px of unobtrusive dark padding underneath the annotated frame.
    filter_video = "scale='min(1280,iw)':-2,pad=iw:ih+88:0:0:color=0x0B1627"
    cmd = [exe, "-nostdin", "-y", "-v", "error", "-i", str(path), "-an", "-vf", filter_video,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(converted)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=360, shell=False)
        if result.returncode != 0 or not converted.is_file() or converted.stat().st_size == 0:
            converted.unlink(missing_ok=True)
            return path, f"H.264 conversion failed: {(result.stderr or '')[-500:]}"
        return converted, None
    except (OSError, subprocess.TimeoutExpired) as exc:
        converted.unlink(missing_ok=True)
        return path, f"H.264 conversion failed: {exc}"


def ensure_h264(path: Path) -> tuple[Path, str | None]:
    """Backward-compatible name used by older dashboard code."""
    return ensure_player_video(path)
