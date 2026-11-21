"""BoxingTracker AI v3 - premium HUD and experimental punch-attempt counter.

Tracks left and right wrist movement relative to shoulder, oriented toward the
opponent. This does NOT classify landed punches and needs manual validation.
"""
import argparse
import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import cv2
import numpy as np

COLORS = [(255, 160, 40), (60, 210, 255)]  # BGR
JOINTS = {"left": (5, 7, 9), "right": (6, 8, 10)}
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
            (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def angle(a, b, c):
    ab = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    cb = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    den = np.linalg.norm(ab) * np.linalg.norm(cb)
    if den <= 1e-7:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ab, cb) / den, -1.0, 1.0))))


def torso_scale(xy, box):
    shoulders = (xy[5] + xy[6]) * 0.5
    hips = (xy[11] + xy[12]) * 0.5
    body_height = np.linalg.norm(shoulders - hips)
    return max(float(body_height), float(box[3] - box[1]) * 0.20, 1.0)


class HandMotion:
    """Debounced forward stroke; count once per extension and require recovery."""

    def __init__(self, min_travel=0.34, min_speed=0.027, recover=0.17):
        self.min_travel = min_travel
        self.min_speed = min_speed
        self.recover = recover
        self.smoothed = None
        self.guard = None
        self.armed = True
        self.last_count = -999
        self.frame = 0
        self.last_speed = 0.0
        self.last_travel = 0.0
        self.hits = 0
        self.missing = 0

    def update(self, forward, elbow_deg, reach, fps=25):
        self.frame += 1
        if forward is None or not np.isfinite(forward):
            self.missing += 1
            # Missing detections do not count and must not cause instant rearming.
            if self.missing > max(5, int(fps * 0.35)):
                self.smoothed = None
                self.guard = None
                self.armed = True
            return False
        self.missing = 0

        if self.smoothed is None:
            self.smoothed = float(forward)
            self.guard = float(forward)
            return False

        previous = self.smoothed
        self.smoothed = 0.66 * float(forward) + 0.34 * previous
        speed = self.smoothed - previous
        self.last_speed = speed

        # Track the fighter's guard position gradually during recovery; do not
        # chase the hand as it extends into a punch.
        if self.armed:
            if self.smoothed <= self.guard:
                self.guard = 0.65 * self.guard + 0.35 * self.smoothed
            elif self.smoothed < self.guard + 0.13:
                self.guard = 0.98 * self.guard + 0.02 * self.smoothed

        travel = max(0.0, self.smoothed - self.guard)
        self.last_travel = travel
        cooldown = (self.frame - self.last_count) >= max(4, int(fps * 0.18))
        valid_arm = elbow_deg >= 82 and reach >= 0.72
        fired = False
        if self.armed and cooldown and valid_arm and travel >= self.min_travel and speed >= self.min_speed:
            self.armed = False
            self.last_count = self.frame
            self.hits += 1
            fired = True

        # A new punch needs a visible hand retraction, not simply elapsed time.
        if not self.armed and travel <= self.recover:
            self.armed = True
            self.guard = self.smoothed
        return fired


class PunchCounter:
    def __init__(self, facing=1, min_travel=0.34):
        self.facing = facing  # +1 if opponent is to the right, else -1
        self.hands = {side: HandMotion(min_travel=min_travel) for side in JOINTS}
        self.counts = defaultdict(int)

    def update(self, xy, confidence, bbox, fps):
        scale = torso_scale(xy, bbox)
        detected = []
        diagnostics = {}
        for side, (shoulder, elbow, wrist) in JOINTS.items():
            valid = all(
                confidence[k] >= 0.20 and np.isfinite(xy[k]).all()
                and np.any(xy[k] != 0)
                for k in (shoulder, elbow, wrist)
            )
            if valid:
                s, e, w = xy[shoulder], xy[elbow], xy[wrist]
                forward = float((w[0] - s[0]) * self.facing / scale)
                ang = angle(s, e, w)
                reach = float(np.linalg.norm(w - s) / scale)
                # Ignore obvious false wrist positions far below the upper body.
                if float(w[1] - s[1]) / scale > 1.15:
                    valid = False
            if not valid:
                forward, ang, reach = None, 0, 0
            counted = self.hands[side].update(forward, ang, reach, fps)
            if counted:
                label = side + "_punch"
                self.counts[label] += 1
                detected.append(label)
            diagnostics[side] = {
                "forward": forward, "travel": self.hands[side].last_travel,
                "speed": self.hands[side].last_speed, "angle": ang,
                "confidence": float(confidence[wrist]) if valid else 0,
                "counted": int(counted),
            }
        return detected, diagnostics


# Colors here are RGB for Pillow. Matching OpenCV colors are declared above.
UI_BLUE = (40, 160, 255)
UI_GOLD = (255, 210, 60)
UI_WHITE = (244, 247, 253)
UI_MUTED = (156, 170, 190)


@lru_cache(maxsize=64)
def ui_font(size, bold=False):
    """Select a clean system font; no external font downloads required."""
    size = max(9, int(size))
    font_paths = (
        ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold else
        ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for path in font_paths:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _blend_overlay(frame, overlay, top):
    """Alpha-composite only a thin image region, not the entire 1440p frame."""
    h, w = overlay.height, overlay.width
    region = frame[top:top + h, :w]
    pil_region = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB)).convert("RGBA")
    pil_region.alpha_composite(overlay)
    region[:] = cv2.cvtColor(np.asarray(pil_region.convert("RGB")), cv2.COLOR_RGB2BGR)


def draw_fighter(frame, xy, confidence, bbox, number, track_id):
    """Colored tracking box, readable ID tag, and slim pose skeleton."""
    color = COLORS[number - 1]
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return

    thickness = max(2, round(w / 900))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # Unlike the earlier plain text, the label has an opaque dark background.
    font_scale = max(0.55, min(1.05, w / 2100))
    label = f"BOXER {number}  |  ID {track_id}"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                         font_scale, 2)
    lp = max(8, round(9 * w / 1920))
    lx = min(x1, max(0, w - tw - 2 * lp - 1))
    tag_height = th + 2 * lp
    ly = y1 - tag_height - 5 if y1 >= tag_height + 5 else y1 + 5
    cv2.rectangle(frame, (lx, ly), (lx + tw + 2 * lp, ly + tag_height),
                  (20, 23, 31), -1)
    cv2.rectangle(frame, (lx, ly), (lx + max(4, lp // 2), ly + tag_height), color, -1)
    cv2.putText(frame, label, (lx + lp, ly + lp + th),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2,
                cv2.LINE_AA)

    for a, b in SKELETON:
        if confidence[a] >= 0.25 and confidence[b] >= 0.25:
            cv2.line(frame, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)),
                     color, thickness, cv2.LINE_AA)
    for i in (5, 6, 7, 8, 9, 10):
        if confidence[i] >= 0.25:
            cv2.circle(frame, tuple(xy[i].astype(int)),
                       max(3, round(w / 540)), (255, 255, 255), -1, cv2.LINE_AA)


def scoreboard(frame, fighters, t, last_event=None, frame_number=0, fps=25):
    """Broadcast-style HUD. All counter values come from the original engine.

    This function affects visuals only; it does not change the counting logic.
    """
    h, w = frame.shape[:2]
    s = max(0.48, min(w / 1920, h / 1080))
    pad = round(24 * s)
    corner = max(7, round(15 * s))

    # Top header: brand, last detected punch (briefly), and elapsed time.
    header_h = min(h, round(91 * s))
    header = Image.new("RGBA", (w, header_h), (0, 0, 0, 0))
    hd = ImageDraw.Draw(header, "RGBA")
    brand_w = round(335 * s)
    pill_h = round(52 * s)
    hd.rounded_rectangle((pad, pad, pad + brand_w, pad + pill_h),
                         radius=corner, fill=(11, 18, 31, 225),
                         outline=(83, 106, 141, 170), width=max(1, round(s)))
    hd.ellipse((pad + round(17*s), pad + round(21*s),
                pad + round(27*s), pad + round(31*s)), fill=UI_BLUE + (255,))
    hd.text((pad + round(37*s), pad + round(12*s)), "BOXINGTRACKER AI",
            font=ui_font(round(23*s), True), fill=UI_WHITE + (255,))

    minutes = int(t // 60)
    seconds = t % 60
    time_text = f"{minutes:02d}:{seconds:04.1f}"
    time_width = round(155*s)
    tx = w - pad - time_width
    hd.rounded_rectangle((tx, pad, tx + time_width, pad + pill_h),
                         radius=corner, fill=(11, 18, 31, 225),
                         outline=(83, 106, 141, 170), width=max(1, round(s)))
    hd.text((tx + round(19*s), pad + round(10*s)), time_text,
            font=ui_font(round(28*s), True), fill=UI_WHITE + (255,))

    if last_event is not None and frame_number - last_event[2] <= max(1, int(fps * 0.55)):
        num, punch, _ = last_event
        caption = f"BOXER {num}  {punch.replace('_', ' ').upper()}"
        alert_w = round(380*s)
        ax = (w - alert_w) // 2
        accent = UI_BLUE if num == 1 else UI_GOLD
        hd.rounded_rectangle((ax, pad, ax + alert_w, pad + pill_h),
                             radius=corner, fill=(11, 18, 31, 235),
                             outline=accent + (255,), width=max(1, round(2*s)))
        hd.text((ax + round(21*s), pad + round(12*s)), caption,
                font=ui_font(round(20*s), True), fill=UI_WHITE + (255,))
    _blend_overlay(frame, header, 0)

    # Bottom: separate cards with very large LEFT / RIGHT / TOTAL counts.
    panel_h = round(174*s)
    y_top = max(0, h - pad - panel_h)
    footer = Image.new("RGBA", (w, h - y_top), (0, 0, 0, 0))
    fd = ImageDraw.Draw(footer, "RGBA")
    gap = round(16*s)
    card_w = (w - 2*pad - gap) // 2

    for num in (1, 2):
        x = pad + (num - 1) * (card_w + gap)
        accent = UI_BLUE if num == 1 else UI_GOLD
        fd.rounded_rectangle((x, 0, x + card_w, panel_h), radius=corner,
                             fill=(11, 18, 31, 231),
                             outline=accent + (170,), width=max(1, round(2*s)))
        fd.rounded_rectangle((x + round(16*s), round(15*s),
                              x + round(23*s), round(45*s)),
                             radius=max(2, round(3*s)), fill=accent + (255,))
        fd.text((x + round(37*s), round(12*s)), f"BOXER {num}",
                font=ui_font(round(24*s), True), fill=accent + (255,))

        obj = fighters.get(num)
        counts = obj["counter"].counts if obj else {}
        left = int(counts.get("left_punch", 0))
        right = int(counts.get("right_punch", 0))
        total = left + right

        metric_y = round(69*s)
        fd.text((x + round(25*s), metric_y), "LEFT HAND",
                font=ui_font(round(17*s), True), fill=UI_MUTED + (255,))
        fd.text((x + round(202*s), metric_y), "RIGHT HAND",
                font=ui_font(round(17*s), True), fill=UI_MUTED + (255,))
        fd.text((x + round(23*s), round(91*s)), f"{left:02d}",
                font=ui_font(round(58*s), True), fill=UI_WHITE + (255,))
        fd.text((x + round(200*s), round(91*s)), f"{right:02d}",
                font=ui_font(round(58*s), True), fill=UI_WHITE + (255,))

        # Put TOTAL next to the right border so its number pops in a preview.
        total_x = x + card_w - round(203*s)
        separator_x = total_x - round(29*s)
        fd.line((separator_x, round(61*s), separator_x, round(150*s)),
                fill=(88, 104, 130, 185), width=max(1, round(2*s)))
        fd.text((total_x, metric_y), "TOTAL PUNCHES",
                font=ui_font(round(16*s), True), fill=UI_MUTED + (255,))
        fd.text((total_x, round(88*s)), f"{total:02d}",
                font=ui_font(round(67*s), True), fill=accent + (255,))

    _blend_overlay(frame, footer, y_top)
    return frame


def parse_crop(value, width, height):
    if not value:
        return 0, 0, width, height
    try:
        x1, y1, x2, y2 = [int(v) for v in value.split(",")]
    except ValueError as e:
        raise SystemExit("--crop must be x1,y1,x2,y2") from e
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise SystemExit(f"Invalid crop for {width}x{height} video")
    return x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser(description="BoxingTracker AI - two boxer punch counting prototype")
    parser.add_argument("--source", default="videos/boxing.mp4")
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="Pose inference resolution (960 recommended for small fighters)")
    parser.add_argument("--min-travel", type=float, default=0.34,
                        help="Minimum forward wrist displacement in torso lengths")
    parser.add_argument("--crop", default="", help="Optional pixel crop x1,y1,x2,y2")
    parser.add_argument("--output-dir", default="outputs", help="Where to save the processed video and CSVs")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--preview-width", type=int, default=960)
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    source = int(args.source) if args.source.isdecimal() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps < 1:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x1, y1, x2, y2 = parse_crop(args.crop, width, height)
    out_w, out_h = x2-x1, y2-y1
    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(out / "annotated_boxing.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit("Could not create output video")

    fighters = {}
    track_to_fighter = {}
    events, metrics = [], []
    frame_number = 0
    last_event = None
    try:
        while True:
            ok, original = cap.read()
            if not ok:
                break
            frame = original[y1:y2, x1:x2].copy()
            result = model.track(frame, persist=True, tracker=args.tracker,
                                 conf=0.25, imgsz=args.imgsz, verbose=False)[0]
            found = []
            if result.boxes is not None and result.boxes.id is not None and result.keypoints is not None:
                ids = result.boxes.id.int().cpu().tolist()
                boxes = result.boxes.xyxy.cpu().numpy()
                keypoints = result.keypoints.xy.cpu().numpy()
                confidences = (result.keypoints.conf.cpu().numpy()
                               if result.keypoints.conf is not None
                               else np.ones((len(ids), 17), dtype=float))
                for tid, box, xy, conf in zip(ids, boxes, keypoints, confidences):
                    if len(xy) < 17:
                        continue
                    area = max(0., (box[2]-box[0]) * (box[3]-box[1]))
                    found.append((int(tid), box, xy, conf, area))
            if not fighters and len(found) >= 2:
                top = sorted(found, key=lambda x: x[4], reverse=True)[:2]
                top.sort(key=lambda x: (x[1][0] + x[1][2]) / 2)
                for num, (tid, box, xy, conf, _) in enumerate(top, start=1):
                    fighters[num] = {"track_id": tid,
                                     "counter": PunchCounter(facing=1 if num == 1 else -1,
                                                             min_travel=args.min_travel)}
                    track_to_fighter[tid] = num
                print("Initial two boxers (track ID -> boxer):", track_to_fighter)

            for tid, box, xy, conf, _ in found:
                num = track_to_fighter.get(tid)
                if num is None:
                    continue
                draw_fighter(frame, xy, conf, box, num, tid)
                punches, stats = fighters[num]["counter"].update(xy, conf, box, fps)
                for punch in punches:
                    t = round(frame_number / fps, 3)
                    events.append({"time_seconds": t, "boxer": num,
                                   "tracker_id": tid, "punch": punch})
                    print(f"{t:6.2f}s  Boxer {num}: {punch}")
                    last_event = (num, punch, frame_number)
                for hand, state in stats.items():
                    metrics.append({"time_seconds": round(frame_number/fps, 3),
                                    "boxer": num, "hand": hand,
                                    **{k: (round(v, 4) if v is not None else "")
                                       for k, v in state.items()}})
            scoreboard(frame, fighters, frame_number / fps,
                       last_event=last_event, frame_number=frame_number, fps=fps)
            writer.write(frame)
            if args.show:
                preview_width = max(240, args.preview_width)
                preview_height = max(1, round(out_h * preview_width / out_w))
                preview = cv2.resize(frame, (preview_width, preview_height))
                cv2.imshow("BoxingTracker AI", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_number += 1
            if frame_number % 150 == 0:
                print(f"Processed {frame_number} frames")
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    with (out / "punch_events.csv").open("w", encoding="utf-8", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["time_seconds", "boxer", "tracker_id", "punch"])
        writer_csv.writeheader()
        writer_csv.writerows(events)
    with (out / "punch_debug.csv").open("w", encoding="utf-8", newline="") as f:
        cols = ["time_seconds", "boxer", "hand", "forward", "travel", "speed", "angle", "confidence", "counted"]
        writer_csv = csv.DictWriter(f, fieldnames=cols)
        writer_csv.writeheader()
        writer_csv.writerows(metrics)
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        cols = ["boxer", "tracker_id", "left_punch", "right_punch", "total"]
        writer_csv = csv.DictWriter(f, fieldnames=cols)
        writer_csv.writeheader()
        for num, fighter in sorted(fighters.items()):
            c = fighter["counter"].counts
            writer_csv.writerow({"boxer": num, "tracker_id": fighter["track_id"],
                                "left_punch": c["left_punch"], "right_punch": c["right_punch"],
                                "total": c["left_punch"] + c["right_punch"]})
    print("Finished! Outputs: annotated_boxing.mp4, punch_events.csv, punch_debug.csv, summary.csv")
    print("IMPORTANT: Heuristic punch-attempt counts are not validated. Verify manually.")


if __name__ == "__main__":
    main()
