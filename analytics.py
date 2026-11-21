"""Pure data helpers for BoxingTracker AI. No Streamlit or model dependency."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import pandas as pd

BLUE = "#4BA6FF"
GOLD = "#F6C45E"


@dataclass
class Analysis:
    folder: Path
    events: pd.DataFrame
    debug: pd.DataFrame
    summary: pd.DataFrame
    duration: float


def _read_csv(file: Path, expected: set[str]) -> pd.DataFrame:
    if not file.is_file():
        raise FileNotFoundError(f"Missing {file.name} in {file.parent}")
    df = pd.read_csv(file)
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"{file.name} is missing required columns: {', '.join(sorted(missing))}")
    return df


def load_analysis(folder: Path, duration: float | None = None) -> Analysis:
    folder = Path(folder)
    e = _read_csv(folder / "punch_events.csv", {"time_seconds", "boxer", "punch"})
    d = _read_csv(folder / "punch_debug.csv", {"time_seconds", "boxer", "hand", "confidence", "counted"})
    s = _read_csv(folder / "summary.csv", {"boxer", "left_punch", "right_punch", "total"})

    e = e.copy()
    e["time_seconds"] = pd.to_numeric(e["time_seconds"], errors="raise")
    e["boxer"] = pd.to_numeric(e["boxer"], errors="raise").astype(int)
    e["hand"] = e["punch"].map({"left_punch": "left", "right_punch": "right"})
    if e["hand"].isna().any() or not e["boxer"].isin([1, 2]).all():
        raise ValueError("Events contain unknown boxer or punch labels")
    e = e.sort_values("time_seconds").reset_index(drop=True)
    e["event_id"] = [f"E{i:03d}" for i in range(1, len(e) + 1)]

    d = d.copy()
    d["time_seconds"] = pd.to_numeric(d["time_seconds"], errors="coerce")
    d["boxer"] = pd.to_numeric(d["boxer"], errors="coerce")
    d["counted"] = pd.to_numeric(d["counted"], errors="coerce").fillna(0)
    d["confidence"] = pd.to_numeric(d["confidence"], errors="coerce")

    s = s.copy()
    for col in ["boxer", "left_punch", "right_punch", "total"]:
        s[col] = pd.to_numeric(s[col], errors="raise").astype(int)
    s = s.sort_values("boxer").reset_index(drop=True)
    if not set(s["boxer"]).issubset({1, 2}):
        raise ValueError("Summary contains unknown boxer IDs")

    counted = d.loc[d["counted"] == 1, ["time_seconds", "boxer", "hand", "confidence"]].copy()
    # Keep confidence aligned with the actual counted wrist when diagnostic rows exist.
    e = e.merge(counted, on=["time_seconds", "boxer", "hand"], how="left")

    if duration is None or duration <= 0:
        duration = max(float(d["time_seconds"].max()) if not d.empty else 0.0,
                       float(e["time_seconds"].max()) if not e.empty else 0.0) + 0.04
    return Analysis(folder, e, d, s, float(duration))


def summary_count(a: Analysis, boxer: int, hand: str) -> int:
    rows = a.summary.loc[a.summary["boxer"] == boxer]
    return int(rows.iloc[0][f"{hand}_punch"]) if not rows.empty else 0


def event_timeline(events: pd.DataFrame, duration: float) -> pd.DataFrame:
    """One row per second, with zeros for idle intervals."""
    n = max(1, math.ceil(duration))
    rows = [{"second": second, "boxer": boxer, "count": 0} for second in range(n) for boxer in (1, 2)]
    result = pd.DataFrame(rows)
    if events.empty:
        return result
    e = events.assign(second=events["time_seconds"].astype(float).floordiv(1).astype(int))
    counts = e.groupby(["second", "boxer"]).size().to_dict()
    result["count"] = [int(counts.get((r.second, r.boxer), 0)) for r in result.itertuples()]
    return result


def normalized_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    missing = {"time_seconds", "boxer", "hand"}.difference(df.columns)
    if missing:
        raise ValueError("Manual labels need columns: time_seconds, boxer, hand")
    if df.empty:
        return pd.DataFrame(columns=["time_seconds", "boxer", "hand"])
    out = df[["time_seconds", "boxer", "hand"]].copy()
    out = out.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")
    if out.isna().any().any():
        raise ValueError("All manual-label rows must have time_seconds, boxer and hand")
    out["time_seconds"] = pd.to_numeric(out["time_seconds"], errors="raise")
    out["boxer"] = pd.to_numeric(out["boxer"], errors="raise").astype(int)
    out["hand"] = out["hand"].astype(str).str.strip().str.lower().str.replace("_punch", "", regex=False)
    if (~out["boxer"].isin([1, 2])).any() or (~out["hand"].isin(["left", "right"])).any():
        raise ValueError("Boxer must be 1/2 and hand must be left/right")
    if (~out["time_seconds"].ge(0)).any() or (~out["time_seconds"].map(math.isfinite)).any():
        raise ValueError("Punch timestamps must be valid nonnegative seconds")
    return out.reset_index(drop=True)


def compare_with_ground_truth(predicted: pd.DataFrame, ground_truth: pd.DataFrame,
                              tolerance: float = 0.24) -> dict:
    """One-to-one timestamp matching (same boxer and hand, within tolerance)."""
    gt = normalized_ground_truth(ground_truth)
    pred = normalized_ground_truth(predicted)
    if not (0.0 < tolerance <= 2):
        raise ValueError("Tolerance must be > 0 and <= 2 seconds")

    # Pair closest matching detections first, without reusing an event.
    candidates = []
    for pi, p in pred.iterrows():
        for gi, g in gt.iterrows():
            if p["boxer"] == g["boxer"] and p["hand"] == g["hand"]:
                delta = abs(float(p["time_seconds"] - g["time_seconds"]))
                if delta <= tolerance + 1e-9:
                    candidates.append((delta, pi, gi))
    selected_p = set()
    selected_g = set()
    pairs = []
    for delta, pi, gi in sorted(candidates):
        if pi in selected_p or gi in selected_g:
            continue
        selected_p.add(pi)
        selected_g.add(gi)
        pairs.append({"predicted_time": float(pred.at[pi, "time_seconds"]),
                      "manual_time": float(gt.at[gi, "time_seconds"]),
                      "boxer": int(pred.at[pi, "boxer"]),
                      "hand": str(pred.at[pi, "hand"]),
                      "difference_seconds": round(delta, 3)})

    tp = len(pairs)
    fp = len(pred) - tp
    fn = len(gt) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    missed = gt.drop(index=list(selected_g)).reset_index(drop=True)
    unmatched = pred.drop(index=list(selected_p)).reset_index(drop=True)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": f1, "matches": pd.DataFrame(pairs), "missed": missed,
            "unmatched_predictions": unmatched}
