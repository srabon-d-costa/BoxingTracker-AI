"""BoxingTracker AI | Streamlit analytics and manual validation dashboard.

Start with: streamlit run app.py
The bundled demo is from the user's v3 detection output. Events are NOT ground truth.
"""
from __future__ import annotations

from pathlib import Path
import os
import hashlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import BLUE, GOLD, load_analysis, summary_count, event_timeline, normalized_ground_truth, compare_with_ground_truth
from media import ensure_player_video, video_info
from workflow import save_upload, process_run, recent_analyses, MAX_UPLOAD_BYTES

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "outputs" / "demo"
LOCAL = ROOT / "outputs"
RUNS = ROOT / "runs"
FIGHTER_COLORS = {1: BLUE, 2: GOLD}

# In public mode, only show video uploads from this browser session.
PUBLIC_MODE = os.environ.get("BOXINGTRACKER_PUBLIC_MODE", "0").lower() in ("1", "true", "yes")

st.set_page_config(page_title="BoxingTracker AI | Fight Intelligence", page_icon="🥊", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [data-testid='stApp'] {font-family: 'DM Sans', 'Segoe UI', sans-serif;}
[data-testid='stAppViewContainer'] {background: radial-gradient(circle at 72% -12%, #123054 0, #0B1526 27%, #080D17 64%);}
[data-testid='stSidebar'] {background: #0D1626; border-right: 1px solid #273448;}
.block-container {max-width: 1530px; padding-top: 1.1rem; padding-bottom: 2.3rem;}
[data-testid='stSidebar'] {min-width: 332px !important;}
[data-testid='stSidebar'] [data-testid='stSelectbox'] {max-width: 100%;}
[data-testid='stSidebar'] [data-testid='stMarkdownContainer'] p {overflow-wrap: anywhere;}
[data-testid='stVideo'] video {border-radius: 12px; border: 1px solid #233E5B;}
header[data-testid='stHeader'] {background: transparent;}
.stTabs [data-baseweb='tab-list'] {gap: 14px; border-bottom: 1px solid #26344A;}
.stTabs [data-baseweb='tab'] {font-weight: 650; font-size: 0.91rem; padding: 12px 14px; color: #9CADC5;}
.stTabs [aria-selected='true'] {color: #EAF3FF !important;}
.stButton>button[kind='primary'], [data-testid='stDownloadButton']>button[kind='primary'] {background: linear-gradient(100deg,#168AFF,#3EAAF5); border: 0; color: white; font-weight: 700;}
.stButton>button {border-radius: 12px;}
[data-testid='stFileUploader'] {border-radius: 14px;}
.hero {position: relative; overflow: hidden; border-radius: 20px; padding: 22px 27px; margin-bottom: 18px;
  border: 1px solid #304765; background: linear-gradient(110deg, #142943, #0C1A30 59%, #122133);}
.hero:before {content: ''; position: absolute; right: -8%; top: -160px; height: 330px; width: 330px;
 border: 1px solid #3B5778; border-radius: 50%; box-shadow: 0 0 0 55px #1F385322, 0 0 0 113px #1F385318;}
.eyebrow {color: #75C5FF; font-size: 0.73rem; font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 8px;}
.hero-title {font-family: 'Space Grotesk', 'Segoe UI', sans-serif; font-weight: 700; font-size: clamp(1.9rem, 4.0vw, 3.1rem); letter-spacing: -0.06em; line-height: 1.04; color: #F5F8FF;}
.hero-title span {color: #5DB8FF;}
.hero-desc {color: #ABBDD4; font-size: 0.96rem; max-width: 770px; margin-top: 13px; margin-bottom: 10px;}
.status {display: inline-block; margin-top: 12px; border: 1px solid #4B6884; border-radius: 999px; padding: 5px 12px;
 color: #B4D6F3; background: #153455; font-size: 0.75rem; font-weight: 700;}
.kpi {height: 123px; border-radius: 17px; border: 1px solid #2A3B55; background: linear-gradient(153deg,#18273B,#0F192A); padding: 19px 19px 14px; margin-bottom: 13px;}
.kpi .label {text-transform: uppercase; font-size: 0.69rem; letter-spacing: 0.13em; color: #91A7C0; font-weight: 700;}
.kpi .value {font-family: 'Space Grotesk', 'Segoe UI', sans-serif; font-size: 2.7rem; font-weight: 700; line-height: 1.25; letter-spacing: -0.045em; margin-top: 2px;}
.kpi .foot {font-size: 0.75rem; color: #92A7BF;}
.sect {font-family: 'Space Grotesk', 'Segoe UI', sans-serif; color: #EBF2FD; font-weight: 700; font-size: 1.25rem; letter-spacing: -.025em; margin: 12px 0 10px;}
.fighter {border: 1px solid #2B3C54; border-radius: 16px; padding: 20px 23px; background: #121F31; margin-bottom: 10px;}
.fighter .name {font-weight: 700; font-size: 0.89rem; letter-spacing: .08em;}
.fighter .total {font-family: 'Space Grotesk', sans-serif; font-size: 3.5rem; font-weight: 700; line-height: 1.03;}
.fighter .split {color: #A2B4CB; font-size: 0.9rem; margin-top: 4px;}
.note {border-left: 3px solid #E6B35D; background: #33281388; border-radius: 5px; padding: 11px 14px; color: #E6D3AD; font-size: .85rem; margin: 13px 0 15px;}
small, .muted {color: #92A9C4;}
hr {border-color: #243348 !important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="sect">{title}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, foot: str, color: str = "#F6F8FF"):
    st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value" style="color:{color}">{value}</div><div class="foot">{foot}</div></div>', unsafe_allow_html=True)


def fighter_card(num: int, left: int, right: int):
    color = FIGHTER_COLORS[num]
    st.markdown(f'''<div class="fighter" style="border-top:3px solid {color}">
    <div class="name" style="color:{color}">BOXER 0{num} &nbsp; / &nbsp; PUNCH DETECTIONS</div>
    <div class="total" style="color:{color}">{left + right:02d}</div>
    <div class="split">LEFT HAND &nbsp; <strong style="color:#EEF5FF">{left:02d}</strong> &nbsp;&nbsp;&nbsp;
    RIGHT HAND &nbsp; <strong style="color:#EEF5FF">{right:02d}</strong></div>
    </div>''', unsafe_allow_html=True)


def style_chart(fig: go.Figure, height: int = 340):
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=20, r=15, t=23, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C9D8E9", family="Arial", size=12),
        hoverlabel=dict(bgcolor="#172A41", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0.01),
        xaxis=dict(showline=False, zeroline=False, gridcolor="#293A4D"),
        yaxis=dict(showline=False, zeroline=False, gridcolor="#293A4D"),
    )
    return fig


def show_video(path: Path | None, start: float = 0.0):
    """Play with a dedicated controls strip, leaving the in-video HUD uncovered."""
    if path is None or not path.exists():
        st.info("No processed video found for this dataset. The CSV charts are still available.")
        return
    target, warn = ensure_player_video(path)
    if warn:
        st.warning(warn)
    st.video(str(target), start_time=int(max(0, start)))


def chart_activity(events: pd.DataFrame, duration: float):
    """Unsmoothed, actual event counts in one-second bins. No interpolated punches."""
    activity = event_timeline(events, duration)
    fig = go.Figure()
    for num in (1, 2):
        d = activity.loc[activity["boxer"] == num]
        fig.add_trace(go.Bar(
            x=d["second"], y=d["count"], name=f"Boxer {num}",
            marker_color=FIGHTER_COLORS[num], opacity=0.92,
            hovertemplate="Bin starts at %{x}s<br>Detections: %{y}<extra>Boxer " + str(num) + "</extra>"
        ))
    fig.update_layout(barmode="group", bargap=0.22, bargroupgap=0.06)
    fig.update_xaxes(title="Fight time — 1-second bins", dtick=1,
                     range=[-0.6, max(0.5, int(max(1, duration - 0.001)) + 0.6)])
    fig.update_yaxes(title="Detected punches", rangemode="tozero", dtick=1)
    return style_chart(fig, 324)


def chart_strike_mix(a):
    fig = go.Figure()
    for hand, color in (("left", "#70C8FF"), ("right", "#9DA7FF")):
        fig.add_trace(go.Bar(x=["Boxer 1", "Boxer 2"],
                             y=[summary_count(a, i, hand) for i in (1, 2)],
                             name=f"{hand.title()} hand", marker_color=color,
                             text=[summary_count(a, i, hand) for i in (1, 2)], textposition="auto"))
    fig.update_layout(barmode="group", bargap=0.36)
    fig.update_yaxes(title="Estimated punches", rangemode="tozero", dtick=2)
    return style_chart(fig, 325)


def chart_event_lanes(events: pd.DataFrame, duration: float):
    lanes = [(1, "left", "B1 · LEFT"), (1, "right", "B1 · RIGHT"),
             (2, "left", "B2 · LEFT"), (2, "right", "B2 · RIGHT")]
    fig = go.Figure()
    for num, hand, title in lanes:
        data = events[(events["boxer"] == num) & (events["hand"] == hand)]
        fig.add_trace(go.Scatter(
            x=data["time_seconds"], y=[title]*len(data),
            mode="markers", name=title, marker=dict(size=12, color=FIGHTER_COLORS[num],
            symbol="circle" if hand == "left" else "diamond", line=dict(width=1.5, color="#0D1625")),
            customdata=data["event_id"],
            hovertemplate="%{customdata} | %{x:.2f}s<extra>" + title + "</extra>"))
    fig.update_xaxes(title="Timestamp (seconds)", range=[-0.3, max(1, duration + 0.3)])
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=[v[2] for v in lanes[::-1]], showgrid=False)
    return style_chart(fig, 350)


def datasets() -> dict[str, Path]:
    options: dict[str, Path] = {}
    if all((DEMO / name).is_file() for name in ("punch_events.csv", "punch_debug.csv", "summary.csv")):
        options["Sample demo · v3"] = DEMO
    if all((LOCAL / name).is_file() for name in ("punch_events.csv", "punch_debug.csv", "summary.csv")):
        options["My local results"] = LOCAL
    if PUBLIC_MODE:
        # Prevent viewers from browsing OTHER visitors' uploaded analyses.
        for title, location in st.session_state.get("my_uploaded_runs", []):
            if all((location / name).is_file() for name in ("punch_events.csv", "punch_debug.csv", "summary.csv")):
                options[title] = location
    else:
        for title, location in recent_analyses(RUNS):
            options[title] = location
    return options


def get_video(folder: Path) -> Path | None:
    """Return the latest source rather than a stale transcoded copy."""
    raw = folder / "annotated_boxing.mp4"
    player = folder / "annotated_boxing_player.mp4"
    h264 = folder / "annotated_boxing_h264.mp4"
    if raw.is_file() and (not player.is_file() or raw.stat().st_mtime > player.stat().st_mtime):
        return raw
    if player.is_file():
        return player
    if h264.is_file():
        return h264
    return raw if raw.is_file() else None


options = datasets()
if not options:
    st.error("No demo/results found. Extract the entire ZIP, including outputs/demo/.")
    st.stop()
# Change the selected dataset only BEFORE creating the associated Streamlit widget.
if "pending_source" in st.session_state:
    st.session_state["dataset_selection"] = st.session_state.pop("pending_source")
if st.session_state.get("dataset_selection") not in options:
    st.session_state["dataset_selection"] = next(iter(options))

with st.sidebar:
    st.markdown("### 🥊 BOXINGTRACKER AI")
    st.caption("Computer Vision · Two-Boxer Analytics")
    st.divider()
    source = st.selectbox("Analysis dataset", list(options), key="dataset_selection")
    folder = options[source]
    st.caption("YOLO11 Pose + ByteTrack · v3 detection baseline")
    st.markdown("**Project status**")
    st.caption("✓ Two-boxer pose tracking\n\n✓ Left/right punch attempts\n\n✓ Analytics & video replay\n\n• Accuracy not yet verified")
    st.divider()
    st.caption("Experimental punch-attempt estimates — not confirmed landed strikes.")


if st.session_state.get("upload_success_message"):
    st.success(st.session_state.pop("upload_success_message"))

st.markdown('''<div class="hero"><div class="eyebrow">COMPUTER VISION / BOXING ANALYTICS</div>
<div class="hero-title">BOXING<span>TRACKER</span> AI</div>
<div class="hero-desc">Two-fighter pose tracking, left/right punch detections, fight timelines and hands-on accuracy review — all in one dashboard.</div>
<div class="status">● EXPERIMENTAL COUNTS &nbsp; · &nbsp; V3 DETECTOR BASELINE</div></div>''', unsafe_allow_html=True)

try:
    video = get_video(folder)
    duration, frames, fps = video_info(video) if video else (0.0, 0, 0.0)
    a = load_analysis(folder, duration=duration)
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Could not load this analysis: {exc}")
    st.caption("If the bundled sample is missing, copy the complete outputs/demo folder from the ZIP.")
    st.stop()

all_total = int(a.summary["total"].sum())
first = int(a.summary[a.summary["boxer"] == 1]["total"].sum())
second = int(a.summary[a.summary["boxer"] == 2]["total"].sum())

metrics = st.columns(4)
with metrics[0]: kpi("Estimated attempts", str(all_total), "Not verified against the video")
with metrics[1]: kpi("Boxer 01", str(first), "Blue corner / all punches", BLUE)
with metrics[2]: kpi("Boxer 02", str(second), "Gold corner / all punches", GOLD)
with metrics[3]: kpi("Clip duration", f"{a.duration:.1f}s", f"{fps:.0f} FPS" if fps else "Imported analysis")

overview, timeline, validate, process = st.tabs(["◈  Fight overview", "⌁  Punch timeline", "✓  Accuracy lab", "+  Analyze video"])

with overview:
    l, r = st.columns([1.55, 1], gap="large")
    with l:
        section("Fight playback")
        show_video(video)
        st.caption("Tracking and punch markers are generated by our v3 YOLO pipeline. The video HUD and the charts show estimated counts.")
        if video is not None and video.exists():
            player_file, _ = ensure_player_video(video)
            st.download_button("⬇ Download browser-ready MP4", data=player_file.read_bytes(),
                               file_name="boxingtracker_replay.mp4", mime="video/mp4", key="dl_video")
            st.caption("The extra dark strip below the video keeps playback controls away from the punch scoreboard. The original analysis video is unchanged.")
    with r:
        section("Corner comparison")
        fighter_card(1, summary_count(a, 1, "left"), summary_count(a, 1, "right"))
        fighter_card(2, summary_count(a, 2, "left"), summary_count(a, 2, "right"))
        st.markdown('<div class="note">Detected ≠ landed or verified. Guard movements and overlapping wrists can create false counts.</div>', unsafe_allow_html=True)
    x, y = st.columns([1.4, 1], gap="large")
    with x:
        section("Punch activity")
        st.plotly_chart(chart_activity(a.events, a.duration), use_container_width=True, config={"displayModeBar": False})
    with y:
        section("Left vs right")
        st.plotly_chart(chart_strike_mix(a), use_container_width=True, config={"displayModeBar": False})

with timeline:
    section("Detection sequence")
    st.caption("Each point is an algorithm-reported attempt. Examine close-together events before accepting them as true punches.")
    st.plotly_chart(chart_event_lanes(a.events, a.duration), use_container_width=True, config={"displayModeBar": False})
    boxer_filter = st.selectbox("Inspect", ["Both fighters", "Boxer 1", "Boxer 2"], key="timeline_filter")
    filtered = a.events if boxer_filter == "Both fighters" else a.events[a.events["boxer"] == int(boxer_filter[-1])]
    cols = ["event_id", "time_seconds", "boxer", "hand", "confidence"]
    st.dataframe(filtered[cols], hide_index=True, use_container_width=True,
                 column_config={"confidence": st.column_config.NumberColumn("Wrist confidence", format="%.2f")})
    st.download_button("Download events CSV", a.events[["time_seconds", "boxer", "punch"]].to_csv(index=False).encode(),
                       file_name="boxing_punch_events.csv", mime="text/csv")
    low = int((a.events["confidence"].isna() | a.events["confidence"].lt(0.65)).sum())
    st.info(f"{low} of {len(a.events)} events have wrist confidence below 0.65 or were missing a high-confidence wrist estimate. This is a review flag, not proof of a false detection.")

with validate:
    section("Manual accuracy validation")
    st.markdown('<div class="note">We cannot calculate real accuracy from model output alone. Label every actual punch in the original clip, then compare timestamps with the model detections.</div>', unsafe_allow_html=True)
    r1, r2 = st.columns([1.25, 1], gap="large")
    with r1:
        st.markdown("**1. Inspect model detections**")
        show_cols = ["event_id", "time_seconds", "boxer", "hand", "confidence"]
        review = a.events[show_cols].copy()
        review["verdict"] = "Unreviewed"
        edited = st.data_editor(review, hide_index=True, use_container_width=True, num_rows="fixed", key="review_editor_" + hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:10],
                                column_config={"verdict": st.column_config.SelectboxColumn("Review verdict", options=["Unreviewed", "Correct", "False positive"], required=True),
                                               "confidence": st.column_config.NumberColumn("Wrist confidence", format="%.2f")},
                                disabled=show_cols)
        st.download_button("Export reviewed detections", edited.to_csv(index=False).encode(),
                           file_name="reviewed_detections.csv", mime="text/csv")
        reviewed = int(edited["verdict"].ne("Unreviewed").sum())
        st.caption(f"{reviewed} / {len(edited)} detections reviewed. These verdicts flag false positives, but do not by themselves identify missed punches.")
    with r2:
        st.markdown("**Jump to a detection**")
        if a.events.empty:
            st.info("No detections to review.")
        else:
            choices = list(a.events.index)
            which = st.selectbox("Choose an event", choices, format_func=lambda i: f"{a.events.at[i, 'event_id']} · {a.events.at[i, 'time_seconds']:.2f}s · Boxer {a.events.at[i, 'boxer']} · {a.events.at[i, 'hand']}")
            event = a.events.iloc[which]
            show_video(video, start=max(0.0, float(event["time_seconds"]) - 2.0))
            st.caption("Video seeks to the preceding whole second. Pause and replay to inspect the punch.")
    st.divider()
    st.markdown("**2. Create complete manual ground truth**")
    st.caption("Add one row for EVERY visible punch attempt, including missed model detections. Use the anatomical left/right hand of each boxer. The template is intentionally empty so we don't treat predictions as truth.")
    blank = pd.DataFrame({"time_seconds": pd.Series(dtype=float), "boxer": pd.Series(dtype=str), "hand": pd.Series(dtype=str)})
    st.download_button("Download blank ground-truth template", "time_seconds,boxer,hand\n".encode(),
                       file_name="manual_ground_truth.csv", mime="text/csv")
    uploaded_labels = st.file_uploader("Or upload your completed manual labels CSV", type=["csv"], key="labels_file")
    if uploaded_labels is not None:
        try:
            labels = pd.read_csv(uploaded_labels)
            st.dataframe(labels, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not read manual labels: {exc}")
            labels = blank
    else:
        labels = st.data_editor(blank, num_rows="dynamic", hide_index=True, use_container_width=True,
                                key="ground_truth_editor_" + hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:10],
                                column_config={"boxer": st.column_config.SelectboxColumn("Boxer", options=["1", "2"], required=True),
                                               "hand": st.column_config.SelectboxColumn("Hand", options=["left", "right"], required=True),
                                               "time_seconds": st.column_config.NumberColumn("Time (s)", min_value=0.0, step=0.04, format="%.2f", required=True)})
    tolerance = st.slider("Timestamp matching tolerance (seconds)", 0.08, 0.60, 0.24, step=0.04)
    completed = st.checkbox("I manually labeled ALL punches in the entire clip", value=False)
    if completed:
        try:
            ground_truth = normalized_ground_truth(labels)
            if ground_truth.empty:
                st.warning("Enter actual punch timestamps first.")
            elif float(ground_truth["time_seconds"].max()) > a.duration + 0.5:
                st.warning("Some manual labels extend beyond this video's duration. Check timestamps.")
            else:
                report = compare_with_ground_truth(a.events, ground_truth, tolerance)
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Precision", f"{report['precision']:.1%}", help="Matched predictions / all predictions")
                with c2: st.metric("Recall", f"{report['recall']:.1%}", help="Matched labels / all manually labeled punches")
                with c3: st.metric("F1 score", f"{report['f1']:.1%}")
                st.caption(f"Matched: {report['tp']} · Extra detections: {report['fp']} · Missed punches: {report['fn']}")
                with st.expander("Inspect missed and unmatched punches"):
                    st.markdown("**Missed manual punches**")
                    st.dataframe(report["missed"], hide_index=True, use_container_width=True)
                    st.markdown("**Unmatched model detections**")
                    st.dataframe(report["unmatched_predictions"], hide_index=True, use_container_width=True)
                st.download_button("Download validated labels", ground_truth.to_csv(index=False).encode(),
                                   file_name="manual_ground_truth.csv", mime="text/csv")
        except (ValueError, TypeError) as exc:
            st.error(f"Please correct your manual labels: {exc}")
    else:
        st.info("Accuracy scores stay hidden until you confirm a complete manual annotation of the clip.")

with process:
    section("Analyze your own boxing clip")
    if PUBLIC_MODE:
        st.warning("Hosted demo: first try a 5-15 second clip at inference size 640. CPU processing can be slow. Uploaded videos and runs are temporary and visible only during this browser session; download your results before leaving.")
    else:
        st.caption("Upload an MP4 with two fighters and a fairly steady camera. Runs are saved separately in your local runs/ folder.")
    clip = st.file_uploader("Choose an MP4 (up to 150 MB / 60 seconds)", type=["mp4"], key="upload_new_clip")
    if clip is not None:
        st.caption(f"Selected: {clip.name} · {len(clip.getbuffer()) / 1024 / 1024:.1f} MB")
        if len(clip.getbuffer()) > MAX_UPLOAD_BYTES:
            st.error("This upload is over the 150 MB limit.")
    col1, col2 = st.columns(2)
    with col1:
        inference_size = st.select_slider("Pose inference image size", options=[640, 960, 1280], value=960,
                                          help="640 runs faster; 960 is our baseline; 1280 may improve small-figure keypoints but is slower.")
    with col2:
        threshold = st.slider("Minimum forward wrist travel", 0.20, 0.60, 0.34, step=0.02,
                              help="0.34 is the unchanged v3 baseline. Different values may increase missed or false counts.")
    if st.button("▶ Analyze uploaded fight", type="primary", disabled=clip is None or (clip is not None and len(clip.getbuffer()) > MAX_UPLOAD_BYTES),
                 use_container_width=True):
        try:
            run_dir, info = save_upload(clip, RUNS)
        except (ValueError, OSError) as exc:
            st.error(f"Could not accept the clip: {exc}")
        else:
            st.info(f"Processing {info.duration:.1f}s · {info.width}×{info.height} · {info.fps:.1f} FPS. YOLO runs on the machine hosting this app; free cloud hosting usually uses CPU.")
            with st.spinner("Detecting both boxers and counting punch attempts. Check the terminal if processing is slow..."):
                result = process_run(run_dir, ROOT, inference_size=inference_size, min_travel=threshold)
            if not result.ok:
                st.error(result.message)
                with st.expander("See error details"):
                    st.code(result.log[-6500:] if result.log else "No process log available")
            else:
                if PUBLIC_MODE:
                    label = f"My upload {run_dir.name[:16]}"
                    st.session_state.setdefault("my_uploaded_runs", []).insert(0, (label, result.output_dir))
                    st.session_state["pending_source"] = label
                else:
                    st.session_state["pending_source"] = next(
                        (label for label, saved in recent_analyses(RUNS) if saved == result.output_dir),
                        "Sample demo - v3",
                    )
                st.session_state["upload_success_message"] = f"Completed: {clip.name}. Your result is selected in the sidebar."
                st.rerun()
    st.markdown("**Alternative: run from VS Code**")
    st.code("python main.py --source videos/boxing.mp4 --show\npython -m streamlit run app.py", language="powershell")
    st.caption("First use may download the YOLO11 pose weights. The app counts punch attempts, not verified landed punches. For testing, use 10–30 second clips.")

st.markdown("<div style='text-align:center;color:#637B99;font-size:.78rem;margin-top:32px'>BOXINGTRACKER AI  ·  EXPERIMENTAL SPORTS VISION  ·  PUNCH-ATTEMPT ESTIMATES</div>", unsafe_allow_html=True)
