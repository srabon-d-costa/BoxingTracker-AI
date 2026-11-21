# 🥊 BoxingTracker AI — Premium Dashboard v5

Two-boxer pose tracking, experimental left/right punch-attempt counting, video replay, and interactive sports analytics built with **YOLO11 Pose + ByteTrack + Python + Streamlit**.

> **Research prototype:** Detected punches are **not** verified landed strikes. Guard movements, blocking, occluded wrists, and ID switching can produce false detections or missed punches. The bundled 13.04-second example reports **30 detections** (Boxer 1: **11**; Boxer 2: **19**). Those are model outputs, **not a validated accuracy score**.

## What's improved in v5

- **Cleaner interface:** More compact headline and metrics, clearer dataset choices, and original blue/gold fighter cards.
- **Honest punch activity chart:** *Discrete, one-second counts* instead of a misleading smoothed line between detected events.
- **Readable replay:** The browser-ready H.264 video includes a narrow dark space *below* the punch scoreboard, so the built-in player controls don't hide the totals. The detector output itself is never modified for this feature.
- **Real upload workflow:** Upload an MP4, validate its duration/format, run the existing v3 model in a subprocess, and automatically open the resulting analysis.
- **Run history:** Results are stored in separate `runs/<unique-id>/outputs/` directories and can be reopened from the sidebar, including after a Streamlit restart.
- **Accuracy lab preserved:** Manual ground-truth review and timestamp-based precision, recall and F1 (only after complete labels are supplied).

**The v3 counting algorithm remains identical.** This release improves presentation, usability, and video-upload handling; it does **not** claim increased punch-detection accuracy.

## Setup — Windows / VS Code

Recommended: Python 3.11 or 3.12. Open your existing `BoxingTracker AI` folder in VS Code.

**If updating your current project:** unzip the v5 package and **copy its contents** (`app.py`, `main.py`, `analytics.py`, `media.py`, `workflow.py`, `.streamlit/`, `outputs/demo/`, etc.) into your existing project folder. Merge the directories. Do **not** replace or delete your own `videos/boxing.mp4` or earlier output folders.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

The Streamlit page will open with the included sample video and results. No inference is needed to view this sample.

### Test the existing boxing video

Put your input at `videos/boxing.mp4`, then:

```powershell
python main.py --source videos/boxing.mp4 --show
```

The unchanged v3 counter writes `annotated_boxing.mp4`, `punch_events.csv`, `punch_debug.csv`, and `summary.csv` to `outputs/`. Select **My local results** in the sidebar to review.

### Analyze a NEW uploaded video

1. Open the **Analyze video** tab of the dashboard.
2. Choose an MP4 (maximum **150 MB** and **60 seconds**; 10–30 seconds recommended for quick experiments).
3. Keep `960` inference image size and `0.34` minimum forward wrist travel to match the v3 baseline. `640` may process faster but may alter pose quality.
4. Click **Analyze uploaded fight**. The first run may download `yolo11n-pose.pt` from Ultralytics. A GPU can accelerate processing; CPU also works.
5. When finished, the new run automatically becomes selected and the page shows its analysis. Later, reopen it using the **Analysis dataset** sidebar selector.

New files are organized as follows:

```text
runs/
└── 20261121T000000Z-abcdef/
    ├── input.mp4
    ├── run.json
    └── outputs/
        ├── annotated_boxing.mp4
        ├── annotated_boxing_player.mp4  # generated on first playback
        ├── punch_events.csv
        ├── punch_debug.csv
        └── summary.csv
```

No previous runs are overwritten. Broken videos are rejected before attempting YOLO inference. If processing fails, expand **See error details**.

## Project structure

```text
BoxingTracker AI/
├── app.py                       # Streamlit dashboard and charts
├── main.py                      # Unchanged v3 two-boxer YOLO punch counter
├── analytics.py                 # CSV and ground-truth data functions
├── media.py                     # H.264 conversion and player-safe video
├── workflow.py                  # Video validation, upload and saved-run management
├── requirements.txt
├── .streamlit/config.toml
├── tests/
│   ├── test_analytics.py
│   └── test_workflow.py
├── videos/                      # Add your boxing.mp4 here
├── outputs/
│   └── demo/                    # Complete sample data and browser-ready replay
└── runs/                        # User-processed videos; excluded from Git
```

## Accuracy validation protocol

The **Accuracy lab** starts with a table of the model's predictions. You can mark suspicious events, but doing so alone doesn't establish **recall** or overall accuracy. To calculate precision, recall and F1, label **every real punch attempt**, including punches the model missed:

```csv
time_seconds,boxer,hand
```

- `boxer`: `1` or `2`
- `hand`: `left` or `right` (anatomical hand of the boxer)
- `time_seconds`: timestamp of the genuine punch attempt

Upload or enter the complete labels, confirm you labeled the **whole video**, and set the timestamp matching tolerance (default 0.24s). Each predicted punch is matched at most once to a label from the same boxer and hand. Avoid claiming measured accuracy before manual annotation is complete.

## Video playback details

OpenCV writes the original `annotated_boxing.mp4` in `mp4v`, which browser players may not support. `media.py` makes a cached browser-compatible H.264 `annotated_boxing_player.mp4` with a dark 88-pixel strip **below** the original HUD, leaving space for the player controls. This only affects the browser replay and downloadable playback copy; the original annotated video stays unchanged. An FFmpeg executable is provided by `imageio-ffmpeg` if not installed separately.

## Publishing to GitHub

Do not add `videos/`, `runs/`, or large model weight files to Git. The local sample video is excluded from Git by default. **If you decide to publish a sample video, check that you have permission to redistribute it**; the data CSVs and code can be committed separately.

```powershell
git add app.py main.py analytics.py media.py workflow.py requirements.txt README.md .streamlit tests start_dashboard.bat run_boxing.bat outputs/demo/*.csv
git commit -m "Polish dashboard and add persistent video upload workflow"
git push origin main
```

If your GitHub clone doesn't include a sample MP4, users can run their own video or add a rights-cleared sample to `outputs/demo`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: analytics` / `media` / `workflow` | Put all `.py` files side by side in the project root; don't extract to a nested directory. |
| Demo dataset missing | Copy the entire `outputs/demo/` folder from the ZIP. |
| Uploaded clip rejected | Use a valid, decodable MP4 of at most 150MB and 60 seconds. |
| Video blank in browser | Check `imageio-ffmpeg` is installed; the dashboard tries to make the H.264 `_player.mp4` copy automatically. |
| YOLO weights download failed | Check your internet access; the first inference run requires the pretrained model. |
| Strange counters / ID switches | These are known v3 limitations; label actual punches and review the event timeline before changing detection thresholds. |

## Testing status

The test suite covers bundled 30-event demo, one-second bin totals, timestamp matching, input video validation, safe file handling, fake-subprocess integration, saved-run retrieval, and H.264 padded replay. **12 tests passed** in the packaging environment. Full Streamlit browser rendering and real YOLO inference with a newly uploaded video still require testing on the target computer (Streamlit and YOLO were not available for execution in the packaging environment).

---
*BoxingTracker AI — a learning-focused computer vision and sports-analytics prototype.*


## Streamlit Community Cloud

See `DEPLOY_STREAMLIT.md`. Deploy `app.py` from `main` using Python 3.11. In **Advanced settings > Secrets** add:

```toml
BOXINGTRACKER_PUBLIC_MODE = "1"
```

The demo opens without running YOLO. Cloud uploads require CPU-intensive inference and may fail on the free tier. Start with 5-15 second MP4 clips and 640 inference size. Uploaded runs are shown only in the current browser session in public mode and are **not persistent** across hosting restarts or redeployments. Download your results promptly. Use local v5 for durable history.
