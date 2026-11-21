# Deploy BoxingTracker AI on Streamlit Community Cloud

The counting algorithm in `main.py` is identical to our v3 baseline. The only app code change isolates public upload history by browser session and makes hosted-mode text accurate.

## Step 1 - Copy and push

Extract this ZIP and copy its CONTENTS into the root of your existing `BoxingTracker AI` folder (beside its `.git` folder). Merge directories; do not nest the package. Back up your local app first.

Confirm `app.py`, `main.py`, `analytics.py`, `media.py`, `workflow.py`, `requirements.txt`, `packages.txt`, `.streamlit/config.toml` and the complete `outputs/demo/` folder exist.

From VS Code Terminal inside the repository root:

```powershell
git status
git add -A
git commit -m "Prepare BoxingTracker AI for Streamlit Cloud"
git push origin main
```

Do not force push. If git says nothing to commit, the files may already be tracked; verify them on GitHub.

## Step 2 - Deploy

1. Open https://share.streamlit.io and sign in with GitHub.
2. Choose Create app -> Yup, I have an app (wording may vary).
3. Repository: `srabon-d-costa/BoxingTracker-AI`
4. Branch: `main`
5. Main file: `app.py`
6. App URL: choose `boxingtracker-ai` if the subdomain is available, or another unique name.
7. Advanced settings: Python version **3.11**.
8. Advanced settings > Secrets: paste this TOML text:

```toml
BOXINGTRACKER_PUBLIC_MODE = "1"
```

9. Click Deploy. Wait for packages to install; watch the cloud build logs.
10. Check the bundled sample video, the 30 detection count, both charts, and Accuracy lab.
11. Try Analyze video using a 5-15 second, low-resolution MP4 and inference size **640** to avoid overloading the server.

## Cloud caveats

- This free Streamlit Cloud server has limited RAM/CPU. It is suitable for showing our existing demo; fresh YOLO video processing may be slow or exceed limits. A supported GPU hosting provider is preferable for heavy uploads.
- File storage is temporary on the hosted server. Uploaded results are visible only to their current browser session with PUBLIC_MODE enabled. Other users cannot select those entries through the UI, but public cloud hosting should NOT be used for sensitive footage.
- Since `runs/` is excluded from Git, any cloud runs may be lost after an app restart/rebuild. Download outputs before leaving.
- The experiment detects punch attempts, not verified landed punches or measured accuracy.

## Troubleshooting

- Missing analytics / media / workflow: files not copied beside app.py in GitHub.
- Missing `outputs/demo/punch_events.csv`: commit the complete demo directory, not just the Python files.
- `libGL.so.1` or glib errors: ensure `packages.txt` is committed in the repository root; rebuild.
- CPU PyTorch install fails: check cloud logs and Python 3.11; requirements use the official CPU wheel index.
- Resource limit reached: try a 5-10 second low-res video at 640 or use the bundled demo only.
- YOLO weights download failure: model weights are obtained on the first inference run; an external download is required.
- Video playback blank: the pre-generated `outputs/demo/annotated_boxing_player.mp4` is included.

Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
