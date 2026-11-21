# BoxingTracker AI v5 — install in your existing VS Code project

1. **Make a backup** of your current `app.py` (optional but recommended).
2. Open your existing `BoxingTracker AI` project folder — the folder that has `app.py`, `main.py`, and `requirements.txt`.
3. Open the v5 ZIP and **copy everything inside it into that existing folder**. Choose **Replace files** and **Merge folders** when Windows asks. **Do not create an extra nested `BoxingTracker-AI-v5` folder.**
4. Keep your `videos/boxing.mp4` and other original `outputs/` files. The update doesn't require deleting them.
5. In VS Code Terminal run:

   ```powershell
   python -m pip install -r requirements.txt
   python -m unittest discover -s tests -v
   python -m streamlit run app.py
   ```

6. Select **Sample demo · v3** in the sidebar to view the 30 existing detections, or select **My local results** for your existing output files.
7. Open **Analyze video**, upload an MP4 (150 MB / 60 s maximum), and click **Analyze uploaded fight**. After success, the new run opens automatically. Future runs can be chosen in the sidebar even after restarting the application.

**What changed:** `app.py`, `media.py`, and new `workflow.py` manage dashboard polish and video upload. **The punch detector in `main.py` is the unchanged v3 baseline**.

**If you see `ModuleNotFoundError`:** `app.py`, `analytics.py`, `media.py`, `workflow.py`, and `main.py` must all be in the same top-level project folder.

**If video playback is blank:** Install `imageio-ffmpeg`; the app makes an H.264 player copy of newly processed videos.

**Accuracy:** The existing counts (11 + 19 = 30) are predictions, not verified strikes. Review them in **Accuracy lab** before reporting accuracy percentages.
