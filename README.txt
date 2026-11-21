BoxingTracker AI - fix missing analytics module

1. Extract this ZIP.
2. Copy analytics.py and media.py to your existing BoxingTracker AI folder,
   alongside app.py and main.py (not inside videos or outputs).
3. In VS Code terminal, run:
   python -m pip install -r requirements.txt
   python -m streamlit run app.py

If an import error remains, run this diagnostic command from the project folder:
   python -c "import analytics, media; print(analytics.__file__, media.__file__)"
