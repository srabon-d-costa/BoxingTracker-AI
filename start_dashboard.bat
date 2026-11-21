@echo off
cd /d "%~dp0"
python -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo Dashboard could not start. First run: python -m pip install -r requirements.txt
    pause
)
