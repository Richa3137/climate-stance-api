@echo off
title ClimaStance - Climate Stance Detection UI
cls
echo ============================================================================
echo   Starting ClimaStance UI (Streamlit Web Application)...
echo ============================================================================
"C:\Users\richa\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run "%~dp0app.py" --logger.hideWelcomeMessage true
pause
