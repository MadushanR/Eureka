@echo off
:: ============================================================
:: start_eureka.bat — Eureka Host Worker Startup Script
:: ============================================================
:: This script activates the Python virtual environment and
:: launches the host worker. It is invoked by start_invisible.vbs
:: with window style 0, so no console window will be visible.
::
:: Logs are written to logs\worker.log in this directory.
:: To view them: type logs\worker.log  (or tail with WSL/Git Bash)
:: ============================================================

:: Change to the rag-daemon directory (same folder as this script)
cd /d "%~dp0"

:: Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

:: Activate the virtual environment
call .venv\Scripts\activate.bat

:: Run the worker; stdout and stderr are appended to the log file
python host_worker.py >> logs\worker.log 2>&1
