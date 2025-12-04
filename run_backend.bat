@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo Starting Backend Server...
python api.py
pause
