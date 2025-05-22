@echo off
:: Activate the virtual environment
call venv/Scripts/activate.bat

:: Run the video sampling script
echo Starting video sampling...
python scripts\sample_video.py --config_path flet_app/assets/config_to_sample.yaml

pause
