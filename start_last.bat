@echo off
:: Activate the virtual environment
call venv/Scripts/activate.bat

:: Run the training script with the config file in a new window
echo Starting training in a new window...
start cmd /k python scripts/train.py flet_app/assets/config_to_train.yaml --num_processes 1 

pause
