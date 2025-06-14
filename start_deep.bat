@echo off
:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Run the training script with the config file in a new window
echo Starting training in a new window...
::start cmd /k python scripts/train_distributed.py flet_app/assets/config_to_train.yaml
deepspeed --num_gpus=1 scripts/train_deepspeed.py --config workspace/configs/TEST.yaml --deepspeed_config deepspeed_config.json

pause