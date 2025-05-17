@echo off
:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Run the Flet app
echo Running Flet app...
flet run flet_app/flet_app.py

pause
