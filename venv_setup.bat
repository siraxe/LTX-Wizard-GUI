:: Use python 3.12.x
:: Create virtual environment
pip install virtualenv
virtualenv -p python3.12.9 venv

::::::::::::::::::::::::::::::::
:: Install external dependencies
::::::::::::::::::::::::::::::::
:: torch
venv\Scripts\pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
:: pyav
venv\Scripts\pip install https://files.pythonhosted.org/packages/84/7d/ed088731274746667e18951cc51d4e054bec941898b853e211df84d47745/av-14.3.0-cp312-cp312-win_amd64.whl

:: Install requirements
venv\Scripts\pip install -r requirements.txt

:: Install local src package
venv\Scripts\pip install -e .

:: Get ffmpeg
call get_ffmpeg.bat

:: Activate virtual environment (to test)
venv\Scripts\activate
