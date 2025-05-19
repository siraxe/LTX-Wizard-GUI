@echo off
setlocal

:: Download ffmpeg and extract to flet_app and rename folder to ffmpeg 

set "FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"

echo Downloading ffmpeg…
where aria2c >nul 2>nul
if %errorlevel%==0 (
    echo Using aria2c for fast download…
    aria2c -x 16 -s 16 -o ffmpeg.zip "%FFMPEG_URL%" || exit /b 1
) else (
    echo Using curl for download…
    curl -L -o ffmpeg.zip "%FFMPEG_URL%" || exit /b 1
)

echo Ensuring flet_app folder exists…
if not exist "flet_app" mkdir flet_app

echo Extracting ffmpeg…
powershell -Command "Expand-Archive -Path 'ffmpeg.zip' -DestinationPath 'flet_app' -Force" || exit /b 1

echo Renaming folder to ffmpeg inside flet_app…
if exist "flet_app\ffmpeg-master-latest-win64-gpl" (
    ren "flet_app\ffmpeg-master-latest-win64-gpl" "ffmpeg" || exit /b 1
) else (
    echo Extracted folder not found in flet_app. Please check the zip contents.
)

echo Deleting ffmpeg.zip…
del /f /q ffmpeg.zip

echo Setup complete.
endlocal
