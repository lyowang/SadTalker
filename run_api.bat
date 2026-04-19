@echo off

IF NOT EXIST venv (
python -m venv venv
) ELSE (
echo venv folder already exists, skipping creation...
)
call .\venv\Scripts\activate.bat

set PYTHON="venv\Scripts\Python.exe"
echo venv %PYTHON%

set PATH=%SystemRoot%\system32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SYSTEMROOT%\System32\WindowsPowerShell\v1.0\;C:\Users\tommy\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin;%PATH%
%PYTHON% sadtalker_api.py

echo.
echo Launch unsuccessful. Exiting.
pause
