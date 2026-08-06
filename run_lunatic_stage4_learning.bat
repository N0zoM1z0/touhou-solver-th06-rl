@echo off
setlocal

set "REPO=%~dp0"
if not defined TH06_GAME_DIR set "TH06_GAME_DIR=D:\Entertainment\Game\Touhou\th06"
if not defined TH06_PYTHON set "TH06_PYTHON=%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe"
set "GAME_EXE=%TH06_GAME_DIR%\th06.exe"
set "NATIVE_DLL=%REPO%build\th06_rl_native.dll"
set "PYTHONPATH=%REPO%src;%PYTHONPATH%"

if not exist "%GAME_EXE%" (
  echo Missing exact TH06 executable: "%GAME_EXE%"
  exit /b 1
)
if not exist "%TH06_PYTHON%" (
  echo Missing Windows Python: "%TH06_PYTHON%"
  exit /b 1
)
if not exist "%NATIVE_DLL%" (
  echo Missing native kernel: "%NATIVE_DLL%"
  exit /b 1
)

echo TH06-RL Lunatic / Reimu-A / Stage 4 latency and learning loop
echo Physical HIT is recorded and recovered in the same stage.
echo Create artifacts\pause-lunatic-stage4 to stop between complete stages.

:trial
if exist "%REPO%artifacts\pause-lunatic-stage4" (
  echo Paused before the next trial.
  exit /b 0
)

start "" /D "%TH06_GAME_DIR%" "%GAME_EXE%"
"%TH06_PYTHON%" "%REPO%scripts\run_th06_rl.py" ^
  --game-dir "%TH06_GAME_DIR%" ^
  --native-library "%NATIVE_DLL%" ^
  --practice-stage 4 ^
  --difficulty lunatic ^
  --exploration-rate 0.03 ^
  --patch-lives ^
  --continuous-stage ^
  --armed ^
  --stop-game ^
  %*
set "STATUS=%ERRORLEVEL%"

if "%STATUS%"=="0" goto trial

echo Learning loop stopped on controller status %STATUS%.
exit /b %STATUS%
