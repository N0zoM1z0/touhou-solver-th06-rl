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

echo TH06-RL Lunatic / Reimu-A / complete Stage 4-5-6 learning cycle
echo Every stage owns an independent policy, trace, and corpus scope.
echo Create artifacts\pause-lunatic-stage456 to stop between complete stages.

:cycle
call :trial 4 %*
set "STATUS=%ERRORLEVEL%"
if "%STATUS%"=="90" goto paused
if not "%STATUS%"=="0" goto failed

call :trial 5 %*
set "STATUS=%ERRORLEVEL%"
if "%STATUS%"=="90" goto paused
if not "%STATUS%"=="0" goto failed

call :trial 6 %*
set "STATUS=%ERRORLEVEL%"
if "%STATUS%"=="90" goto paused
if not "%STATUS%"=="0" goto failed
goto cycle

:trial
if exist "%REPO%artifacts\pause-lunatic-stage456" exit /b 90
"%TH06_PYTHON%" "%REPO%scripts\check_storage_budget.py" "%REPO%artifacts" --limit-gib 45 --reserve-mib 512
if errorlevel 1 exit /b 90
"%TH06_PYTHON%" "%REPO%scripts\check_host_memory.py" --reserve-gib 4
if errorlevel 1 exit /b 90

echo Starting independent Lunatic Practice Stage %1.
start "" /D "%TH06_GAME_DIR%" "%GAME_EXE%"
"%TH06_PYTHON%" "%REPO%scripts\run_th06_rl.py" ^
  --game-dir "%TH06_GAME_DIR%" ^
  --native-library "%NATIVE_DLL%" ^
  --practice-stage %1 ^
  --difficulty lunatic ^
  --exploration-rate 0.03 ^
  --patch-lives ^
  --continuous-stage ^
  --armed ^
  --stop-game ^
  %2 %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL%

:paused
echo Stage 4-5-6 learning cycle paused before the next complete Stage.
exit /b 0

:failed
echo Stage 4-5-6 learning cycle stopped on controller status %STATUS%.
exit /b %STATUS%
