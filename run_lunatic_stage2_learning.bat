@echo off
setlocal

set "REPO=%~dp0"
if not defined TH06_GAME_DIR set "TH06_GAME_DIR=D:\Entertainment\Game\Touhou\th06"
if not defined TH06_PYTHON set "TH06_PYTHON=%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe"
if not defined TH06_CORPUS_SPOOL set "TH06_CORPUS_SPOOL=D:\th06-rl-corpus-spool"
set "GAME_EXE=%TH06_GAME_DIR%\th06.exe"
set "NATIVE_DLL=%REPO%build\th06_rl_native.dll"
set "PYTHONPATH=%REPO%src;%PYTHONPATH%"
set "MENU_RETRIES=0"

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

echo TH06-RL Lunatic / Reimu-A / Stage 2 focused learning loop
echo Physical HIT is recorded and recovered in the same complete Stage.
echo Create artifacts\pause-lunatic-stage2 to stop between complete stages.

:trial
"%TH06_PYTHON%" "%REPO%scripts\finalize_corpus_spool.py" "%TH06_CORPUS_SPOOL%" "%REPO%artifacts\corpus"
if errorlevel 1 exit /b 78
if exist "%REPO%artifacts\pause-lunatic-stage2" (
  echo Paused before the next trial.
  exit /b 0
)

"%TH06_PYTHON%" "%REPO%scripts\check_storage_budget.py" "%REPO%artifacts" --limit-gib 45 --reserve-mib 512
if errorlevel 1 (
  echo Learning loop paused before game launch by the storage guard.
  exit /b 0
)

"%TH06_PYTHON%" "%REPO%scripts\check_host_memory.py" --reserve-gib 4
if errorlevel 1 (
  echo Learning loop paused before game launch by the host-memory guard.
  exit /b 0
)

start "" /D "%TH06_GAME_DIR%" "%GAME_EXE%"
"%TH06_PYTHON%" "%REPO%scripts\run_th06_rl.py" ^
  --game-dir "%TH06_GAME_DIR%" ^
  --native-library "%NATIVE_DLL%" ^
  --practice-stage 2 ^
  --difficulty lunatic ^
  --exploration-rate 0.03 ^
  --patch-lives ^
  --continuous-stage ^
  --corpus-root "%TH06_CORPUS_SPOOL%" ^
  --defer-corpus-compression ^
  --armed ^
  --stop-game ^
  %*
set "STATUS=%ERRORLEVEL%"
"%TH06_PYTHON%" "%REPO%scripts\finalize_corpus_spool.py" "%TH06_CORPUS_SPOOL%" "%REPO%artifacts\corpus"
if errorlevel 1 exit /b 78

if "%STATUS%"=="0" (
  set "MENU_RETRIES=0"
  goto trial
)
if "%STATUS%"=="77" goto menu_retry

echo Learning loop stopped on controller status %STATUS%.
exit /b %STATUS%

:menu_retry
set /a MENU_RETRIES+=1
echo Retrying transient background menu attempt %MENU_RETRIES% of 4.
if %MENU_RETRIES% GEQ 4 goto menu_failed
timeout /t 2 /nobreak >nul
goto trial

:menu_failed
echo Learning loop stopped after repeated background menu failures.
exit /b 77
