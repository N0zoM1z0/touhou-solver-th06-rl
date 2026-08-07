@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "REPO=%~dp0"
if not defined TH06_GAME_DIR set "TH06_GAME_DIR=D:\Entertainment\Game\Touhou\th06"
if not defined TH06_PYTHON set "TH06_PYTHON=%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe"
if not defined TH06_CORPUS_SPOOL set "TH06_CORPUS_SPOOL=D:\th06-rl-corpus-spool"
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

echo TH06-RL Lunatic / Reimu-A / Stage 1-2-3-4 rotation
echo Each non-mastered Stage runs three complete trials before rotation.
echo Three latest trustworthy no-HIT clears mark a Stage mastered and skipped.
echo Every Stage owns an independent policy, trace, and corpus scope.
echo Create artifacts\pause-lunatic-stage1234 to stop between complete Stages.

:cycle
set "ACTIVE_STAGES=0"
call :stage_block 1 %*
set "STATUS=%ERRORLEVEL%"
if "%STATUS%"=="90" goto paused
if not "%STATUS%"=="0" goto failed
call :stage_block 2 %*
set "STATUS=%ERRORLEVEL%"
if "%STATUS%"=="90" goto paused
if not "%STATUS%"=="0" goto failed
call :stage_block 3 %*
set "STATUS=%ERRORLEVEL%"
if "%STATUS%"=="90" goto paused
if not "%STATUS%"=="0" goto failed
call :stage_block 4 %*
set "STATUS=%ERRORLEVEL%"
if "%STATUS%"=="90" goto paused
if not "%STATUS%"=="0" goto failed
if "%ACTIVE_STAGES%"=="0" goto all_mastered
goto cycle

:stage_block
set "BLOCK_STAGE=%1"
"%TH06_PYTHON%" "%REPO%scripts\check_stage_mastery.py" "%REPO%artifacts\corpus" --difficulty lunatic --stage %BLOCK_STAGE% --consecutive-clears 3
set "MASTERY_STATUS=%ERRORLEVEL%"
if "%MASTERY_STATUS%"=="0" (
  echo Skipping mastered Lunatic Practice Stage %BLOCK_STAGE%.
  exit /b 0
)
if not "%MASTERY_STATUS%"=="1" exit /b 78
set /a ACTIVE_STAGES+=1
for /L %%R in (1,1,3) do (
  echo Stage %BLOCK_STAGE% trial %%R of 3.
  call :trial %BLOCK_STAGE% %2 %3 %4 %5 %6 %7 %8 %9
  if errorlevel 1 exit /b !ERRORLEVEL!
)
exit /b 0

:trial
set "TRIAL_STAGE=%1"
set "MENU_RETRIES=0"

:trial_attempt
"%TH06_PYTHON%" "%REPO%scripts\finalize_corpus_spool.py" "%TH06_CORPUS_SPOOL%" "%REPO%artifacts\corpus"
if errorlevel 1 exit /b 78
if exist "%REPO%artifacts\pause-lunatic-stage1234" exit /b 90
"%TH06_PYTHON%" "%REPO%scripts\check_storage_budget.py" "%REPO%artifacts" --limit-gib 45 --reserve-mib 512
if errorlevel 1 exit /b 90
"%TH06_PYTHON%" "%REPO%scripts\check_host_memory.py" --reserve-gib 4
if errorlevel 1 exit /b 90

echo Starting independent Lunatic Practice Stage %TRIAL_STAGE%.
start "" /D "%TH06_GAME_DIR%" "%GAME_EXE%"
"%TH06_PYTHON%" "%REPO%scripts\run_th06_rl.py" ^
  --game-dir "%TH06_GAME_DIR%" ^
  --native-library "%NATIVE_DLL%" ^
  --practice-stage %TRIAL_STAGE% ^
  --difficulty lunatic ^
  --exploration-rate 0.03 ^
  --patch-lives ^
  --continuous-stage ^
  --corpus-root "%TH06_CORPUS_SPOOL%" ^
  --defer-corpus-compression ^
  --armed ^
  --stop-game ^
  %2 %3 %4 %5 %6 %7 %8 %9
set "TRIAL_STATUS=%ERRORLEVEL%"
"%TH06_PYTHON%" "%REPO%scripts\finalize_corpus_spool.py" "%TH06_CORPUS_SPOOL%" "%REPO%artifacts\corpus"
if errorlevel 1 exit /b 78
if "%TRIAL_STATUS%"=="77" goto trial_menu_retry
exit /b %TRIAL_STATUS%

:trial_menu_retry
set /a MENU_RETRIES+=1
echo Retrying transient background menu attempt %MENU_RETRIES% of 4.
if %MENU_RETRIES% GEQ 4 exit /b 77
timeout /t 2 /nobreak >nul
goto trial_attempt

:paused
echo Stage 1-2-3-4 rotation paused before the next complete Stage.
exit /b 0

:all_mastered
echo All Lunatic Practice Stages 1-4 have three consecutive trustworthy clears.
exit /b 0

:failed
echo Stage 1-2-3-4 rotation stopped on controller status %STATUS%.
exit /b %STATUS%
