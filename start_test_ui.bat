@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

call :resolve_python || exit /b 1

set "MODE="
set "DRY_RUN=0"
set "RVC_PORT="
set "RVC_MODELS_DIR=%CD%\.cache\models"
set "RVC_BASE_MODEL_ID=%RVC_MODELS_DIR%\Qwen3-TTS-12Hz-1.7B-Base"
set "RVC_DESIGN_MODEL_ID=%RVC_MODELS_DIR%\Qwen3-TTS-12Hz-1.7B-VoiceDesign"
set "RVC_CUSTOM_VOICE_MODEL_ID=%RVC_MODELS_DIR%\Qwen3-TTS-12Hz-1.7B-CustomVoice"
set "RVC_QWEN_ASR_LOCAL_DIR=%RVC_MODELS_DIR%\Qwen3-ASR-0.6B"
set "RVC_TEXT_CORRECTION_MODEL_ID=%RVC_MODELS_DIR%\Qwen2.5-1.5B-Instruct"
set "RVC_MODEL_PROVIDER=modelscope"
set "RVC_TEXT_CORRECTION_PROVIDER=siliconflow"
set "RVC_SILICONFLOW_MODEL=zai-org/GLM-4.5-Air"

:parse_args
if "%~1"=="" goto after_args
if /i "%~1"=="--dry-run" set "DRY_RUN=1"
if /i "%~1"=="--fake" set "MODE=fake"
if /i "%~1"=="--real" set "MODE=real"
shift
goto parse_args

:after_args

if not defined MODE (
  set "MODE=real"
  echo [Launcher] No mode specified, defaulting to REAL.
)

if /i "%MODE%"=="real" (
  set "RVC_FAKE_MODE=0"
  set "RVC_REQUIRE_GPU=1"
  set "RVC_FAKE_CAPTURE_INPUT=0"
  set "RVC_STARTUP_WARMUP=0"
  set "RVC_EMBED_BOOT_TIMEOUT_S=180"
  set "RVC_CAPTURE_SR=48000"
  set "RVC_TTS_GPU_TURBO=1"
  set "RVC_TTS_FORCE_SDPA=1"
  set "RVC_TTS_INFER_WORKERS=3"
  set "RVC_TTS_MODEL_REPLICAS=3"
  set "RVC_TTS_NON_STREAMING_MODE=1"
  set "RVC_TTS_MAX_NEW_TOKENS_MIN=96"
  set "RVC_TTS_MAX_NEW_TOKENS_CAP=420"
  set "RVC_TTS_TOKENS_PER_CHAR=5.2"
  set "RVC_ASR_BACKEND=qwen3"
  set "RVC_QWEN_ASR_MODEL_ID=Qwen/Qwen3-ASR-0.6B"
  set "RVC_QWEN_ASR_MAX_NEW_TOKENS=72"
  set "RVC_QWEN_ASR_MAX_INFERENCE_BATCH_SIZE=1"
  set "RVC_VAD_SILENCE_MS=120"
  set "RVC_MAX_SEGMENT_MS=2200"
  set "RVC_PTT_ENABLED_DEFAULT=1"
  set "RVC_LLM_CORRECTION_ENABLED_DEFAULT=0"
  set "RVC_TEXT_CORRECTION_MAX_NEW_TOKENS=48"
  set "RVC_PORT=8788"
  echo [Launcher] Mode: REAL ^(provider=%RVC_MODEL_PROVIDER%, gpu_turbo=on^)
) else (
  set "RVC_FAKE_MODE=1"
  set "RVC_REQUIRE_GPU=0"
  set "RVC_FAKE_CAPTURE_INPUT=1"
  set "RVC_STARTUP_WARMUP=0"
  set "RVC_CAPTURE_SR=16000"
  set "RVC_TTS_NON_STREAMING_MODE=1"
  set "RVC_ASR_BACKEND=qwen3"
  set "RVC_PORT=8787"
  echo [Launcher] Mode: FAKE
)
if /i "%RVC_TEXT_CORRECTION_PROVIDER%"=="siliconflow" (
  if not defined RVC_SILICONFLOW_API_KEY (
    if defined SILICONFLOW_API_KEY (
      set "RVC_SILICONFLOW_API_KEY=%SILICONFLOW_API_KEY%"
    )
  )
  if not defined RVC_SILICONFLOW_API_KEY (
    echo [Launcher] Warning: RVC_SILICONFLOW_API_KEY is not set. Text correction will fallback to local rules.
  )
)
set "RVC_URL=http://127.0.0.1:%RVC_PORT%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=UTF-8"

call :set_model_download_env
if not "%DRY_RUN%"=="1" (
  if /i "%MODE%"=="real" call :ensure_sox
  if /i "%MODE%"=="real" if /i not "%RVC_MODEL_PROVIDER%"=="modelscope" call :ensure_hf_xet
  if /i "%MODE%"=="real" if /i "%RVC_MODEL_PROVIDER%"=="modelscope" call :ensure_modelscope || exit /b 1
  if /i "%MODE%"=="real" call :ensure_qwen_asr || exit /b 1
  if /i "%MODE%"=="real" call :predownload_real_models
  call :kill_port "%RVC_PORT%"
)

echo [Launcher] Python: %PYEXE%
echo [Launcher] URL: %RVC_URL%
echo [Launcher] Command: %PYEXE% scripts\run_webview.py --url %RVC_URL%

if "%DRY_RUN%"=="1" (
  echo [Launcher] Dry-run done.
  exit /b 0
)

"%PYEXE%" scripts\run_webview.py --url %RVC_URL%
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo.
  echo [Launcher] Start failed with code %EXITCODE%.
  echo Install dependencies if needed:
  echo   pip install -e ".[desktop]"
  pause
)
exit /b %EXITCODE%

:resolve_python
if exist "%CD%\.venv\Scripts\python.exe" (
  set "PYEXE=%CD%\.venv\Scripts\python.exe"
  exit /b 0
)
where python >nul 2>nul
if not errorlevel 1 (
  set "PYEXE=python"
  exit /b 0
)
echo [Launcher] Python not found.
echo Setup once:
echo   python -m venv .venv
echo   .venv\Scripts\Activate.ps1
echo   pip install -e ".[desktop]"
pause
exit /b 1

:set_model_download_env
if not defined HF_HUB_DOWNLOAD_TIMEOUT set "HF_HUB_DOWNLOAD_TIMEOUT=1200"
if not defined HF_HUB_ETAG_TIMEOUT set "HF_HUB_ETAG_TIMEOUT=60"
if not defined HF_HUB_DISABLE_SYMLINKS_WARNING set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
if not defined HF_XET_HIGH_PERFORMANCE set "HF_XET_HIGH_PERFORMANCE=1"
exit /b 0

:ensure_hf_xet
"%PYEXE%" -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('hf_xet') else 1)" >nul 2>nul
if not errorlevel 1 exit /b 0
echo [Launcher] Installing hf_xet for stable model download...
"%PYEXE%" -m pip install --disable-pip-version-check "huggingface-hub>=0.34.0,<1.0" hf_xet
if errorlevel 1 (
  echo [Launcher] Warning: hf_xet install failed. Continuing with default download backend.
) else (
  echo [Launcher] hf_xet installed.
)
exit /b 0

:ensure_qwen_asr
"%PYEXE%" -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('qwen_asr') else 1)" >nul 2>nul
if not errorlevel 1 exit /b 0
echo [Launcher] Installing qwen-asr ...
"%PYEXE%" -m pip install --disable-pip-version-check -U qwen-asr
if errorlevel 1 (
  echo [Launcher] ERROR: qwen-asr install failed.
  echo [Launcher] Run manually:
  echo   %PYEXE% -m pip install -U qwen-asr
  exit /b 1
)
echo [Launcher] qwen-asr installed.
exit /b 0

:ensure_modelscope
"%PYEXE%" -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('modelscope') else 1)" >nul 2>nul
if not errorlevel 1 exit /b 0
echo [Launcher] Installing modelscope ...
"%PYEXE%" -m pip install --disable-pip-version-check -U modelscope
if errorlevel 1 (
  echo [Launcher] ERROR: modelscope install failed.
  echo [Launcher] Run manually:
  echo   %PYEXE% -m pip install -U modelscope
  exit /b 1
)
echo [Launcher] modelscope installed.
exit /b 0

:predownload_real_models
call :is_model_ready "%RVC_BASE_MODEL_ID%" BASE_READY
call :is_model_ready "%RVC_DESIGN_MODEL_ID%" DESIGN_READY
call :is_model_ready "%RVC_CUSTOM_VOICE_MODEL_ID%" CUSTOM_READY
call :is_model_ready "%RVC_QWEN_ASR_LOCAL_DIR%" ASR_READY
if "%BASE_READY%"=="1" if "%DESIGN_READY%"=="1" if "%CUSTOM_READY%"=="1" if "%ASR_READY%"=="1" (
  echo [Launcher] Local models already exist under: %RVC_MODELS_DIR%
  exit /b 0
)
echo [Launcher] Pre-downloading local models to: %RVC_MODELS_DIR%
"%PYEXE%" scripts\predownload_models.py --base --design --custom --asr --provider "%RVC_MODEL_PROVIDER%" --models-dir "%RVC_MODELS_DIR%" --report-json reports\model_predownload.json
if errorlevel 1 (
  echo [Launcher] ERROR: predownload failed. Local model directories are incomplete.
  echo [Launcher] Stop startup to keep strict local-only runtime.
  echo [Launcher] Re-run after network/model source is stable:
  echo   %PYEXE% scripts\predownload_models.py --all --provider "%RVC_MODEL_PROVIDER%" --models-dir "%RVC_MODELS_DIR%"
  exit /b 1
) else (
  echo [Launcher] Predownload complete.
)
exit /b 0

:is_model_ready
set "%~2=0"
if not exist "%~1\config.json" exit /b 0
if exist "%~1\model.safetensors" (
  set "%~2=1"
  exit /b 0
)
if exist "%~1\model.safetensors.index.json" (
  set "%~2=1"
  exit /b 0
)
if exist "%~1\pytorch_model.bin" (
  set "%~2=1"
  exit /b 0
)
exit /b 0

:ensure_sox
where sox >nul 2>nul
if not errorlevel 1 exit /b 0

set "SOX_DIR="
set "SOX_DIR=%LOCALAPPDATA%\Microsoft\WinGet\Packages\ChrisBagwell.SoX_Microsoft.Winget.Source_8wekyb3d8bbwe\sox-14.4.2"
if exist "%SOX_DIR%\sox.exe" (
  set "PATH=%SOX_DIR%;%PATH%"
  where sox >nul 2>nul
  if not errorlevel 1 (
    echo [Launcher] SoX detected from WinGet cache.
    exit /b 0
  )
)

echo [Launcher] Installing SoX ^(required for real voice clone^)...
where winget >nul 2>nul
if errorlevel 1 (
  echo [Launcher] Warning: winget is not available. Please run scripts\install_sox.ps1 manually.
  exit /b 0
)
winget install -e --id ChrisBagwell.SoX --scope user --accept-package-agreements --accept-source-agreements --silent
if errorlevel 1 (
  echo [Launcher] Warning: SoX install failed. Real clone/design may fail until SoX is available in PATH.
  exit /b 0
)
if exist "%SOX_DIR%\sox.exe" set "PATH=%SOX_DIR%;%PATH%"
where sox >nul 2>nul
if not errorlevel 1 (
  echo [Launcher] SoX installed.
) else (
  echo [Launcher] Warning: SoX still not in current PATH. Restart terminal if real mode fails.
)
exit /b 0

:kill_port
set "TARGET_PORT=%~1"
if "%TARGET_PORT%"=="" exit /b 0
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%TARGET_PORT% .*LISTENING"') do (
  if not "%%P"=="0" (
    echo [Launcher] Stopping stale process on port %TARGET_PORT% ; PID=%%P
    taskkill /PID %%P /F >nul 2>nul
  )
)
exit /b 0
