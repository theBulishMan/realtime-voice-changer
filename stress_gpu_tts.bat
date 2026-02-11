@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

set "PYEXE="
if exist "%CD%\.venv\Scripts\python.exe" (
  set "PYEXE=%CD%\.venv\Scripts\python.exe"
) else (
  where python >nul 2>nul
  if not errorlevel 1 set "PYEXE=python"
)

if not defined PYEXE (
  echo [stress] Python not found.
  exit /b 1
)

set "BASE_URL=http://127.0.0.1:8788"
if not "%~1"=="" set "BASE_URL=%~1"

echo [stress] base_url=%BASE_URL%
echo [stress] This will inject heavy TTS workload to maximize GPU utilization.

"%PYEXE%" scripts\benchmark_tts_loopback.py ^
  --base-url "%BASE_URL%" ^
  --inject-mode queue ^
  --inject-batches 48 ^
  --repeat 4 ^
  --batch-pause-ms 0 ^
  --append-silence-ms 80 ^
  --realtime-pacing false ^
  --max-wait-seconds 90 ^
  --poll-interval 0.25 ^
  --input-text "这是GPU高负载压测文本，用于持续触发较长TTS推理并提升显卡占用。This is a long stress text for GPU saturation in realtime TTS benchmark." ^
  --report-json reports\gpu_stress_tts.json

set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo [stress] benchmark exited with code %EXITCODE%
) else (
  echo [stress] report written to reports\gpu_stress_tts.json
)
exit /b %EXITCODE%

