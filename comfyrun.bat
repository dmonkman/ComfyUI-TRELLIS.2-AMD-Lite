@echo off
setlocal enabledelayedexpansion

:: add or remove commandline parameters into this line, these are recommended for 6000 series amd gpu's.
:: for debugging: --disable-all-custom-nodes --whitelist-custom-nodes ComfyUI-Trellis2
set PARAMS= --use-quad-cross-attention --disable-smart-memory --disable-pinned-memory --disable-api-nodes --preview-method latent2rgb --disable-cuda-malloc

:: change the path below *only below* to wherever you have installed comfyui to. (in this example it is : "D:\ComfyUI") , it will change everything else automatically.

:: change this to your comfyui folder
set COMFYUI_PATH=D:\path\to\your\comfyui
cd /d "%COMFYUI_PATH%"
call ".\venv\Scripts\activate"

rocm-sdk init
for /f "delims=" %%i in ('rocm-sdk path --root') do set ROCM_ROOT=%%i
for /f "delims=" %%i in ('rocm-sdk path --bin') do set ROCM_BIN=%%i
set ROCM_HOME=%ROCM_ROOT%
set PATH=%ROCM_ROOT%\lib\llvm\bin;%ROCM_BIN%;%PATH%

set CC=clang-cl
set CXX=clang-cl
set DISTUTILS_USE_SDK=1

set TRITON_CACHE_DIR=%COMFYUI_PATH%\.triton
set COMFYUI_ENABLE_MIOPEN=1
set MIOPEN_FIND_MODE=2

:: set MIOPEN_DEBUG_CONV_DIRECT=0 ; this seems to break stuff for some users so it is left here but disabled ; if you want you can remove this line and the one below (which replaced this line) all together, miopen would still work, it would take longer to solve.
set MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=0
set MIOPEN_LOG_LEVEL=0

:: echo Updating ComfyUI...
:: git pull

set "DISPLAY_PARAMS=%PARAMS:--=%"
echo ComfyUI is starting with: [ !DISPLAY_PARAMS! ]
python main.py %PARAMS%