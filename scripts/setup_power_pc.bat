@echo off
REM ============================================================================
REM One-Click Setup Script for Power PC (RTX 4090 24GB, i9-14900K, 64GB RAM)
REM ============================================================================

REM Add standard uv installation paths to current session PATH
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%LOCALAPPDATA%\uv;%PATH%"

echo [1/4] Checking uv package manager...
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Installing uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%LOCALAPPDATA%\uv;%PATH%"
)

echo [2/4] Initializing Python virtual environment (.venv)...
uv venv .venv
call .venv\Scripts\activate.bat

echo [3/4] Installing CUDA 12.4+ PyTorch for RTX 4090...
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo [4/4] Installing project dependencies with GPU acceleration...
uv pip install -e .
uv pip install xgboost lightgbm pulp tabulate matplotlib seaborn tqdm pyyaml

echo.
echo ============================================================================
echo [SUCCESS] Power PC setup completed!
echo GPU Device Verification:
.venv\Scripts\python.exe -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
.venv\Scripts\python.exe -c "from src.utils.hardware import get_hardware_info; print(get_hardware_info())"
echo ============================================================================
pause
