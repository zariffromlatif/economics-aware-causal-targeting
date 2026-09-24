@echo off
REM ============================================================================
REM Synchronize & Pull on Power PC (RTX 4090 Workstation)
REM ============================================================================

echo [INFO] Pulling latest updates from remote repository...
git pull origin master

if %errorlevel% equ 0 (
    echo [SUCCESS] Repository updated! Ready to run experiments on RTX 4090.
) else (
    echo [ERROR] Git pull failed. Please check your network and git configuration.
)
pause
