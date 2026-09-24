@echo off
REM ============================================================================
REM Synchronize & Push from Local Machine to Git Remote
REM ============================================================================

echo [INFO] Git status:
git status -s

set /p COMMIT_MSG="Enter commit message (or press enter for default): "
if "%COMMIT_MSG%"=="" set COMMIT_MSG="chore: update research pipeline and models"

git add -A
git commit -m "%COMMIT_MSG%"
echo [INFO] Pushing to remote repository...
git push origin master

if %errorlevel% equ 0 (
    echo [SUCCESS] Changes successfully pushed! You can now pull on the Power PC.
) else (
    echo [WARNING] Git push encountered an issue. Ensure your remote origin is set.
    echo Example: git remote add origin https://github.com/<your-username>/<repo-name>.git
)
pause
