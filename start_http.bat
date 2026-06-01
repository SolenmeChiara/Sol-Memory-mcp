@echo off
cd /d "%~dp0"

echo Checking for old Memory MCP process on port 3456...
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr ":3456.*LISTENING"') do (
    echo   Killing old process PID %%p
    taskkill /f /pid %%p >nul 2>&1
)
echo.

echo Starting Memory MCP HTTP server on port 3456...
python memory_mcp.py --http --port 3456 --db ./memory.db
pause
