@echo off
cd /d "%~dp0"
echo Starting Memory MCP HTTP server on port 3456...
python memory_mcp.py --http --port 3456 --db ./memory.db
pause
