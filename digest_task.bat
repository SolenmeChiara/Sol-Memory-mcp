@echo off
rem Weekly digest_rollup task — registered 2026-08-29, runs Mon 12:15.
rem Paths are hard-coded because Task Scheduler's PATH differs from the interactive shell.
cd /d D:\ClaudeExtentions\MCP\Sol-Memory-mcp
C:\Python313\python.exe digest_rollup.py --db "D:\ClaudeExtentions\MCP\Sol-Memory-mcp\memory.db" --claude-cmd "C:\Users\xgq19\.local\bin\claude.exe" --exclude-tier seabed >> logs\digest.log 2>&1
