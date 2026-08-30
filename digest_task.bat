@echo off
rem digest_rollup task — DAILY at 12:15 (was weekly Mon 12:15 until the day/half-month/year
rem layers landed). One slot covers every layer: at 12:15 yesterday is finished, and on a
rem Monday the week that ended at 00:00 is finished too, so the whole cascade runs in order.
rem Idempotent — a re-run generates nothing new, so a missed day just gets picked up next time
rem (the day layer looks back 3 days, the half-month layer at the newest finished period).
rem Paths are hard-coded because Task Scheduler's PATH differs from the interactive shell.
cd /d D:\ClaudeExtentions\MCP\Sol-Memory-mcp
echo. >> logs\digest.log
echo ===== run %DATE% %TIME% ===== >> logs\digest.log
C:\Python313\python.exe digest_rollup.py --db "D:\ClaudeExtentions\MCP\Sol-Memory-mcp\memory.db" --claude-cmd "C:\Users\xgq19\.local\bin\claude.exe" --exclude-tier seabed >> logs\digest.log 2>&1
