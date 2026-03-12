#!/bin/bash
set -euo pipefail

export PATH="$HOME/.npm-global/bin:$HOME/miniconda3/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
DIR="$HOME/claude/cortex"
PROMPT="$DIR/heartbeat_v2.md"
LOG="$DIR/heartbeat.log"
LOCK="$DIR/.heartbeat.lock"

if [ -f "$LOCK" ]; then
  LOCK_AGE=$(( $(date +%s) - $(stat -c %Y "$LOCK") ))
  if [ "$LOCK_AGE" -lt 540 ]; then
    echo "$(date -Iseconds) SKIP: previous run still active (${LOCK_AGE}s)" >> "$LOG"
    exit 0
  fi
  echo "$(date -Iseconds) STALE lock (${LOCK_AGE}s), removing" >> "$LOG"
fi
touch "$LOCK"

for port in 8080 8082; do
  if ! curl -sf "http://localhost:$port/api/stats" > /dev/null 2>&1; then
    echo "$(date -Iseconds) ERROR: server on port $port not responding" >> "$LOG"
    rm -f "$LOCK"
    exit 1
  fi
done

TIMESTAMP=$(date -Iseconds)
echo "--- activation $TIMESTAMP ---" >> "$LOG"

# Inject timestamp into prompt
INJECTED_PROMPT=$(sed "s/{{TIMESTAMP}}/$TIMESTAMP/g" "$PROMPT")

cd "$DIR"
echo "$INJECTED_PROMPT" | /home/jfischer/.npm-global/bin/claude --print \
  --allowedTools "mcp__claude_ai_Autonomous__*,mcp__claude_ai_cortex__*,mcp__*web_search*,mcp__*web_fetch*" \
  --model claude-opus-4-20250514 \
  -p - \
  >> "$LOG" 2>&1

echo "--- end $TIMESTAMP ---" >> "$LOG"
rm -f "$LOCK"
