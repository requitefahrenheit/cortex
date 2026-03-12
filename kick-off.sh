#!/bin/bash
source ~/claude/env.sh

# kick-off.sh — Start dual Cortex servers + Cloudflare tunnel

export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY first}"

DIR="$HOME/claude/cortex"
cd "$DIR"

echo "Killing existing processes…"
pkill -f cloudflared 2>/dev/null
pkill -f dual-server.py 2>/dev/null
sleep 2

echo "Starting main server (port 8080)…"
~/miniconda3/bin/python3 -u "$DIR/dual-server.py" > cortex.log 2>&1 &
sleep 15

echo "Starting autonomous server (port 8082)…"
CORTEX_DB=~/cortex/autonomous.db CORTEX_PORT=8082 ~/miniconda3/bin/python3 -u "$DIR/dual-server.py" > autonomous.log 2>&1 &
sleep 15

echo "Starting tunnel…"
~/cloudflared tunnel run cortex > tunnel.log 2>&1 &
sleep 15

echo "Checking…"
curl -s https://cortex.fahrenheitrequited.dev/api/stats | python3 -m json.tool | head
