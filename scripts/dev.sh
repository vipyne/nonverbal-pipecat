#!/usr/bin/env bash
# Start server + client together; tear both down on Ctrl-C. Invoked by `make dev`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$REPO_ROOT/scripts/fetch-model.sh"

if [ ! -d "$REPO_ROOT/server/.venv" ]; then
  (cd "$REPO_ROOT/server" && uv sync)
fi

if [ ! -d "$REPO_ROOT/client/node_modules" ]; then
  (cd "$REPO_ROOT/client" && npm install)
fi

cleanup() {
  kill "$SERVER_PID" "$CLIENT_PID" 2>/dev/null || true
  wait "$SERVER_PID" "$CLIENT_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

(cd "$REPO_ROOT/server" && uv run bot.py) &
SERVER_PID=$!

(cd "$REPO_ROOT/client" && npm run dev) &
CLIENT_PID=$!

wait "$SERVER_PID" "$CLIENT_PID"
