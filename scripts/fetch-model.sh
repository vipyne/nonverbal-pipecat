#!/usr/bin/env bash
# Download MediaPipe Face Landmarker model into client/public/ if missing.
# Model file is gitignored; `make dev` calls this first.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/client/public/face_landmarker.task"
URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if [ -s "$DEST" ]; then
  echo "face_landmarker.task already present ($(wc -c <"$DEST" | tr -d ' ') bytes)"
  exit 0
fi

mkdir -p "$(dirname "$DEST")"
echo "Downloading face_landmarker.task ..."
curl -fL -o "$DEST" "$URL"
echo "Saved $DEST ($(wc -c <"$DEST" | tr -d ' ') bytes)"
