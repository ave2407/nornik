#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

mkdir -p logs

pkill -f "uvicorn panorama_app.backend.main:app" 2>/dev/null || true
pkill -f "vite.*5173" 2>/dev/null || true

nohup uvicorn panorama_app.backend.main:app --host 0.0.0.0 --port 7860 \
  > logs/panorama_backend.log 2>&1 &

cd panorama_app/frontend
nohup npm run dev -- --host 0.0.0.0 --port 5173 \
  > ../../logs/panorama_frontend.log 2>&1 &

echo "Backend:  http://127.0.0.1:7860"
echo "Frontend: http://127.0.0.1:5173"
