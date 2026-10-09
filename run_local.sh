#!/bin/sh
# Reuse the verified packages already on Finley's Mac; no installation.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONPATH='/Users/finleymynott/Documents/Codex/2026-09-28/ple/outputs/streamlit-starter/.venv/lib/python3.14/site-packages'
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/codex-final-project-mplconfig
exec /Users/finleymynott/miniconda3/envs/main/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --server.headless true --browser.gatherUsageStats false
