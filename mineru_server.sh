#!/usr/bin/env bash
# Start MinerU's local parsing service for Scrawl's "MinerU" engine (install it first with ./setup_mineru.sh).
#
#   ./mineru_server.sh            the best tier this machine supports (Standard on Apple Silicon and GPUs)
#   ./mineru_server.sh standard   force a tier ceiling: basic | standard | advanced
#
# Leave it running and start Scrawl (server.py) as usual; Scrawl finds it on http://127.0.0.1:8766.
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv-mineru/bin/mineru-kit ] || { echo "MinerU is not installed: run ./setup_mineru.sh first"; exit 1; }

export MINERU_HOME="$PWD/vendor/mineru-home"        # models, settings and short-lived uploads stay in this folder
export MINERU_TELEMETRY_DISABLED=1                  # no usage statistics
export MINERU_MODEL_SOURCE=local HF_HUB_OFFLINE=1   # never download anything while reading documents
# Uploaded pages and results live in vendor/mineru-home/uploads (never in the project folder, where git could see
# them) and are deleted 10 minutes after each page is done (MinerU's default keeps them a day).
mkdir -p "$MINERU_HOME/uploads"
# MinerU's working copies of each page go to a temp folder of its own, emptied at every start (an interrupted job
# would otherwise leave its page behind in the system temp folder).
export TMPDIR="$MINERU_HOME/tmp"
rm -rf "$TMPDIR" && mkdir -p "$TMPDIR"
exec .venv-mineru/bin/mineru-kit api-server --host 127.0.0.1 --port "${SCRAWL_MINERU_PORT:-8766}" \
     --tier "${1:-auto}" --retention-seconds 600 --upload-dir "$MINERU_HOME/uploads"
