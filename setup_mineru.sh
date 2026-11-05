#!/usr/bin/env bash
# Install MinerU (github.com/opendatalab/MinerU) for Scrawl's "MinerU" engine, in its own environment
# (.venv-mineru) so its packages cannot clash with Scrawl's, and download its models (about 2 GB).
#
#   ./setup_mineru.sh            Standard tier: layout + OCR + MinerU's 1.2B vision model (Apple Silicon, GPUs)
#   ./setup_mineru.sh basic      Basic tier only: layout + OCR, no vision model (small machines)
#
# Then start it with ./mineru_server.sh and choose "MinerU" in Scrawl's engine switch.
set -euo pipefail
cd "$(dirname "$0")"
TIER="${1:-standard}"
MINERU_VERSION=4.1.7

export MINERU_HOME="$PWD/vendor/mineru-home"   # models and settings stay inside this folder
export MINERU_TELEMETRY_DISABLED=1             # no usage statistics
mkdir -p "$MINERU_HOME"

if [ ! -x .venv-mineru/bin/python ]; then
  echo "==> Creating .venv-mineru (Python 3.12)"
  if command -v uv >/dev/null; then
    uv venv --python 3.12 .venv-mineru
  else
    PY=$(command -v python3.12 || command -v python3.13 || command -v python3.11 || true)
    [ -n "$PY" ] || { echo "MinerU needs Python 3.11-3.14: install uv (brew install uv) or python@3.12"; exit 1; }
    "$PY" -m venv .venv-mineru
  fi
fi

echo "==> Installing MinerU $MINERU_VERSION"
if command -v uv >/dev/null; then
  uv pip install --python .venv-mineru/bin/python "mineru==$MINERU_VERSION"
else
  .venv-mineru/bin/pip install --upgrade pip
  .venv-mineru/bin/pip install "mineru==$MINERU_VERSION"
fi

echo "==> Downloading MinerU's $TIER models into vendor/mineru-home (about 1-2 GB)"
.venv-mineru/bin/mineru-kit models download --tier "$TIER"
.venv-mineru/bin/mineru-kit models verify --tier "$TIER"

echo "==> Hardware MinerU will use:"
.venv-mineru/bin/mineru-kit hardware info --json | .venv-mineru/bin/python -c "
import json, sys
d = json.load(sys.stdin)
gpus = ', '.join(str(g.get('name')) for g in d.get('gpus') or []) or 'no GPU'
print('   ', (d.get('cpu') or {}).get('model'), '|', round((d.get('memory') or {}).get('total_bytes', 0) / 2**30), 'GB RAM |', gpus)
" || true
echo "==> Done. Start MinerU with:  ./mineru_server.sh"
echo "    then Scrawl as usual (.venv/bin/python server.py) and pick \"MinerU\" in the engine switch."
