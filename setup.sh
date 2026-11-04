#!/usr/bin/env bash
# One-time setup: Python environment, SimpleHTR code and its pretrained models. Everything stays in this folder.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3.11}
if [ ! -x .venv/bin/python ]; then
  echo "==> Creating .venv with $PYTHON"
  if command -v uv >/dev/null; then uv venv --python "$PYTHON" .venv; else "$PYTHON" -m venv .venv; fi
fi
# GPU: detected automatically; override with  GPU=nvidia|amd|intel|apple|cpu ./setup.sh
GPU=${GPU:-auto}
if [ "$GPU" = auto ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then GPU=nvidia
  elif command -v rocminfo >/dev/null 2>&1 && rocminfo 2>/dev/null | grep -q gfx; then GPU=amd
  elif [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then GPU=apple
  else GPU=cpu; fi  # Intel GPUs: only on request (GPU=intel), most integrated ones are not supported
fi
case "$GPU" in
  nvidia) TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu132} ;;
  amd)    TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/rocm7.14} ;;
  intel)  TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/xpu} ;;
  apple)  TORCH_INDEX=${TORCH_INDEX:-https://pypi.org/simple} ;;  # macOS builds include Metal (MPS)
  *)      TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cpu} ;;
esac
# TensorFlow (SimpleHTR, small) runs on the CPU everywhere: its GPU build needs other CUDA libraries than PyTorch
TF_PACKAGE="tensorflow-cpu==2.15.1"
[ "$(uname -s)" = Darwin ] && TF_PACKAGE="tensorflow==2.15.1"

pip_install() {
  if command -v uv >/dev/null; then uv pip install --python .venv/bin/python "$@"; else .venv/bin/pip install "$@"; fi
}
echo "==> Installing Python packages for: $GPU (PyTorch from $TORCH_INDEX)"
pip_install torch --index-url "$TORCH_INDEX"
pip_install "$TF_PACKAGE" -r requirements.txt
if [ "$GPU" = nvidia ]; then
  echo "==> GPU text detection: replacing onnxruntime with onnxruntime-gpu"
  if command -v uv >/dev/null; then
    uv pip uninstall --python .venv/bin/python onnxruntime
    uv pip install --python .venv/bin/python --reinstall onnxruntime-gpu
  else
    .venv/bin/pip uninstall -y onnxruntime
    .venv/bin/pip install --force-reinstall onnxruntime-gpu
  fi
fi

SIMPLEHTR_COMMIT=bc16124f286b6877d64ea4d435764b252d2a56fd  # the version this was tested with
if [ ! -d vendor/SimpleHTR ]; then
  echo "==> Fetching SimpleHTR"
  git clone https://github.com/githubharald/SimpleHTR vendor/SimpleHTR
  git -C vendor/SimpleHTR checkout --quiet "$SIMPLEHTR_COMMIT"
fi

download_model() {  # name url
  if [ -f "models/$1/checkpoint" ]; then return; fi
  echo "==> Downloading SimpleHTR $1 model"
  tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/model.zip" "$2"
  mkdir -p "models/$1"
  unzip -q -o "$tmp/model.zip" -d "models/$1"
  rm -rf "$tmp"
}
# link from the SimpleHTR README (the line model reads whole lines; the word model is not needed)
download_model line "https://www.dropbox.com/s/7xwkcilho10rthn/line-model.zip?dl=1"

if [ ! -d "models/hf/models--microsoft--trocr-base-handwritten" ]; then
  echo "==> Downloading TrOCR handwriting model (~1.3 GB, one time; it then runs offline)"
  .venv/bin/python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel as M; n='microsoft/trocr-base-handwritten'; TrOCRProcessor.from_pretrained(n, cache_dir='models/hf'); M.from_pretrained(n, cache_dir='models/hf')"
fi

.venv/bin/python -c "import scrawl.hardware as h, torch; d = h.torch_device(); print('==> Handwriting (TrOCR) will run on:', h.torch_device_name(d)); print('==> Text detection will run on:', h.onnx_options()[1])"
echo "==> Done. Start the web UI with:  .venv/bin/python server.py"
