from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIMPLEHTR_DIR = ROOT / 'vendor' / 'SimpleHTR'
SIMPLEHTR_SRC = SIMPLEHTR_DIR / 'src'
MODELS_DIR = ROOT / 'models'  # models/line holds SimpleHTR's pretrained line checkpoint
CUSTOM_VOCAB = ROOT / 'vocab' / 'custom_words.txt'
MEDICAL_VOCAB = ROOT / 'vocab' / 'medical_terms.txt'  # built-in clinical vocabulary
WEB_DIR = ROOT / 'web'
