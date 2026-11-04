"""Handwriting recognition with SimpleHTR (https://github.com/githubharald/SimpleHTR).

The network and pretrained weights are SimpleHTR's, used unchanged. On top of it this module adds:
  * crop clean-up: tight crop around the ink, rescaled to the resolution the model was trained on
  * test-time augmentation: the crop is read at several stroke thicknesses and the per-frame
    character probabilities are averaged, which steadies the output on messy writing
  * dictionary-guided correction scored against the network output (see lexicon.py)
"""
import os
import re
import sys
import threading
import warnings
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore', category=UserWarning)
import tensorflow as tf  # noqa: E402
from tensorflow.python.util import deprecation  # noqa: E402

deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.get_logger().setLevel('ERROR')

from . import ctc  # noqa: E402
from .hardware import CPU_THREADS  # noqa: E402
from .lexicon import Lexicon  # noqa: E402
from .paths import MODELS_DIR, SIMPLEHTR_SRC  # noqa: E402

sys.path.insert(0, str(SIMPLEHTR_SRC))
from model import DecoderType, Model  # noqa: E402  (SimpleHTR)

NET_HEIGHT = 32    # SimpleHTR input height
WORK_HEIGHT = 96   # crops are first brought to roughly IAM scan resolution, then resized like SimpleHTR does
CLOSING_PUNCT = ',.;:!?)'
MAX_GAP = 0.45  # widest word gap kept, as a fraction of the line height


class _RestoredModel(Model):
    """SimpleHTR's Model, restored from a given directory instead of the hard-coded ../model/."""

    def __init__(self, char_list: List[str], model_dir: Path) -> None:
        self._model_dir = model_dir
        super().__init__(char_list, DecoderType.BestPath, must_restore=True)

    def setup_tf(self):
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=CPU_THREADS,
                                          inter_op_parallelism_threads=CPU_THREADS)
        config.gpu_options.allow_growth = True  # a GPU build of TensorFlow must leave room for PyTorch
        sess = tf.compat.v1.Session(config=config)
        saver = tf.compat.v1.train.Saver(max_to_keep=1)
        checkpoint = tf.train.latest_checkpoint(str(self._model_dir))
        if not checkpoint:
            raise FileNotFoundError(f'No SimpleHTR checkpoint in {self._model_dir} (run ./setup.sh)')
        saver.restore(sess, checkpoint)
        return sess, saver


def tidy_punctuation(text: str) -> str:
    """The line model was trained on IAM's tokenised words, so it reads "dark ; and" - attach punctuation."""
    text = re.sub(rf'\s+([{re.escape(CLOSING_PUNCT)}])', r'\1', text)
    text = re.sub(r'\(\s+', '(', text)
    return re.sub(r'\s{2,}', ' ', text).strip()


def _attach_punctuation(words: List[dict]) -> List[dict]:
    out = []
    for word in words:
        if out and word['text'] and all(c in CLOSING_PUNCT for c in word['text']):
            out[-1] = {**out[-1], 'text': out[-1]['text'] + word['text']}
        else:
            out.append(word)
    return out


def prepare_crop(gray: np.ndarray) -> Optional[np.ndarray]:
    """Tight crop around the ink, rescaled so the text is WORK_HEIGHT px tall. None if the crop is blank."""
    if gray.size == 0 or gray.min() > 190:
        return None
    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (gray < min(thresh, 200)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    specks = stats[:, cv2.CC_STAT_AREA] < max(6, (0.03 * gray.shape[0]) ** 2)  # leftovers of neighbouring lines
    specks[0] = True
    rows = np.where((~specks[labels]).sum(axis=1) > 0)[0]
    cols = np.where(ink.sum(axis=0) > 0)[0]
    if len(rows) < 3 or len(cols) < 3:
        return None
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    margin = max(2, int(0.06 * (y1 - y0)))
    y0, y1 = max(0, y0 - margin), min(gray.shape[0], y1 + margin)
    x0, x1 = max(0, x0 - margin), min(gray.shape[1], x1 + margin)
    crop = gray[y0:y1, x0:x1]

    f = WORK_HEIGHT / crop.shape[0]
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * f)), WORK_HEIGHT), interpolation=interp)
    return shrink_gaps(crop, max_gap=int(MAX_GAP * WORK_HEIGHT))


def shrink_gaps(img: np.ndarray, max_gap: int) -> np.ndarray:
    """Cap blank runs between words: SimpleHTR's line model never saw gaps wider than ~half the line height."""
    blank = (img < 160).sum(axis=0) == 0
    keep = np.ones(img.shape[1], bool)
    start = None
    for x, is_blank in enumerate(list(blank) + [False]):
        if is_blank and start is None:
            start = x
        elif not is_blank and start is not None:
            if x - start > max_gap:
                keep[start + max_gap // 2:x - (max_gap - max_gap // 2)] = False
            start = None
    return img[:, keep]


def stroke_variants(img: np.ndarray) -> List[np.ndarray]:
    """Same geometry, different stroke thickness (white paper, dark ink)."""
    k = np.ones((3, 3), np.uint8)
    return [img, cv2.erode(img, k), cv2.dilate(img, k)]


def to_network_input(img: np.ndarray, width: Optional[int] = None) -> np.ndarray:
    """SimpleHTR's inference preprocessing: height 32, dynamic width, 16px padding, transposed, in [-0.5, 0.5]."""
    h, w = img.shape
    f = NET_HEIGHT / h
    if width is None:
        width = int(f * w + 16)
        width += (4 - width) % 4
    tx = (width - w * f) / 2
    target = np.full((NET_HEIGHT, width), 255.0)
    out = cv2.warpAffine(img.astype(np.float64), np.float32([[f, 0, tx], [0, f, 0]]), dsize=(width, NET_HEIGHT),
                         dst=target, borderMode=cv2.BORDER_TRANSPARENT)
    return cv2.transpose(out) / 255 - 0.5


class HandwritingReader:
    def __init__(self, models_dir: Path = MODELS_DIR, lexicon: Optional[Lexicon] = None) -> None:
        self.models_dir = models_dir
        self.lexicon = lexicon
        self.char_list = list((models_dir / 'line' / 'charList.txt').read_text())
        self.char_index = {c: i for i, c in enumerate(self.char_list)}
        self.blank = len(self.char_list)
        self._models = {}
        self._lock = threading.Lock()  # one TF session per model, shared across threads
        gpus = tf.config.list_physical_devices('GPU')
        self.device_name = f'GPU ({gpus[0].name})' if gpus else f'CPU ({CPU_THREADS} threads)'

    def _model(self, kind: str) -> _RestoredModel:
        if kind not in self._models:
            graph = tf.Graph()  # one graph per model so several can coexist
            with graph.as_default():
                self._models[kind] = _RestoredModel(self.char_list, self.models_dir / kind)
        return self._models[kind]

    def encode(self, text: str) -> List[int]:
        return [self.char_index[c] for c in text if c in self.char_index]

    def log_probs(self, gray: np.ndarray, kind: str = 'line', tta: bool = True) -> Optional[np.ndarray]:
        """T x C per-frame log probabilities for a crop (averaged over augmentations)."""
        work = prepare_crop(gray)
        if work is None:
            return None
        batch = np.stack([to_network_input(v) for v in (stroke_variants(work) if tta else [work])])
        with self._lock:
            model = self._model(kind)
            logits = model.sess.run(model.ctc_in_3d_tbc, {model.input_imgs: batch, model.is_train: False})
        probs = np.exp(ctc.log_softmax(logits)).mean(axis=1)  # T x B x C -> T x C
        return np.log(probs + 1e-12)

    def read(self, gray: np.ndarray, kind: str = 'line', tta: bool = True,
             lexicon_strength: float = 3.0) -> Optional[dict]:
        """Recognise one line (or word) crop. Returns text, confidence and per-word details."""
        lp = self.log_probs(gray, kind, tta)
        if lp is None:
            return None
        raw = ''.join(self.char_list[i] for i in ctc.best_path(lp, self.blank))

        def score(texts: List[str]) -> np.ndarray:
            return ctc.log_likelihood(lp, [self.encode(t) for t in texts], self.blank)

        if self.lexicon is not None and lexicon_strength > 0 and raw.strip():
            text, words = self.lexicon.correct(raw, score, self.char_index, strength=lexicon_strength)
        else:
            text, words = raw, [{'text': w, 'confidence': None, 'alternatives': []} for w in raw.split()]

        log_lik = float(score([text])[0])
        confidence = float(np.exp(log_lik / max(1, len(text)))) if np.isfinite(log_lik) else 0.0
        return {'text': tidy_punctuation(text), 'raw_text': raw.strip(), 'confidence': round(confidence, 3),
                'words': _attach_punctuation(words)}
