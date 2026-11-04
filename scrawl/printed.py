"""Printed text: detection of all text regions and recognition of typed text with RapidOCR (PP-OCR, ONNX)."""
import threading
import unicodedata
from typing import List, Tuple

import numpy as np
from rapidocr_onnxruntime import RapidOCR

from . import hardware


class PrintedReader:
    def __init__(self) -> None:
        # Thresholds are lower than RapidOCR's defaults so faint or scrawled handwriting is still detected;
        # false positives are dropped later when neither recogniser finds text in them.
        options, self.device = hardware.onnx_options()  # GPU if available, all CPU threads otherwise
        self.engine = RapidOCR(det_thresh=0.2, det_box_thresh=0.3, det_unclip_ratio=1.8,
                               det_limit_side_len=1280, det_limit_type='max', max_side_len=4000,
                               use_cls=False, text_score=0.0, **options)
        self._lock = threading.Lock()

    def detect(self, bgr: np.ndarray) -> List[np.ndarray]:
        """Quadrilaterals (4x2 float arrays) around text, in image coordinates."""
        with self._lock:
            boxes, _ = self.engine(bgr, use_det=True, use_cls=False, use_rec=False, box_thresh=0.3, unclip_ratio=1.8)
        return [np.array(b, dtype=np.float32) for b in (boxes or [])]

    def recognize(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Text and confidence for each BGR crop."""
        if not crops:
            return []
        with self._lock:
            results, _ = self.engine.text_rec(crops)
        return [(unicodedata.normalize('NFKC', text).strip(), float(score)) for text, score, *_ in results]
