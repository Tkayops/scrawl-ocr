"""Mindee (cloud, API v2) as a reading engine - its "OCR" utility model returns every word with its position.

DOCUMENTS ARE SENT TO MINDEE. Use it only where your data-protection arrangements allow it.

Two kinds of Mindee model work: an OCR model (Document Utilities > OCR: every word with its position) or an
Extraction model (your own data schema: named fields with confidence and locations, plus the raw text). The kind
is detected automatically.

Setup: in app.mindee.com create a model and an API key, then set
  SCRAWL_MINDEE_KEY        the API key (md_...)
  SCRAWL_MINDEE_MODEL_ID   the OCR model's ID
or put {"key": "...", "model_id": "..."} in mindee.json next to server.py (kept out of git).
The key is only ever sent to Mindee's API address.
"""
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import imaging, layout
from .layout import Region
from .paths import ROOT

API_URL = 'https://api-v2.mindee.net'
CONFIG_FILE = ROOT / 'mindee.json'


class MindeeError(ValueError):  # shown to the user as a readable error
    pass


class MindeeReader:
    def __init__(self) -> None:
        config = json.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}
        self.key = os.environ.get('SCRAWL_MINDEE_KEY') or config.get('key') or ''
        self.model_id = os.environ.get('SCRAWL_MINDEE_MODEL_ID') or config.get('model_id') or ''
        self.url = os.environ.get('SCRAWL_MINDEE_URL') if os.environ.get('SCRAWL_MINDEE_ALLOW_ANY_URL') == '1' \
            else API_URL  # override only for tests
        self.kind = config.get('kind')  # 'ocr' or 'extraction'; found out on first use when not set

    def problem(self) -> Optional[str]:
        if not self.key or not self.model_id:
            return 'not configured (set SCRAWL_MINDEE_KEY and SCRAWL_MINDEE_MODEL_ID, or create mindee.json)'
        return None

    def configured(self) -> bool:
        return self.problem() is None

    def describe(self) -> str:
        kind = {'ocr': 'OCR model', 'extraction': 'extraction model'}.get(self.kind, 'model')
        return f'Mindee {kind} (api-v2.mindee.net)' if self.configured() else f'Mindee: {self.problem()}'

    def _call(self, url: str, body: Optional[bytes] = None, content_type: Optional[str] = None) -> dict:
        headers = {'Authorization': self.key}  # raw key, no "Bearer"
        if content_type:
            headers['Content-Type'] = content_type
        req = urllib.request.Request(url, data=body, headers=headers, method='POST' if body is not None else 'GET')
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read() or b'{}')
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < 5:
                    time.sleep(float(exc.headers.get('Retry-After') or 2 ** attempt))
                    continue
                detail = exc.read().decode(errors='replace')[:300]
                hint = ' (trial credits used up, or a feature not in your plan)' if exc.code == 402 else ''
                raise MindeeError(f'Mindee returned {exc.code}{hint}: {detail}') from None
            except (urllib.error.URLError, OSError) as exc:
                raise MindeeError(f'cannot reach Mindee ({exc})') from None
        raise MindeeError('Mindee kept throttling the request')

    def read(self, image: bytes, filename: str = 'page.jpg', timeout: float = 300):
        """Run one page image through the model; returns (kind, inference result)."""
        if not self.configured():
            raise MindeeError(self.problem())
        if self.kind != 'extraction':
            try:
                result = self._run('ocr', image, filename, timeout, {})
                self.kind = 'ocr'
                return 'ocr', result
            except MindeeError as exc:
                if "'extraction' model" not in str(exc):
                    raise
                self.kind = 'extraction'  # Mindee told us which kind the model is
        options = {'raw_text': 'true', 'polygon': 'true', 'confidence': 'true'}
        return 'extraction', self._run('extraction', image, filename, timeout, options)

    def _run(self, product: str, image: bytes, filename: str, timeout: float, options: dict) -> dict:
        """Enqueue on /v2/products/{product}/enqueue, wait for the job, return the inference result."""
        boundary = uuid.uuid4().hex
        form = [('model_id', self.model_id)] + list(options.items())
        parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in form]
        parts += [(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                   f'Content-Type: image/jpeg\r\n\r\n').encode() + image + b'\r\n',
                  f'--{boundary}--\r\n'.encode()]
        job = self._call(f'{self.url}/v2/products/{product}/enqueue', b''.join(parts),
                         f'multipart/form-data; boundary={boundary}').get('job') or {}
        if not job.get('id'):
            raise MindeeError('unexpected reply from Mindee (no job)')
        deadline, wait = time.time() + timeout, 1.0
        while time.time() < deadline:
            time.sleep(wait)
            job = self._call(f"{self.url}/v2/jobs/{job['id']}?redirect=false").get('job') or job
            status = job.get('status')
            if status == 'Processed' and job.get('result_url'):
                return self._call(job['result_url'])['inference']['result']
            if status == 'Failed':
                raise MindeeError(f"Mindee could not read the page: {(job.get('error') or {}).get('detail', '')}")
            wait = min(wait * 1.5, 4.0)
        raise MindeeError('Mindee took too long to read the page')


def to_lines(result: dict, bgr: np.ndarray, page_index: int = 0) -> List[dict]:  # OCR model
    """Mindee words (relative polygons) -> Scrawl lines. Mindee gives no lines, confidence or handwriting flag:
    words are joined into lines by position, and a line counts as handwritten when its ink is mostly coloured
    pen (the same blue-ink test the local engine uses)."""
    height, width = bgr.shape[:2]
    regions = []
    for word in result['pages'][page_index].get('words', []):
        pts = np.array(word['polygon'], dtype=np.float32) * [width, height]
        region = Region(int(pts[:, 0].min()), int(pts[:, 1].min()), int(np.ceil(pts[:, 0].max())), int(np.ceil(pts[:, 1].max())))
        region.printed_text = word['content']
        regions.append(region)
    pen = imaging.pen_weight(bgr) > 0.5
    ink = imaging.ink_mask(imaging.normalize_illumination(bgr.min(axis=2)))
    lines = []
    for line in layout.merge_lines(regions, gap_factor=1.2):
        words = sorted(line.parts or [line], key=lambda r: r.x0)
        box_ink = ink[line.y0:line.y1, line.x0:line.x1].sum()
        coloured = pen[line.y0:line.y1, line.x0:line.x1].sum() / max(box_ink, 1)
        lines.append({'engine': 'handwriting' if coloured > 0.5 else 'printed', 'reader': 'mindee',
                      'text': ' '.join(w.printed_text for w in words), 'confidence': 0.85, 'bbox': line.bbox})
    return lines


CONFIDENCE = {'Certain': 0.99, 'High': 0.9, 'Medium': 0.7, 'Low': 0.4}


def _flatten(fields: dict, prefix: str = ''):
    """(label, value, confidence, locations) for every leaf field; objects and lists are spelled out."""
    for name, field in (fields or {}).items():
        label = f"{prefix} / {name.replace('_', ' ')}" if prefix else name.replace('_', ' ').capitalize()
        if field is None:
            continue
        if 'items' in field:
            for i, item in enumerate(field.get('items') or [], 1):
                if item.get('fields') is not None:
                    yield from _flatten(item['fields'], f'{label} {i}')
                else:
                    yield f'{label} {i}', item.get('value'), item.get('confidence'), item.get('locations') or []
        elif field.get('fields') is not None:
            yield from _flatten(field['fields'], label)
        else:
            yield label, field.get('value'), field.get('confidence'), field.get('locations') or []


def extraction_page(result: dict, bgr: np.ndarray, page_index: int = 0):
    """Extraction model -> (lines, fields, raw text). Each field value with a location on this page becomes a
    line (handwritten if its ink is mostly coloured pen); fields keep Mindee's names and confidence."""
    height, width = bgr.shape[:2]
    pen = imaging.pen_weight(bgr) > 0.5
    ink = imaging.ink_mask(imaging.normalize_illumination(bgr.min(axis=2)))
    lines, fields = [], []
    for label, value, confidence, locations in _flatten(result.get('fields') or {}):
        if value in (None, ''):
            continue
        conf = CONFIDENCE.get(confidence, 0.8)
        boxes = []
        for loc in locations:
            if loc.get('page', 0) == page_index and loc.get('polygon'):
                pts = np.array(loc['polygon'], dtype=np.float32) * [width, height]
                boxes.append([int(pts[:, 0].min()), int(pts[:, 1].min()), int(np.ceil(pts[:, 0].max())), int(np.ceil(pts[:, 1].max()))])
        field = {'label': label, 'value': str(value), 'confidence': conf, '_line': None}
        if boxes:
            x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
            x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
            coloured = pen[y0:y1, x0:x1].sum() / max(ink[y0:y1, x0:x1].sum(), 1)
            line = {'engine': 'handwriting' if coloured > 0.5 else 'printed', 'reader': 'mindee',
                    'text': str(value), 'confidence': conf, 'bbox': [x0, y0, x1, y1]}
            lines.append(line)
            field['_line'] = line
        elif page_index > 0:
            continue  # a field without a location is reported once, with the first page
        fields.append(field)
    raw_pages = ((result.get('raw_text') or {}).get('pages') or [])
    raw = raw_pages[page_index].get('content', '') if page_index < len(raw_pages) else ''
    return lines, fields, raw
