"""Azure AI Document Intelligence (cloud) as the reading engine.

DOCUMENTS ARE SENT TO MICROSOFT AZURE. Use it only where your data-protection arrangements allow it.

Configuration (environment variables, or azure.json next to server.py - kept out of git):
  SCRAWL_AZURE_ENDPOINT   https://<your-resource>.cognitiveservices.azure.com/
  SCRAWL_AZURE_KEY        one of the resource's keys
  SCRAWL_AZURE_MODEL      prebuilt-layout (default: text, handwriting, tables, key/value pairs, checkboxes)
                          or prebuilt-read (text only, cheaper)
The key is only ever sent to Azure AI services domains.
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np

from .paths import ROOT

API_VERSION = '2024-11-30'  # Document Intelligence v4.0 GA
CONFIG_FILE = ROOT / 'azure.json'
AZURE_DOMAINS = ('.cognitiveservices.azure.com', '.api.cognitive.microsoft.com',
                 '.cognitiveservices.azure.us', '.cognitiveservices.azure.cn')


class AzureError(ValueError):  # shown to the user as a readable error
    pass


class AzureReader:
    def __init__(self) -> None:
        config = {}
        if CONFIG_FILE.exists():
            config = json.loads(CONFIG_FILE.read_text())
        self.endpoint = (os.environ.get('SCRAWL_AZURE_ENDPOINT') or config.get('endpoint') or '').rstrip('/')
        self.key = os.environ.get('SCRAWL_AZURE_KEY') or config.get('key') or ''
        self.model = os.environ.get('SCRAWL_AZURE_MODEL') or config.get('model') or 'prebuilt-layout'
        self.allow_any_host = os.environ.get('SCRAWL_AZURE_ALLOW_ANY_HOST') == '1'  # tests / private proxies only

    def problem(self) -> Optional[str]:
        if not self.endpoint or not self.key:
            return 'not configured (set SCRAWL_AZURE_ENDPOINT and SCRAWL_AZURE_KEY, or create azure.json)'
        host = urlparse(self.endpoint).hostname or ''
        if not self.allow_any_host and (urlparse(self.endpoint).scheme != 'https' or not host.endswith(AZURE_DOMAINS)):
            return f'{self.endpoint} is not an Azure AI services address (refused, the key would leak)'
        return None

    def configured(self) -> bool:
        return self.problem() is None

    def describe(self) -> str:
        return f'Azure Document Intelligence ({self.model}) at {urlparse(self.endpoint).hostname}' \
            if self.configured() else f'Azure: {self.problem()}'

    def _request(self, url: str, data: Optional[bytes] = None, content_type: Optional[str] = None):
        headers = {'Ocp-Apim-Subscription-Key': self.key}
        if content_type:
            headers['Content-Type'] = content_type
        req = urllib.request.Request(url, data=data, headers=headers, method='POST' if data is not None else 'GET')
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.status, dict(r.headers), r.read()
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 503) and attempt < 5:  # throttled: wait as told
                    time.sleep(float(exc.headers.get('Retry-After') or 2 ** attempt))
                    continue
                detail = exc.read().decode(errors='replace')[:300]
                raise AzureError(f'Azure returned {exc.code}: {detail}') from None
            except (urllib.error.URLError, OSError) as exc:
                raise AzureError(f'cannot reach Azure ({exc})') from None
        raise AzureError('Azure kept throttling the request')

    def analyze(self, image: bytes, content_type: str = 'image/jpeg', timeout: float = 180) -> dict:
        """Submit one page image and wait for the result (Document Intelligence is asynchronous)."""
        if not self.configured():
            raise AzureError(self.problem())
        features = '&features=keyValuePairs' if self.model == 'prebuilt-layout' else ''
        url = (f'{self.endpoint}/documentintelligence/documentModels/{self.model}:analyze'
               f'?api-version={API_VERSION}{features}')
        status, headers, _ = self._request(url, image, content_type)
        location = {k.lower(): v for k, v in headers.items()}.get('operation-location')
        if status != 202 or not location:
            raise AzureError(f'unexpected Azure reply ({status})')
        deadline, wait = time.time() + timeout, 1.0
        while time.time() < deadline:
            time.sleep(wait)
            _, _, body = self._request(location)
            result = json.loads(body)
            if result.get('status') == 'succeeded':
                return result['analyzeResult']
            if result.get('status') == 'failed':
                raise AzureError(f"Azure could not read the page: {result.get('error', {}).get('message', '')}")
            wait = min(wait * 1.5, 4.0)
        raise AzureError('Azure took too long to read the page')


# ---------------------------------------------------------------- conversion to Scrawl's page format

def _bbox(polygon: List[float]) -> List[int]:
    xs, ys = polygon[0::2], polygon[1::2]
    return [int(min(xs)), int(min(ys)), int(np.ceil(max(xs))), int(np.ceil(max(ys)))]


def _overlaps(spans: List[dict], other: List[dict]) -> bool:
    return any(a['offset'] < b['offset'] + b['length'] and b['offset'] < a['offset'] + a['length']
               for a in spans for b in other)


def to_lines(result: dict, page_index: int = 0) -> Tuple[List[dict], dict]:
    """Azure lines -> Scrawl line dicts (handwriting flag, confidence from the words), plus Azure's own
    key/value pairs, tables and checkboxes for this page, in page pixel coordinates."""
    page = result['pages'][page_index]
    handwritten = [span for style in result.get('styles', []) if style.get('isHandwritten')
                   for span in style.get('spans', [])]
    words = page.get('words', [])
    lines = []
    for line in page.get('lines', []):
        spans = line.get('spans', [])
        confs = [w['confidence'] for w in words if 'span' in w and _overlaps([w['span']], spans)]
        is_hand = _overlaps(spans, handwritten)
        lines.append({'engine': 'handwriting' if is_hand else 'printed', 'reader': 'azure', 'text': line['content'],
                      'confidence': round(float(np.mean(confs)) if confs else 0.9, 3), 'bbox': _bbox(line['polygon'])})
    number = page.get('pageNumber', page_index + 1)

    def on_page(element):
        return [r for r in element.get('boundingRegions', []) if r.get('pageNumber', 1) == number]

    extras = {'key_values': [], 'tables': [], 'checkboxes': []}
    for kv in result.get('keyValuePairs', []):
        key, value = kv.get('key') or {}, kv.get('value') or {}
        if on_page(key):
            extras['key_values'].append({'label': key.get('content', '').strip().rstrip(':').strip(),
                                         'value': value.get('content', '').strip(),
                                         'boxes': [_bbox(r['polygon']) for r in on_page(key) + on_page(value)],
                                         'confidence': kv.get('confidence')})
    for table in result.get('tables', []):
        if not on_page(table):
            continue
        grid = [[''] * table['columnCount'] for _ in range(table['rowCount'])]
        boxes = []
        for cell in table.get('cells', []):
            grid[cell['rowIndex']][cell['columnIndex']] = cell.get('content', '').strip()
            boxes += [_bbox(r['polygon']) for r in on_page(cell)]
        extras['tables'].append({'columns': table['columnCount'], 'rows': grid, 'boxes': boxes})
    for mark in page.get('selectionMarks', []):
        extras['checkboxes'].append({'selected': mark.get('state') == 'selected', 'bbox': _bbox(mark['polygon']),
                                     'confidence': mark.get('confidence')})
    return lines, extras


def line_ids_in(boxes: List[List[int]], lines: List[dict]) -> List[int]:
    """Ids of the lines whose centre lies inside any of the boxes."""
    ids = []
    for line in lines:
        x0, y0, x1, y1 = line['bbox']
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if any(b[0] - 3 <= cx <= b[2] + 3 and b[1] - 3 <= cy <= b[3] + 3 for b in boxes):
            ids.append(line['id'])
    return ids


MAX_SIDE = 3000  # pages are brought to at most this size first, so Azure's coordinates match the page shown


def fit_page(bgr: np.ndarray) -> np.ndarray:
    scale = min(1.0, MAX_SIDE / max(bgr.shape[:2]))
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr


def encode_page(bgr: np.ndarray) -> bytes:
    """High-quality JPEG: sharp enough for handwriting, well under the free tier's 4 MB limit (PNG often is not)."""
    return cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def summary(reader: AzureReader) -> Dict:
    return {'configured': reader.configured(), 'description': reader.describe()}
