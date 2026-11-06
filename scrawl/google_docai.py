"""Google Cloud Document AI as a reading engine.

DOCUMENTS ARE SENT TO GOOGLE CLOUD. Use it only where your data-protection arrangements allow it.

Create a processor in the Google Cloud console (Document AI > Processors), in the "eu" or "us" location:
  Enterprise Document OCR   text and layout. With its font-style add-on (asked for by default here) every word is
                            marked handwritten or printed. List price about US$1.50 per 1,000 pages, plus US$6 for
                            the add-on.
  Form Parser               text, layout, key/value pairs, tables and checkboxes. About US$30 per 1,000 pages.

Configuration (environment variables, or google.json next to server.py - kept out of git):
  SCRAWL_GOOGLE_ENDPOINT      the processor's "Prediction endpoint" shown in the console, e.g.
                              https://eu-documentai.googleapis.com/v1/projects/123/locations/eu/processors/abc:process
  SCRAWL_GOOGLE_CREDENTIALS   a service-account JSON key with the "Document AI API User" role (default:
                              GOOGLE_APPLICATION_CREDENTIALS, or `gcloud auth application-default login`)
  SCRAWL_GOOGLE_STYLE_INFO=0  don't ask for the handwriting add-on (OCR processor only)
The credentials only ever go to Google's token service, and pages only to *documentai.googleapis.com.
"""
import base64
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

from . import imaging
from .paths import ROOT

CONFIG_FILE = ROOT / 'google.json'
SCOPE = 'https://www.googleapis.com/auth/cloud-platform'
TOKEN_URI = 'https://oauth2.googleapis.com/token'
ADC_FILE = Path.home() / '.config' / 'gcloud' / 'application_default_credentials.json'
PROCESS_PATH = re.compile(r'^/v1(beta3)?/projects/[^/]+/locations/[^/]+/processors/[^/:]+(/processorVersions/[^/:]+)?:process$')


class GoogleError(ValueError):  # shown to the user as a readable error
    pass


class GoogleReader:
    def __init__(self) -> None:
        config = json.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}
        endpoint = os.environ.get('SCRAWL_GOOGLE_ENDPOINT') or config.get('endpoint') or ''
        if not endpoint and config.get('project') and config.get('processor'):
            location = config.get('location', 'us')
            endpoint = (f"https://{location}-documentai.googleapis.com/v1/projects/{config['project']}"
                        f"/locations/{location}/processors/{config['processor']}")
        if endpoint and not endpoint.rstrip('/').endswith(':process'):
            endpoint = endpoint.rstrip('/') + ':process'
        self.endpoint = endpoint
        self.credentials_file = (os.environ.get('SCRAWL_GOOGLE_CREDENTIALS') or config.get('credentials')
                                 or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') or '')
        if self.credentials_file and not Path(self.credentials_file).is_absolute():
            self.credentials_file = str(ROOT / self.credentials_file)
        style = os.environ.get('SCRAWL_GOOGLE_STYLE_INFO', config.get('style_info', True))
        self.style_info = str(style).lower() not in ('0', 'false', 'no')
        self.allow_any_host = os.environ.get('SCRAWL_GOOGLE_ALLOW_ANY_HOST') == '1'  # tests only
        self._credentials = None
        self._lock = threading.Lock()

    def problem(self) -> Optional[str]:
        if not self.endpoint:
            return 'not configured (set SCRAWL_GOOGLE_ENDPOINT and SCRAWL_GOOGLE_CREDENTIALS, or create google.json)'
        url = urlparse(self.endpoint)
        host = url.hostname or ''
        google_host = url.scheme == 'https' and (host == 'documentai.googleapis.com'
                                                 or host.endswith('-documentai.googleapis.com'))
        if not self.allow_any_host and not google_host:
            return (f'{self.endpoint} is not an https://...documentai.googleapis.com address '
                    f'(refused, pages and credentials would leak)')
        if not PROCESS_PATH.match(url.path):
            return 'the endpoint should be a processor\'s "Prediction endpoint" (.../processors/<id>:process)'
        if self.credentials_file:
            if not Path(self.credentials_file).is_file():
                return f'credentials file not found: {self.credentials_file}'
        elif not ADC_FILE.is_file():
            return 'no credentials (set SCRAWL_GOOGLE_CREDENTIALS to a service-account key file)'
        return None

    def configured(self) -> bool:
        return self.problem() is None

    def describe(self) -> str:
        if not self.configured():
            return f'Google: {self.problem()}'
        location = re.search(r'/locations/([^/]+)/', self.endpoint)
        return f"Google Document AI ({location.group(1) if location else '?'}) at {urlparse(self.endpoint).hostname}"

    # ------------------------------------------------------------------ authentication
    def _token_headers(self) -> dict:
        """OAuth access token for Document AI (google-auth signs the service-account JWT and refreshes it)."""
        import google.auth
        import google.auth.transport.requests
        from google.oauth2 import service_account

        with self._lock:
            if self._credentials is None:
                if self.credentials_file:
                    try:
                        info = json.loads(Path(self.credentials_file).read_text())
                    except (OSError, ValueError) as exc:
                        raise GoogleError(f'cannot read the credentials file ({exc})') from None
                    if info.get('type') != 'service_account':
                        raise GoogleError('the credentials file is not a service-account key (type "service_account")')
                    if info.get('token_uri', TOKEN_URI) != TOKEN_URI and not self.allow_any_host:
                        raise GoogleError(f"the key's token_uri {info.get('token_uri')} is not Google's (refused)")
                    self._credentials = service_account.Credentials.from_service_account_info(info, scopes=[SCOPE])
                else:
                    self._credentials, _ = google.auth.default(scopes=[SCOPE])
            if not self._credentials.valid:
                try:
                    self._credentials.refresh(google.auth.transport.requests.Request())
                except Exception as exc:  # google.auth.exceptions.*: bad key, revoked, clock skew
                    raise GoogleError(f'Google sign-in failed: {str(exc)[:300]}') from None
            headers = {'Authorization': f'Bearer {self._credentials.token}'}
            if getattr(self._credentials, 'quota_project_id', None):  # user (gcloud) credentials need one
                headers['x-goog-user-project'] = self._credentials.quota_project_id
            return headers

    # ------------------------------------------------------------------ processing
    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        for attempt in range(6):
            headers = {'Content-Type': 'application/json; charset=utf-8', **self._token_headers()}
            req = urllib.request.Request(self.endpoint, data=data, headers=headers, method='POST')
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 503) and attempt < 5:  # throttled or busy: wait as told
                    time.sleep(float(exc.headers.get('Retry-After') or 2 ** attempt))
                    continue
                detail = exc.read().decode(errors='replace')
                try:
                    detail = json.loads(detail)['error']['message']
                except (ValueError, KeyError, TypeError):
                    pass
                raise GoogleError(f'Google returned {exc.code}: {str(detail)[:300]}') from None
            except (urllib.error.URLError, OSError) as exc:
                raise GoogleError(f'cannot reach Google ({exc})') from None
        raise GoogleError('Google kept throttling the request')

    def process(self, image: bytes, mime_type: str = 'image/jpeg') -> dict:
        """Process one page image synchronously; returns Document AI's Document."""
        if not self.configured():
            raise GoogleError(self.problem())
        body = {'rawDocument': {'content': base64.b64encode(image).decode(), 'mimeType': mime_type},
                'imagelessMode': True}  # don't send the page image back
        if self.style_info:
            body['processOptions'] = {'ocrConfig': {'premiumFeatures': {'computeStyleInfo': True}}}
            try:
                return self._post(body)['document']
            except GoogleError as exc:
                if not str(exc).startswith('Google returned 400'):
                    raise
                del body['processOptions']
                document = self._post(body)['document']  # a real problem with the page fails again here
                self.style_info = False  # Form Parser and older OCR versions don't take the add-on
                return document
        return self._post(body)['document']


# ---------------------------------------------------------------- conversion to Scrawl's page format

def _text(document_text: str, layout: dict) -> str:
    anchor = (layout or {}).get('textAnchor') or {}
    if anchor.get('textSegments'):
        # indices are strings (int64); a missing startIndex means 0
        return ''.join(document_text[int(s.get('startIndex', 0)):int(s.get('endIndex', 0))]
                       for s in anchor['textSegments'])
    return anchor.get('content', '')


def _spans(layout: dict) -> List[Tuple[int, int]]:
    segments = ((layout or {}).get('textAnchor') or {}).get('textSegments') or []
    return [(int(s.get('startIndex', 0)), int(s.get('endIndex', 0))) for s in segments]


def _box(layout: dict, page: dict, width: int, height: int) -> Optional[List[int]]:
    poly = (layout or {}).get('boundingPoly') or {}
    if poly.get('normalizedVertices'):
        xs = [v.get('x', 0.0) * width for v in poly['normalizedVertices']]
        ys = [v.get('y', 0.0) * height for v in poly['normalizedVertices']]
    elif poly.get('vertices'):
        dim = page.get('dimension') or {}
        sx, sy = width / float(dim.get('width') or width), height / float(dim.get('height') or height)
        xs = [v.get('x', 0) * sx for v in poly['vertices']]
        ys = [v.get('y', 0) * sy for v in poly['vertices']]
    else:
        return None
    box = [max(0, int(min(xs))), max(0, int(min(ys))), min(width, int(np.ceil(max(xs)))), min(height, int(np.ceil(max(ys))))]
    return box if box[2] > box[0] and box[3] > box[1] else None


def _clean(text: str) -> str:
    return ' '.join(text.split())


def to_lines(document: dict, bgr: np.ndarray, page_index: int = 0) -> Tuple[List[dict], dict]:
    """Document AI lines -> Scrawl line dicts, plus its key/value pairs, tables and checkboxes for this page (same
    shape as azure.to_lines). Words are marked handwritten by the OCR add-on; without it (Form Parser), a line counts
    as handwritten when its ink is mostly coloured pen (the test the local engine uses)."""
    text = document.get('text', '')
    page = (document.get('pages') or [{}])[page_index]
    height, width = bgr.shape[:2]

    tokens = page.get('tokens') or []
    has_style = any('styleInfo' in t for t in tokens)
    handwritten = np.zeros(len(text) + 1, bool)
    for token in tokens:
        if (token.get('styleInfo') or {}).get('handwritten'):
            for start, end in _spans(token.get('layout')):
                handwritten[start:end] = True
    if not has_style:
        pen = imaging.pen_weight(bgr) > 0.5
        ink = imaging.ink_mask(imaging.normalize_illumination(bgr.min(axis=2)))

    lines = []
    for line in page.get('lines') or []:
        layout = line.get('layout') or {}
        content, box = _clean(_text(text, layout)), _box(layout, page, width, height)
        if not content or box is None:
            continue
        if has_style:
            chars = [i for start, end in _spans(layout) for i in range(start, min(end, len(text)))
                     if not text[i].isspace()]
            is_hand = bool(chars) and handwritten[chars].mean() > 0.5
        else:
            x0, y0, x1, y1 = box
            is_hand = pen[y0:y1, x0:x1].sum() / max(ink[y0:y1, x0:x1].sum(), 1) > 0.5
        lines.append({'engine': 'handwriting' if is_hand else 'printed', 'reader': 'google', 'text': content,
                      'confidence': round(float(layout.get('confidence', 0.9)), 3), 'bbox': box})

    extras = {'key_values': [], 'tables': [], 'checkboxes': []}
    for field in page.get('formFields') or []:
        name, value = field.get('fieldName') or {}, field.get('fieldValue') or {}
        label = _clean(_text(text, name)).rstrip(':').strip()
        boxes = [b for b in (_box(name, page, width, height), _box(value, page, width, height)) if b]
        kind = field.get('valueType') or ''
        if kind in ('filled_checkbox', 'unfilled_checkbox'):
            if boxes:
                extras['checkboxes'].append({'selected': kind == 'filled_checkbox', 'bbox': boxes[-1],
                                             'confidence': value.get('confidence')})
            content = '[x]' if kind == 'filled_checkbox' else '[ ]'
        else:
            content = _clean(_text(text, value))
        extras['key_values'].append({'label': label, 'value': content, 'boxes': boxes,
                                     'confidence': value.get('confidence')})
    for table in page.get('tables') or []:
        rows = []
        for row in (table.get('headerRows') or []) + (table.get('bodyRows') or []):
            cells = []
            for cell in row.get('cells') or []:
                cells += [_clean(_text(text, cell.get('layout')))] + [''] * (int(cell.get('colSpan') or 1) - 1)
            rows.append(cells)
        if rows:
            columns = max(len(r) for r in rows)
            box = _box(table.get('layout'), page, width, height)
            extras['tables'].append({'columns': columns, 'rows': [r + [''] * (columns - len(r)) for r in rows],
                                     'boxes': [box] if box else []})
    for element in page.get('visualElements') or []:  # checkboxes from the OCR processor's selection-mark add-on
        if element.get('type') in ('filled_checkbox', 'unfilled_checkbox'):
            box = _box(element.get('layout'), page, width, height)
            if box:
                extras['checkboxes'].append({'selected': element['type'] == 'filled_checkbox', 'bbox': box,
                                             'confidence': (element.get('layout') or {}).get('confidence')})
    return lines, extras
