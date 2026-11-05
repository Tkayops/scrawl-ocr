"""MinerU (github.com/opendatalab/MinerU) as a local reading engine, instead of TrOCR.

MinerU is a document parser: page layout, reading order, tables and printed OCR, and at its Standard tier a
1.2B vision-language model that reads every text block. It runs as its own local service, in its own Python
environment so its packages cannot clash with Scrawl's:

  ./setup_mineru.sh        install it and download its models (once, about 2 GB)
  ./mineru_server.sh       start the service on http://127.0.0.1:8766

Each page image is sent to that service over HTTP and never leaves the machine: like the vision model, the address
must be on this machine or the private network.

  SCRAWL_MINERU_URL=http://127.0.0.1:8766   service address
  SCRAWL_MINERU_TIER=standard               basic (layout + OCR only) | standard | advanced (default: the best
                                            tier the service offers)
"""
import hashlib
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import zipfile
from html.parser import HTMLParser
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

from . import imaging

DEFAULT_URL = 'http://127.0.0.1:8766'
RECHECK_SECONDS = 60
TIER_ORDER = ('standard', 'basic', 'advanced', 'flash')  # MinerU's own preference


class MinerUError(ValueError):  # shown to the user as a readable error
    pass


class MinerUReader:
    def __init__(self, url: Optional[str] = None, tier: Optional[str] = None) -> None:
        self.url = (url or os.environ.get('SCRAWL_MINERU_URL') or DEFAULT_URL).rstrip('/')
        self.tier = tier or os.environ.get('SCRAWL_MINERU_TIER') or None
        self.tiers: List[str] = []
        self._checked_at, self.problem = 0.0, 'not checked yet'
        self._lock = threading.Lock()

    def available(self) -> bool:
        """Is the service up? Re-checked at most once a minute (start MinerU any time)."""
        with self._lock:
            if self.tiers and time.time() - self._checked_at < RECHECK_SECONDS:
                return True
            if not self.tiers and time.time() - self._checked_at < RECHECK_SECONDS:
                return False
            self._checked_at = time.time()
            from .vlm import is_local_address
            if not is_local_address(self.url):
                self.tiers, self.problem = [], f'{self.url} is not on this machine or the private network (refused)'
                return False
            try:
                with urllib.request.urlopen(self.url + '/v1/tiers', timeout=3) as r:
                    self.tiers = _tier_names(json.load(r))
                self.problem = None if self.tiers else 'the MinerU service offers no tier for images'
            except (urllib.error.URLError, OSError, ValueError):
                self.tiers, self.problem = [], f'no MinerU service at {self.url} (start it with ./mineru_server.sh)'
            return bool(self.tiers)

    def chosen_tier(self) -> Optional[str]:
        if self.tier:
            return self.tier
        return next((t for t in TIER_ORDER if t in self.tiers), None)

    def describe(self) -> str:
        if not self.available():
            return f'MinerU: {self.problem}'
        return f'MinerU {self.chosen_tier()} tier at {self.url}'

    # ------------------------------------------------------------------ Tasks API
    def _call(self, url: str, body: Optional[bytes] = None, content_type: Optional[str] = None,
              method: Optional[str] = None, timeout: float = 60, headers: Optional[dict] = None):
        if urlparse(url).netloc != urlparse(self.url).netloc:  # upload / result links come from the service
            raise MinerUError(f'MinerU pointed to another address ({urlparse(url).netloc}); refused')
        headers = {**({'Content-Type': content_type} if content_type else {}), **(headers or {})}
        req = urllib.request.Request(url, data=body, headers=headers, method=method or ('POST' if body else 'GET'))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors='replace')[:300]
            raise MinerUError(f'MinerU returned {exc.code}: {detail}') from None
        except (urllib.error.URLError, OSError) as exc:
            raise MinerUError(f'cannot reach MinerU at {self.url} ({exc})') from None

    def parse(self, image: bytes, filename: str = 'page.png', timeout: float = 1800) -> dict:
        """One page image -> {'markdown': str, 'middle': dict, 'structured': dict, 'tier': str, 'seconds': float}."""
        if not self.available():
            raise MinerUError(self.problem)
        manifest = {'files': [{'file_name': filename, 'bytes': len(image),
                               'sha256sum': hashlib.sha256(image).hexdigest()}],
                    'ocr_mode': 'ocr', 'output_formats': ['default']}
        if self.chosen_tier():
            manifest['tier'] = self.chosen_tier()
        task = json.loads(self._call(f'{self.url}/v1/tasks', json.dumps(manifest).encode(), 'application/json'))
        upload = task['files'][0].get('upload')
        if upload:  # absent when MinerU already holds these exact bytes
            self._call(upload['url'], image, 'application/octet-stream', method='PUT', timeout=120,
                       headers=upload.get('headers'))
        deadline, wait = time.time() + timeout, 0.5
        while time.time() < deadline:
            time.sleep(wait)
            entry = json.loads(self._call(f"{self.url}/v1/tasks/{task['task_id']}"))['files'][0]
            if entry['status'] == 'completed':
                archive = zipfile.ZipFile(io.BytesIO(self._call(entry['result_url'], timeout=120)))
                members = {name.rsplit('/', 1)[-1]: name for name in archive.namelist()}
                read = lambda name: archive.read(members[name]).decode('utf-8') if name in members else ''
                return {'markdown': read('markdown.md'), 'middle': json.loads(read('middle_json.json') or '{}'),
                        'structured': json.loads(read('structured_content.json') or '{}'),
                        'tier': (entry.get('parse') or {}).get('tier', manifest.get('tier')),
                        'seconds': (entry.get('parse') or {}).get('duration_ms', 0) / 1000}
            if entry['status'] in ('failed', 'canceled'):
                error = entry.get('error') or {}
                raise MinerUError(f"MinerU could not read the page: {error.get('code', '')} {error.get('message', '')}")
            wait = min(wait * 1.5, 3.0)
        raise MinerUError('MinerU took too long to read the page')


def _tier_names(data) -> List[str]:
    """/v1/tiers -> tier names usable for images, in whatever shape the service returns them."""
    items = data.get('data', data.get('tiers', data)) if isinstance(data, dict) else data
    names = []
    for item in items or []:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and item.get('available', True):
            names.append(item.get('tier') or item.get('id') or item.get('name'))
    return [n for n in names if n]


# ---------------------------------------------------------------- conversion to Scrawl's page format

CONFIDENCE = 0.85  # MinerU reports no confidence
CONTAINERS = {'list', 'index', 'image', 'chart', 'code'}  # blocks whose content is more blocks
SKIPPED = {'image_body', 'chart_body'}  # pictures (a signature, a logo, a stamp) have no text to read
LATEX_SYMBOLS = {'times': 'x', 'cdot': '.', 'circ': '°', 'degree': '°', 'pm': '±', 'geq': '>=', 'ge': '>=',
                 'leq': '<=', 'le': '<=', 'sim': '~', 'approx': '~', 'mu': 'µ', 'rightarrow': '->', 'to': '->'}


def to_lines(middle: dict, bgr: np.ndarray, page_index: int = 0) -> Tuple[List[dict], List[dict]]:
    """MinerU's middle JSON -> Scrawl lines, plus MinerU's tables as {'columns', 'rows', 'lines'}.

    MinerU locates blocks (a paragraph, a field, a table), not lines: a block over several text lines is cut at the
    gaps between its rows of ink and its words shared out by the number of written words on each row; table rows
    divide the table evenly. A line counts as handwritten when its ink is mostly coloured pen (the test the local engine uses)."""
    pages = middle.get('pages') or []
    page = next((p for p in pages if p.get('page_idx') == page_index), None) or {}
    height, width = bgr.shape[:2]
    pen = imaging.pen_weight(bgr) > 0.5
    ink = imaging.ink_mask(imaging.normalize_illumination(bgr.min(axis=2)))
    lines, tables = [], []

    def add(text: str, bbox: List[int], heading=False) -> Optional[dict]:
        text = ' '.join(text.split())
        if not text:
            return None
        x0, y0, x1, y1 = bbox
        coloured = pen[y0:y1, x0:x1].sum() / max(ink[y0:y1, x0:x1].sum(), 1)
        line = {'engine': 'handwriting' if coloured > 0.5 else 'printed', 'reader': 'mineru', 'text': text,
                'confidence': CONFIDENCE, 'bbox': bbox, 'heading': heading}  # MinerU's titles, not Scrawl's guess
        lines.append(line)
        return line

    def walk(block: dict, parent_box: Optional[List[int]]) -> None:
        kind, content = block.get('type'), block.get('content')
        bbox = _pixels(block.get('bbox'), width, height) or parent_box
        if kind in SKIPPED or bbox is None:
            return
        if kind == 'table':
            for child in content or []:
                if child.get('type') != 'table_body':
                    walk(child, bbox)
                    continue
                rows = [row for row in parse_table(child.get('content') or '') if any(row)]
                if rows:
                    row_lines = [add('  '.join(c for c in row if c), band) for row, band in zip(rows, _even_rows(bbox, len(rows)))]
                    tables.append({'columns': max(len(r) for r in rows), 'rows': rows,
                                   'lines': [l for l in row_lines if l is not None]})
        elif kind in CONTAINERS:
            for child in content or []:
                walk(child, bbox)
        elif isinstance(content, str):  # equations and code
            add(latex_to_text(content) if kind == 'equation' else content, bbox)
        else:
            heading = {'doc_title': 1, 'paragraph_title': 2}.get(kind, False)
            for text, band in split_block(span_text(content or []), bbox, ink):
                add(text, band, heading)

    for block in page.get('blocks') or []:
        walk(block, None)
    return lines, tables


def _pixels(bbox, width: int, height: int) -> Optional[List[int]]:
    """MinerU boxes are fractions of the page: [x0, y0, x1, y1] in 0..1."""
    if not bbox:
        return None
    x0, y0, x1, y1 = bbox
    box = [max(0, int(x0 * width)), max(0, int(y0 * height)),
           min(width, int(np.ceil(x1 * width))), min(height, int(np.ceil(y1 * height)))]
    return box if box[2] > box[0] and box[3] > box[1] else None


def _even_rows(bbox: List[int], n: int) -> List[List[int]]:
    x0, y0, x1, y1 = bbox
    edges = np.linspace(y0, y1, n + 1).astype(int)
    return [[x0, int(edges[i]), x1, max(int(edges[i + 1]), int(edges[i]) + 1)] for i in range(n)]


def span_text(spans: list) -> str:
    """Inline spans (text, inline equations, code, links) -> plain text."""
    out = ''
    for span in spans:
        kind = span.get('type')
        if kind == 'hyperlink':
            piece = span_text(span.get('content') or [])
        elif kind == 'equation_inline':
            piece = ' ' + latex_to_text(span.get('content') or '') + ' '
        else:
            piece = span.get('content') or ''
        out += piece
    return ' '.join(out.split())


def latex_to_text(tex: str) -> str:
    """MinerU sometimes reads handwritten numbers, dates and units as formulas ($7/3/23$, $37^{\\circ}C$)."""
    tex = tex.strip().strip('$')
    tex = re.sub(r'\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'\1/\2', tex)
    tex = re.sub(r'\\(?:mathrm|text|textbf|textit|mathbf|mathit|operatorname|mathsf)\s*\{([^{}]*)\}', r'\1', tex)
    tex = re.sub(r'\\([%#&_$])', r'\1', tex)
    tex = re.sub(r'\\(?:left|right|big|Big|quad|qquad)\b|\\[,;:! ]', ' ', tex)
    tex = re.sub(r'\\([A-Za-z]+)', lambda m: LATEX_SYMBOLS.get(m.group(1), m.group(1)), tex)
    tex = tex.replace('~', ' ').replace('^', '').replace('{', '').replace('}', '')
    return ' '.join(tex.split())


def _ink_rows(ink: np.ndarray) -> List[Tuple[int, int]]:
    """(top, bottom) of each row of text in a block, from the ink profile; dots and descenders stay with their row."""
    profile = ink.sum(axis=1) > max(2, 0.01 * ink.shape[1])
    runs, start = [], None
    for y, on in enumerate(list(profile) + [False]):
        if on and start is None:
            start = y
        elif not on and start is not None:
            runs.append((start, y))
            start = None
    if len(runs) < 2:
        return runs
    for _ in range(2):  # join runs split by small gaps, using the typical row height
        height = float(np.median([b - a for a, b in runs]))
        merged = [runs[0]]
        for a, b in runs[1:]:
            if a - merged[-1][1] < 0.35 * height:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        runs = merged
    height = float(np.median([b - a for a, b in runs]))
    return [(a, b) for a, b in runs if b - a >= 0.4 * height]


def split_block(text: str, bbox: List[int], ink: np.ndarray) -> List[Tuple[str, List[int]]]:
    """Cut a block that covers several rows of writing into one line per row (see to_lines)."""
    x0, y0, x1, y1 = bbox
    words = text.split()
    rows = _ink_rows(ink[y0:y1, x0:x1])
    if len(rows) < 2 or len(words) < len(rows):
        return [(text, bbox)]
    extents, counts = [], []
    for a, b in rows:
        cols = np.flatnonzero(ink[y0 + a:y0 + b, x0:x1].any(axis=0))
        extents.append((int(cols[0]), int(cols[-1]) + 1) if cols.size else (0, x1 - x0))
        # pieces of writing separated by a gap of 0.3 row heights: roughly the words (on sample_2.png this puts
        # 69 of 78 words on their row; sharing by ink width, 59)
        counts.append(1 + int((np.diff(cols) - 1 >= 0.3 * (b - a)).sum()) if cols.size else 1)
    share = np.cumsum(counts, dtype=float)
    share /= share[-1]
    groups = [[] for _ in rows]
    for j, word in enumerate(words):
        groups[min(int(np.searchsorted(share, (j + 0.5) / len(words))), len(rows) - 1)].append(word)
    if any(not g for g in groups):  # the ink does not match the text: keep the block whole
        return [(text, bbox)]
    return [(' '.join(g), [x0 + e[0], y0 + a, x0 + e[1], y0 + b]) for g, (a, b), e in zip(groups, rows, extents)]


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._cell: Optional[List[str]] = None
        self._span = 1

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.rows.append([])
        elif tag in ('td', 'th'):
            if not self.rows:
                self.rows.append([])
            self._cell = []
            span = dict(attrs).get('colspan') or '1'
            self._span = int(span) if span.isdigit() else 1
        elif tag == 'br' and self._cell is not None:
            self._cell.append(' ')

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self._cell is not None:
            self.rows[-1] += [' '.join(''.join(self._cell).split())] + [''] * (self._span - 1)
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_table(html: str) -> List[List[str]]:
    """MinerU's table body (HTML) -> rows of cell texts, padded to the same width."""
    parser = _TableParser()
    parser.feed(html)
    rows = [r for r in parser.rows if r]
    width = max((len(r) for r in rows), default=0)
    return [r + [''] * (width - len(r)) for r in rows]
