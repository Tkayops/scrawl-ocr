"""Optional vision-language model reader, served locally (Ollama, LM Studio, mlx-vlm, vLLM, ...).

It reads each handwritten line independently of TrOCR; the pipeline then lets TrOCR check the reading
against the image, so text the model invents is rejected. Only servers on this machine or the private
network are accepted, so documents cannot leave the network by misconfiguration.

  SCRAWL_VLM_URL=http://127.0.0.1:11434    server address (Ollama's default)
  SCRAWL_VLM_MODEL=qwen2.5vl:7b            model name as the server knows it
  SCRAWL_VLM_API=ollama|openai             API style (default: detected)
  SCRAWL_VLM_ALLOW_REMOTE=1                allow a public address (sends documents off the network!)
"""
import base64
import ipaddress
import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import List, Optional
from urllib.parse import urlparse

import cv2
import numpy as np

DEFAULT_URL = 'http://127.0.0.1:11434'
DEFAULT_MODEL = 'qwen2.5vl:7b'
RECHECK_SECONDS = 60  # how often an unavailable server is looked for again (start Ollama any time)
PROMPT = ('You are transcribing handwriting from a scanned medical document. The image shows one handwritten '
          'line{hint}. Write exactly the characters that are written: keep the original spelling, abbreviations, '
          'numbers, units and punctuation. Do not correct, expand, translate or explain anything. If a word cannot '
          'be read, write [?] for it. Reply with the transcription only.')


PAGE_PROMPT = """You are reading a scanned medical form (image attached). An OCR system has already extracted the \
printed field labels and its reading of the handwritten answers, listed below as JSON. The handwriting readings are \
often wrong, and some answers were missed completely.

Look at the image and give the handwritten answer of every field you can see, including fields the OCR missed. Copy \
exactly what is written: keep abbreviations, dates, numbers and units as written; do not correct, expand, translate \
or explain. Use "" for an empty field and "[?]" for one you cannot read. Never add information that is not written \
on the page.

OCR reading:
{ocr}

Reply with JSON only, in this form: {{"fields": [{{"label": "<label as printed>", "value": "<handwritten answer>"}}]}}"""


def is_local_address(url: str) -> bool:
    """True if every address the host resolves to is loopback, private or link-local."""
    host = urlparse(url).hostname
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0].split('%')[0])
        if not (ip.is_loopback or ip.is_private or ip.is_link_local):
            return False
    return bool(infos)


class VisionReader:
    def __init__(self, url: Optional[str] = None, model: Optional[str] = None, api: Optional[str] = None,
                 timeout: float = 120.0) -> None:
        self.url = (url or os.environ.get('SCRAWL_VLM_URL') or DEFAULT_URL).rstrip('/')
        self.model = model or os.environ.get('SCRAWL_VLM_MODEL') or DEFAULT_MODEL
        self.api = api or os.environ.get('SCRAWL_VLM_API')
        self.timeout = timeout
        self.allow_remote = os.environ.get('SCRAWL_VLM_ALLOW_REMOTE') == '1'
        self._checked_at, self._ready, self.problem = 0.0, False, 'not checked yet'
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ connection
    def _get(self, path: str):
        with urllib.request.urlopen(self.url + path, timeout=3) as r:
            return json.load(r)

    def _post(self, path: str, body: dict):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def available(self) -> bool:
        """Is the server up and does it have the model? Re-checked at most once a minute."""
        with self._lock:
            if self._ready or time.time() - self._checked_at < RECHECK_SECONDS:
                return self._ready
            self._checked_at = time.time()
            if not self.allow_remote and not is_local_address(self.url):
                self.problem = f'{self.url} is not on this machine or the private network (refused)'
                return False
            try:
                if self.api in (None, 'ollama'):
                    try:
                        names = [m['name'] for m in self._get('/api/tags').get('models', [])]
                        self.api = 'ollama'
                    except (urllib.error.URLError, OSError, ValueError):
                        if self.api == 'ollama':
                            raise
                        names = [m['id'] for m in self._get('/v1/models').get('data', [])]
                        self.api = 'openai'
                else:
                    names = [m['id'] for m in self._get('/v1/models').get('data', [])]
            except (urllib.error.URLError, OSError, ValueError) as exc:
                self.problem = f'no vision model server at {self.url} ({exc.__class__.__name__})'
                return False
            wanted = {self.model, f'{self.model}:latest'}
            if not wanted & set(names):
                self.problem = f'model {self.model!r} not found on {self.url} (has: {", ".join(names[:5]) or "none"})'
                return False
            self._ready, self.problem = True, None
            return True

    def describe(self) -> str:
        return f'{self.model} at {self.url}' if self._ready else (self.problem or 'not connected')

    # ------------------------------------------------------------------ reading
    def read(self, gray: np.ndarray, label: Optional[str] = None) -> Optional[str]:
        """Transcription of one handwritten line crop, or None if the server failed."""
        image = _prepare(gray)
        if image is None:
            return None
        hint = f", written as the answer to the printed field '{label}'" if label else ''
        prompt = PROMPT.format(hint=hint)
        b64 = base64.b64encode(cv2.imencode('.png', image)[1].tobytes()).decode()
        try:
            if self.api == 'openai':
                out = self._post('/v1/chat/completions', {
                    'model': self.model, 'temperature': 0, 'max_tokens': 96,
                    'messages': [{'role': 'user', 'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}}]}]})
                text = out['choices'][0]['message']['content']
            else:
                out = self._post('/api/chat', {
                    'model': self.model, 'stream': False, 'keep_alive': '30m',
                    'options': {'temperature': 0, 'num_predict': 96},
                    'messages': [{'role': 'user', 'content': prompt, 'images': [b64]}]})
                text = out['message']['content']
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError):
            with self._lock:
                self._ready, self._checked_at, self.problem = False, time.time(), 'server stopped answering'
            return None
        return _clean(text)

    def read_page(self, bgr: np.ndarray, ocr: dict) -> Optional[List[dict]]:
        """The model's reading of every form field on a page, given the page and what the OCR read."""
        prompt = PAGE_PROMPT.format(ocr=json.dumps(ocr, ensure_ascii=False, indent=1))
        image = _page_image(bgr)
        try:
            if self.api == 'openai':
                out = self._post('/v1/chat/completions', {
                    'model': self.model, 'temperature': 0, 'max_tokens': 1500,
                    'messages': [{'role': 'user', 'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image}'}}]}]})
                text = out['choices'][0]['message']['content']
            else:
                out = self._post('/api/chat', {
                    'model': self.model, 'stream': False, 'keep_alive': '30m', 'format': 'json',
                    'options': {'temperature': 0, 'num_predict': 1500},
                    'messages': [{'role': 'user', 'content': prompt, 'images': [image]}]})
                text = out['message']['content']
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError):
            return None
        return _parse_fields(text)


def _page_image(bgr: np.ndarray, max_side: int = 1600) -> str:
    scale = min(1.0, max_side / max(bgr.shape[:2]))
    if scale < 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return base64.b64encode(cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()).decode()


def _parse_fields(text: str) -> List[dict]:
    """The {"fields": [...]} object from a model reply, tolerating code fences and chatter around it."""
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return []
    items = data.get('fields', []) if isinstance(data, dict) else data
    out = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and isinstance(item.get('label'), str) and isinstance(item.get('value'), (str, int, float)):
            out.append({'label': item['label'].strip(), 'value': str(item['value']).strip()})
    return out


def _prepare(gray: np.ndarray) -> Optional[np.ndarray]:
    """Tight crop with a white margin, at a height vision models read well."""
    ys, xs = np.where(gray < 200)
    if len(ys) < 5:
        return None
    m = max(6, int(0.15 * (ys.max() - ys.min())))
    crop = gray[max(0, ys.min() - m):ys.max() + m + 1, max(0, xs.min() - m):xs.max() + m + 1]
    crop = cv2.copyMakeBorder(crop, m, m, m, m, cv2.BORDER_CONSTANT, value=255)
    if crop.shape[0] < 64:
        f = 64 / crop.shape[0]
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return crop


def _clean(text: str) -> str:
    text = re.sub(r'^```\w*|```$', '', text.strip()).strip()
    text = re.sub(r'^(transcription|text|answer)\s*:\s*', '', text, flags=re.I)
    text = text.splitlines()[0].strip() if text else ''
    if len(text) >= 2 and text[0] == text[-1] and text[0] in '"\'`':
        text = text[1:-1].strip()
    return text
