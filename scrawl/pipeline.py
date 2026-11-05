"""Page pipeline: clean up -> detect text -> read printed text (RapidOCR) and handwriting (SimpleHTR) -> structure."""
import base64
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from rapidfuzz import fuzz

from . import context as clinical_context
from . import hardware, imaging, layout, structure, trocr
from .vlm import VisionReader
from .htr import HandwritingReader, tidy_punctuation
from .layout import Region
from .lexicon import WORD_RE, default_lexicon
from .paths import CUSTOM_VOCAB, MEDICAL_VOCAB, MODELS_DIR, SIMPLEHTR_DIR
from .printed import PrintedReader

LABEL_VALUE_RE = re.compile(r'^([^:]{1,40}:)\s*(\S.*)$')
SURE_PRINTED = 0.98  # printed-model confidence above which a box is never reassigned to handwriting
MIN_PEN_FRACTION = 0.0005  # share of coloured-ink pixels from which a page gets a separate pen layer
MAX_LINE_TILT = 12  # degrees; steeper text in the pen layer is a stamp, not handwriting
CONTEXT_SURE = 0.5  # known words read at least this confidently are not reconsidered from context (speed)
NAME_FIELD = re.compile(r'name|surname|signature|signed|doctor|\bdr\b|nurse|clinician|prepared by', re.I)


@dataclass
class Options:
    mode: str = 'auto'               # auto | handwriting | printed
    tta: bool = True                 # read each handwritten line at several stroke widths and average
    lexicon_strength: float = 3.0    # 0 = no dictionary correction; higher = trust the dictionary more
    printed_threshold: float = 0.9   # auto mode: printed-OCR confidence needed to accept its reading
    merge_gap: float = 1.5           # join handwritten boxes on a line if the gap is < this x line height
    rescue: bool = True              # look for handwriting the detector missed
    pen_layer: bool = True           # read coloured ink (blue/red pen) separately from the black print
    handwriting_model: str = 'trocr'  # trocr (accurate, ~3 s/line) | simplehtr (fast); trocr needs its model
    context: bool = True             # accurate mode: reconsider unclear words using the sentence around them
    context_weight: float = 0.5      # how much the clinical language model's opinion counts
    vlm: bool = True                 # accurate mode: also ask a local vision-language model, if one is running
    vlm_veto: float = 25.0           # TrOCR rejects a vision reading only if it fits the image this much worse
    deskew: bool = True
    include_image: bool = False      # embed the (deskewed) page as a JPEG data URL, for the web UI


class Scrawl:
    def __init__(self, models_dir: Path = MODELS_DIR, custom_vocab: Path = CUSTOM_VOCAB) -> None:
        self.custom_vocab = custom_vocab
        hardware.configure_opencv()
        self.printed = PrintedReader()
        self.htr = HandwritingReader(models_dir, default_lexicon(SIMPLEHTR_DIR, [MEDICAL_VOCAB, custom_vocab]))
        self._trocr = None
        self._trocr_lock = threading.Lock()
        self._context = None
        self._context_lock = threading.Lock()
        self._vlm = VisionReader()

    def vlm(self):
        """The local vision-language model reader when its server is up and has the model, else None."""
        return self._vlm if self._vlm.available() else None

    def context(self):
        """The clinical language model, loaded on first use; None if it is not installed."""
        with self._context_lock:
            if self._context is None and clinical_context.available():
                self._context = clinical_context.ContextModel()
        return self._context

    def trocr(self):
        """The TrOCR reader, loaded on first use; None if it is not installed."""
        with self._trocr_lock:
            if self._trocr is None and trocr.available():
                self._trocr = trocr.TrOCRReader()
        return self._trocr

    def hardware(self) -> dict:
        """Where each engine runs, e.g. for the web UI's status line."""
        return {'cpu_threads': hardware.CPU_THREADS, 'detection_and_printed': self.printed.device,
                'simplehtr': self.htr.device_name,
                'trocr': self._trocr.device_name if self._trocr is not None else None,
                'context_model': hardware.torch_device_name(self._context.device) if self._context is not None else None,
                'vision_model': self._vlm.describe()}

    def reload_vocabulary(self) -> int:
        self.htr.lexicon = default_lexicon(SIMPLEHTR_DIR, [MEDICAL_VOCAB, self.custom_vocab])
        return len(self.htr.lexicon)

    def read_file(self, path: Path, options: Optional[Options] = None) -> dict:
        return self.read_bytes(Path(path).read_bytes(), str(path), options)

    def read_bytes(self, data: bytes, filename: str = '', options: Optional[Options] = None) -> dict:
        options = options or Options()
        start = time.time()
        pages = [self.read_page(img, options) for img in imaging.load_pages(data, filename)]
        for number, page in enumerate(pages, 1):
            page['page'] = number
        return {'source': Path(filename).name if filename else None, 'pages': pages,
                'options': asdict(options), 'seconds': round(time.time() - start, 2), 'hardware': self.hardware()}

    def read_page(self, bgr: np.ndarray, options: Options) -> dict:
        norm = imaging.normalize_illumination(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        pen = None
        if options.pen_layer and options.mode != 'printed':
            weight = imaging.pen_weight(bgr)
            if (weight > 0.5).sum() > MIN_PEN_FRACTION * weight.size:
                norm, pen = imaging.split_layers(bgr, norm, weight)

        clean = layout.remove_rules(norm, imaging.ink_mask(norm))
        quads = self.printed.detect(cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR))
        angle = imaging.skew_angle(quads) if options.deskew else 0.0
        if abs(angle) > 0.3:
            clean, quads = imaging.rotate(clean, angle, quads)
            bgr, _ = imaging.rotate(bgr, angle, [])
            clean = layout.remove_rules(clean, imaging.ink_mask(clean))  # tilted rules are only straight now
            if pen is not None:
                pen, _ = imaging.rotate(pen, angle, [])
        else:
            angle = 0.0
        height, width = clean.shape

        self._page_has_pen = pen is not None
        lines = self._read_layer(clean, quads, options, options.mode)
        if pen is not None:
            # coloured ink = what was filled in by hand; stamps are coloured too but sit at an angle
            pen_quads = [q for q in self.printed.detect(cv2.cvtColor(pen, cv2.COLOR_GRAY2BGR))
                         if abs(_quad_angle(q)) < MAX_LINE_TILT]
            for line in self._read_layer(pen, pen_quads, options, options.mode, pen=True):
                line['engine'] = 'handwriting'
                lines.append(line)

        built = structure.build(lines)
        changes = vlm_changes = 0
        if options.context and options.handwriting_model == 'trocr' and self.context() is not None:
            changes = self._reconstruct(built, options)
            if changes:
                built = structure.build(built['lines'])  # fields, tables and formatting see the new words
        if options.vlm and options.handwriting_model == 'trocr' and self.vlm() is not None:
            vlm_changes = self._vision_pass(built, options)
            built = structure.build(built['lines'])
        for line in built['lines']:
            line.pop('_scorer', None)
            line.pop('_crop', None)
        page = {'width': width, 'height': height, 'deskew_degrees': round(angle, 2),
                'pen_layer': pen is not None, 'context_changes': changes, 'vision_changes': vlm_changes, **built}
        if options.include_image:
            page['image'] = _jpeg_data_url(bgr)
        return page

    def _read_layer(self, clean: np.ndarray, quads: List[np.ndarray], options: Options, mode: str,
                    pen: bool = False) -> List[dict]:
        ink = imaging.ink_mask(clean)
        height, width = clean.shape
        regions = [layout.quad_to_region(q, width, height) for q in quads]
        regions = [r for r in regions if r.h >= 6 and r.x1 - r.x0 >= 4]
        self._read_printed(clean, regions)

        if mode == 'printed':
            printed, handwritten = regions, []
        elif mode == 'handwriting' or pen:
            printed, handwritten = [], regions
        else:
            regions = self._split_labels(clean, ink, regions, options.printed_threshold)
            printed = [r for r in regions if r.printed_score >= options.printed_threshold
                       and re.search(r'[A-Za-z0-9]', r.printed_text)]
            chosen = {id(r) for r in printed}
            handwritten = [r for r in regions if id(r) not in chosen]
            printed, handwritten = _absorb_into_handwriting(printed, handwritten, options.merge_gap)

        if mode != 'printed' and options.rescue:
            line_height = float(np.median([r.h for r in regions])) if regions else max(20.0, height / 40)
            handwritten += layout.rescue_missed_ink(ink, regions, line_height)
        handwritten = layout.merge_lines(handwritten, options.merge_gap)
        if pen:  # the pieces were read separately; read each merged line as a whole
            self._read_printed(clean, [r for r in handwritten if r.parts])

        lines = [{'engine': 'printed', 'reader': 'printed-ocr', 'text': r.printed_text,
                  'confidence': round(r.printed_score, 3), 'bbox': r.bbox} for r in printed
                 if r.printed_text and not _is_junk(r.printed_text, r.printed_score, False)]
        crops = [layout.line_crop(clean, ink, r) for r in handwritten]
        # TrOCR reads the handwriting itself; when a pen layer exists the print layer only holds printed text
        use_trocr = options.handwriting_model == 'trocr' and (pen or not options.pen_layer or mode != 'auto'
                                                              or not getattr(self, '_page_has_pen', False))
        trocr_reads = (self.trocr().read(crops, self.htr.lexicon, options.lexicon_strength)
                       if use_trocr and crops and self.trocr() else [None] * len(crops))
        for r, crop, deep in zip(handwritten, crops, trocr_reads):
            result = self.htr.read(crop, 'line', options.tta, options.lexicon_strength)
            line = {'engine': 'handwriting', 'reader': 'simplehtr', 'bbox': r.bbox, **(result or {'text': '', 'confidence': 0.0}),
                    'printed_reading': r.printed_text or None}
            if deep and deep['confidence'] >= trocr.MIN_CONFIDENCE and re.search(r'[A-Za-z0-9]', deep['text']):
                if _letters(deep['text']) == _letters(line['text']):  # both agree on the letters
                    line.update(text=_with_case_of(deep['text'], line['text']), reader='trocr+simplehtr')
                else:
                    line.update(text=deep['text'], confidence=deep['confidence'], reader='trocr',
                                simplehtr_reading=line['text'] or None, words=deep['words'])
                line['_scorer'] = deep['_scorer']
                if deep['raw_text'] != deep['text']:
                    line['trocr_reading'] = deep['raw_text']  # before vocabulary correction
            elif mode == 'auto' and _printed_reading_wins(r.printed_text, r.printed_score, line['confidence'], pen):
                line.update(text=r.printed_text, confidence=round(r.printed_score, 3), reader='printed-ocr',
                            printed_reading=None, simplehtr_reading=line['text'] or None, words=None)
            if _is_junk(line['text'], line['confidence'], r.rescued):
                continue
            if not pen and line['reader'] == 'printed-ocr':
                line['engine'] = 'printed'
            line['_crop'] = crop  # for the vision model pass
            lines.append(line)
        return lines

    def _vision_pass(self, built: dict, options: Options) -> int:
        """Ask the local vision-language model to read every handwritten line, then reconcile.

        The model never sees TrOCR's reading. Its reading wins unless TrOCR, scoring both against the image,
        finds it clearly incompatible: measured on a real referral letter, correct readings TrOCR had wrong
        scored -21..+9 against TrOCR's own, invented ones -32 or worse, hence `vlm_veto` = 25. Accepted changes
        to numbers or medicine names, doubtful acceptances and near-miss rejections are flagged for review."""
        reader, lexicon, veto = self.vlm(), self.htr.lexicon, options.vlm_veto
        lines, rows = built['lines'], built['rows']
        by_id = {line['id']: line for line in lines}
        row_of = {i: r for r, row in enumerate(rows) for i in row}
        changed = 0
        for line in lines:
            crop = line.get('_crop')
            if crop is None or line['engine'] != 'handwriting':
                continue
            row = rows[row_of[line['id']]]
            labels = [by_id[i]['text'] for i in row[:row.index(line['id'])] if by_id[i]['engine'] == 'printed']
            seen = reader.read(crop, structure._clean_label(labels[-1]) if labels else None)
            if not seen:
                continue
            current = line['text']
            line['vlm_reading'] = seen
            if _letters(seen) == _letters(current):  # same letters: keep the vision model's punctuation (N/A, not N.A)
                line.update(text=seen, readers_agree=True, confidence=max(line['confidence'], 0.9))
                continue
            scorer = line.get('_scorer')
            if scorer is None or '[?]' in seen:
                line['needs_review'] = True  # two readings, no way to check which fits the image
                continue
            ll_current, ll_seen = scorer([current, seen])
            gap = float(ll_seen - ll_current)  # how much worse the vision reading fits the image, per TrOCR
            accepted = gap > -veto
            if accepted:
                line.update(text=seen, reader='vlm', trocr_text=current,
                            words=[{'text': w, 'confidence': None, 'alternatives': [], 'read_as': None}
                                   for w in seen.split()])
                changed += 1
            risky = _digits(current) != _digits(seen) or bool(_drug_words(current, lexicon) ^ _drug_words(seen, lexicon))
            doubtful = accepted and gap < -veto / 2
            near_miss = not accepted and gap > -veto - 5
            # a reading that doesn't fit the image at all was most likely invented: kept only as an alternative
            if (accepted and risky) or doubtful or near_miss:
                line['needs_review'] = True
        for line in lines:
            if line.get('needs_review'):
                line['confidence'] = min(line['confidence'], 0.59)  # shows as "please check" everywhere
        return changed

    def _reconstruct(self, built: dict, options: Options) -> int:
        """Second look at handwritten words the reader was unsure of, using the text around them.

        Candidates: similar dictionary / clinical words plus words the clinical language model expects in the
        gap. Each is scored by TrOCR (does it match the handwriting?), the dictionary bonus and the language
        model (does it fit the sentence?). Names are never guessed, and a drug name that only context favours
        is offered as a suggestion instead of being filled in."""
        model, lexicon = self.context(), self.htr.lexicon
        lines, rows = built['lines'], built['rows']
        by_id = {line['id']: line for line in lines}
        row_of = {i: r for r, row in enumerate(rows) for i in row}

        def row_text(r: int) -> str:
            return ' '.join(by_id[i]['text'] for i in rows[r]) if 0 <= r < len(rows) else ''

        total = 0
        for line in lines:
            scorer, words = line.get('_scorer'), line.get('words')
            if scorer is None or not words:
                continue
            r = row_of[line['id']]
            k = rows[r].index(line['id'])
            before = [by_id[i]['text'] for i in rows[r][:k]]
            after = [by_id[i]['text'] for i in rows[r][k + 1:]]
            if any(NAME_FIELD.search(t) for t in before):
                continue  # names are never guessed
            texts, changed = [w['text'] for w in words], 0
            for wi, word in enumerate(words):
                match = WORD_RE.search(word['text'])
                if not match or len(match.group()) < 3 or re.search(r'\d', word['text']):
                    continue
                core, pre, post = match.group(), word['text'][:match.start()], word['text'][match.end():]
                if (core.isupper() and len(core) <= 4) or \
                        (core in lexicon and (word.get('confidence') is None or word['confidence'] >= CONTEXT_SURE)):
                    continue
                left = ' '.join([row_text(r - 1), *before, *texts[:wi]])
                right = ' '.join([*texts[wi + 1:], *after, row_text(r + 1)])
                options_ = [core] + [lexicon._matching_case(c, core) for c in lexicon.candidates(core, 6, common_only=True)]
                options_ += [lexicon._matching_case(p.lower(), core) for p in model.propose(left, right)
                             if fuzz.ratio(p.lower(), core.lower()) >= 50][:3]
                options_ = list(dict.fromkeys(options_))
                if len(options_) == 1:
                    continue
                variants = [' '.join(texts[:wi] + [pre + o + post] + texts[wi + 1:]) for o in options_]
                base = scorer(variants) + np.array([lexicon.bonus(o, options.lexicon_strength) for o in options_])
                top = sorted(set(np.argsort(-base)[:6].tolist()) | {0})
                fit = model.score(left, right, [options_[j] for j in top])
                best = top[int(np.argmax(base[top] + options.context_weight * fit))]
                if best == 0:
                    continue
                choice, original = pre + options_[best] + post, word['text']
                by_context = best != int(np.argmax(base))
                if by_context and options_[best].lower() in lexicon.drugs:
                    word['suggestion'] = choice  # a medicine guessed from context: shown, never filled in
                    word['alternatives'] = [choice] + [a for a in word.get('alternatives', []) if a != choice][:2]
                    continue
                word.update(text=choice, read_as=word.get('read_as') or original, inferred=by_context,
                            alternatives=[original] + [a for a in word.get('alternatives', []) if a not in (choice, original)][:2])
                texts[wi] = choice
                changed += 1
            if changed:
                line['text'] = tidy_punctuation(' '.join(texts))
                line['context_changes'] = changed
                total += changed
        return total

    def _read_printed(self, clean: np.ndarray, regions: List[Region]) -> None:
        crops = [cv2.cvtColor(clean[r.y0:r.y1, r.x0:r.x1], cv2.COLOR_GRAY2BGR) for r in regions]
        for r, (text, score) in zip(regions, self.printed.recognize(crops)):
            r.printed_text, r.printed_score = text, score

    def _split_labels(self, clean: np.ndarray, ink: np.ndarray, regions: List[Region], threshold: float) -> List[Region]:
        """A printed 'Label:' and a handwritten answer are often detected as one box; split them at the gap."""
        out, new = [], []
        for r in regions:
            match = LABEL_VALUE_RE.match(r.printed_text)
            if r.printed_score >= threshold or not match:
                out.append(r)
                continue
            box_w = r.x1 - r.x0
            guess = int(len(match.group(1)) / len(r.printed_text) * box_w)
            cols = ink[r.y0:r.y1, r.x0:r.x1].sum(axis=0)
            lo, hi = max(1, guess - box_w // 6), min(box_w - 1, guess + box_w // 6)
            split = r.x0 + _widest_gap(cols[lo:hi], lo, guess)
            left, right = Region(r.x0, r.y0, split, r.y1), Region(split, r.y0, r.x1, r.y1)
            out += [left, right]
            new += [left, right]
        self._read_printed(clean, new)
        return out


def _digits(text: str) -> str:
    return re.sub(r'\D', '', text)


def _drug_words(text: str, lexicon) -> set:
    return {w.lower() for w in WORD_RE.findall(text) if w.lower() in lexicon.drugs}


def _letters(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', text.lower())


def _with_case_of(text: str, reference: str) -> str:
    """TrOCR's punctuation (better) with SimpleHTR's capitals (TrOCR tends to lower-case names)."""
    cases = iter(c.isupper() for c in reference if c.isalnum())
    out = ''.join((c.upper() if next(cases, False) else c.lower()) if c.isalnum() else c for c in text)
    if out.endswith('.') and not reference.rstrip().endswith('.'):  # TrOCR likes to end lines with a full stop
        out = out[:-1].rstrip()
    return out


def _quad_angle(quad: np.ndarray) -> float:
    top = quad[1] - quad[0]
    return float(np.degrees(np.arctan2(top[1], top[0])))


def _printed_reading_wins(printed_text: str, printed_score: float, htr_confidence: float, pen: bool) -> bool:
    """Neither model is best everywhere. SimpleHTR is far better on cursive, but the printed model reads
    digits, dates and IDs better and handles printed text that ended up on the handwriting path. On cursive the
    printed model's confidence runs high even when it is wrong, so in the pen layer it needs a clear win."""
    if not re.search(r'[A-Za-z0-9]', printed_text):
        return False
    if not pen:
        return printed_score > htr_confidence
    digits = len(re.findall(r'[0-9]', printed_text)) / max(1, len(re.findall(r'[A-Za-z0-9]', printed_text)))
    return digits >= 0.5 and printed_score > 0.4 or printed_score >= max(0.92, htr_confidence + 0.1)


def _is_junk(text: str, confidence: float, rescued: bool) -> bool:
    """Logos, stamps, signatures and specks come out as a few random characters at low confidence."""
    alnum = len(re.findall(r'[A-Za-z0-9]', text))
    return (alnum == 0 or confidence < 0.12 or (confidence < 0.3 and alnum < 4)
            or (rescued and confidence < 0.6))


def _absorb_into_handwriting(printed: List[Region], handwritten: List[Region], gap_factor: float):
    """The printed model sometimes reads a handwritten word confidently ("beastly" -> "beasH" at 0.93).
    A not-quite-certain printed box that continues a handwritten line is treated as handwriting."""
    changed = True
    while changed:
        changed = False
        for p in list(printed):
            if p.printed_score >= SURE_PRINTED:
                continue
            for h in handwritten:
                gap = max(p.x0, h.x0) - min(p.x1, h.x1)
                if layout.same_line(p, h) and gap <= gap_factor * min(p.h, h.h):
                    printed.remove(p)
                    handwritten.append(p)
                    changed = True
                    break
    return printed, handwritten


def _widest_gap(cols: np.ndarray, offset: int, fallback: int) -> int:
    best, best_len, run_start = fallback, 0, None
    for i, value in enumerate(list(cols) + [1]):
        if value == 0 and run_start is None:
            run_start = i
        elif value != 0 and run_start is not None:
            if i - run_start > best_len:
                best, best_len = offset + (run_start + i) // 2, i - run_start
            run_start = None
    return best


def _jpeg_data_url(bgr: np.ndarray, max_side: int = 1800) -> str:
    scale = min(1.0, max_side / max(bgr.shape[:2]))
    if scale < 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()
