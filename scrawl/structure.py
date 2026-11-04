"""Structure from recognised lines: reading order, key/value fields, tables and typed values."""
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from . import document

LABEL_RE = re.compile(r"^\s*([A-Za-z][^:\s]*(?:\s+[^:\s]+){0,3}?)\s*:\s*(.*)$")  # label: at most 4 words
MONTH = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?'
ENTITY_PATTERNS = [
    ('email', r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+'),
    ('url', r'\bhttps?://\S+|\bwww\.\S+'),
    ('date', rf'\b\d{{4}}-\d{{1,2}}-\d{{1,2}}\b|\b\d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{2,4}}\b'
             rf'|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH},?\s+\d{{2,4}}\b|\b{MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{2,4}}\b'),
    ('time', r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[aApP]\.?[mM]\.?)?(?!\d)'),
    ('blood_pressure', r'\b\d{2,3}\s*/\s*\d{2,3}\s*mm\s*Hg\b|(?<=BP)\W{0,3}\d{2,3}\s*/\s*\d{2,3}\b'),
    ('quantity', r'\b\d+(?:[.,]\d+)?\s*(?:mg/dL|mmol/L|mcg|mg|µg|kg|g|ml|mL|L|IU|units?|tabs?|tablets?|caps?|capsules?'
                 r'|drops?|puffs?|bpm|cm|mm|%)(?![A-Za-z])'),
    ('money', r'[$€£]\s?\d[\d,]*(?:\.\d{1,2})?'),
    ('phone', r'(?<![\w/.-])\+?\(?\d[\d\s().-]{7,}\d(?![\w/])'),
]


def group_rows(lines: List[dict]) -> List[List[dict]]:
    """Lines whose vertical centres fall inside each other's extent form a row; rows top to bottom, cells left to right."""
    rows: List[List[dict]] = []
    for line in sorted(lines, key=lambda l: l['bbox'][1] - l['bbox'][3]):  # tallest first, small marks join them
        x0, y0, x1, y1 = line['bbox']
        cy = (y0 + y1) / 2
        for row in rows:
            ry0 = np.median([c['bbox'][1] for c in row])
            ry1 = np.median([c['bbox'][3] for c in row])
            overlap = min(y1, ry1) - max(y0, ry0)
            mutual = (ry0 <= cy <= ry1 and y0 <= (ry0 + ry1) / 2 <= y1) or overlap >= 0.5 * min(y1 - y0, ry1 - ry0)
            is_mark = y1 - y0 < 0.7 * (ry1 - ry0) and x1 - x0 <= 2 * (y1 - y0) and overlap >= 0.3 * (y1 - y0)
            if mutual or is_mark:
                row.append(line)
                break
        else:
            rows.append([line])
    rows = _fold_punctuation_rows(rows)
    for row in rows:
        row.sort(key=lambda l: l['bbox'][0])
    rows.sort(key=lambda r: min(c['bbox'][1] for c in r))
    return rows


def _fold_punctuation_rows(rows: List[List[dict]]) -> List[List[dict]]:
    """A row holding nothing but stray punctuation (a raised ?' or a comma) joins the row it touches most."""
    def extent(row):
        return min(c['bbox'][1] for c in row), max(c['bbox'][3] for c in row)

    kept = [r for r in rows if any(re.search(r'\w', c['text']) for c in r)]
    for row in rows:
        if row in kept:
            continue
        y0, y1 = extent(row)
        overlaps = [min(y1, extent(k)[1]) - max(y0, extent(k)[0]) for k in kept]
        if overlaps and max(overlaps) > 0:
            kept[int(np.argmax(overlaps))].extend(row)
        else:
            kept.append(row)
    return kept


def row_text(row: List[dict]) -> str:
    out = row[0]['text']
    for prev, cur in zip(row, row[1:]):
        gap = cur['bbox'][0] - prev['bbox'][2]
        height = min(prev['bbox'][3] - prev['bbox'][1], cur['bbox'][3] - cur['bbox'][1])
        if not re.search(r'\w', cur['text']):  # stray punctuation detected on its own
            out += cur['text']
        else:
            out += ('\t' if gap > 2 * height else ' ') + cur['text']
    return out


def _is_label(text: str) -> bool:
    return bool(LABEL_RE.match(text)) and not text.strip()[0].isdigit()


def _clean_label(text: str) -> str:
    return re.sub(r'[\s.:…_]+$', '', re.sub(r'^[\s.…_]+', '', text))


def _clean_value(text: str) -> str:
    return re.sub(r'^[\s.…_:]+', '', text).strip()  # dotted leaders: "Name:........Jane"


def _is_printed_label(cell: dict) -> bool:
    """A short printed caption ("Referring doctor", "Clinical History....") - on forms the handwriting next to
    it is its answer, colon or not."""
    text = _clean_label(cell['text'])
    letters = sum(c.isalpha() for c in text)
    return (cell.get('engine') == 'printed' and 2 <= letters and len(text) <= 40 and len(text.split()) <= 6
            and letters >= 0.6 * len(text.replace(' ', '')))


def find_fields(rows: List[List[dict]]) -> List[dict]:
    """Label/value pairs: "Label: value" inline, value in the next cell or on the line below the label, and a
    printed caption directly followed by handwriting."""
    fields = []
    for r, row in enumerate(rows):
        k = 0
        while k < len(row):
            cell = row[k]
            nxt = row[k + 1] if k + 1 < len(row) else None
            match = LABEL_RE.match(cell['text'])
            if match and not cell['text'].strip()[0].isdigit():
                label, value, ids = match.group(1).strip(), _clean_value(match.group(2)), [cell['id']]
                if not value and nxt is not None and not _is_label(nxt['text']):
                    k += 1
                    value, ids = nxt['text'], ids + [nxt['id']]
                elif not value and r + 1 < len(rows):
                    below = _cell_below(cell, rows[r + 1])
                    if below is not None:
                        value, ids = below['text'], ids + [below['id']]
            elif nxt is not None and _is_printed_label(cell) and nxt.get('engine') == 'handwriting':
                answer = []  # every handwritten piece up to the next printed cell
                while k + 1 < len(row) and row[k + 1].get('engine') == 'handwriting':
                    k += 1
                    answer.append(row[k])
                label, value = _clean_label(cell['text']), _clean_value(' '.join(c['text'] for c in answer))
                ids = [cell['id']] + [c['id'] for c in answer]
            else:
                k += 1
                continue
            if value:
                fields.append({'label': _clean_label(label), 'value': value, 'line_ids': ids})
            k += 1
    return fields


def _cell_below(cell: dict, next_row: List[dict]) -> Optional[dict]:
    x0, y0, x1, y1 = cell['bbox']
    for cand in next_row:
        cx0, cy0, cx1, _ = cand['bbox']
        if min(x1, cx1) - max(x0, cx0) > 0 and cy0 - y1 < 2.5 * (y1 - y0) and not _is_label(cand['text']):
            return cand
    return None


def find_tables(rows: List[List[dict]]) -> List[dict]:
    """Runs of consecutive multi-cell rows whose cells line up in columns."""
    tables, run = [], []
    for row in rows + [[]]:
        if len(row) >= 2 and not _is_label(row[0]['text']):
            run.append(row)
            continue
        if len(run) >= 2:
            table = _build_table(run)
            if table:
                tables.append(table)
        run = []
    return tables


def _build_table(run: List[List[dict]]) -> Optional[dict]:
    cells = [c for row in run for c in row]
    if any(LABEL_RE.match(c['text']) for c in cells) or any(len(c['text'].split()) > 5 for c in run[0]):
        return None  # "Label: value" cells or sentence-long headers: a form laid out in columns, not a table
    answered = sum(_is_printed_label(r[0]) and any(c.get('engine') == 'handwriting' for c in r[1:]) for r in run)
    if answered >= len(run) / 2:
        return None  # printed captions with handwritten answers: a form, reported as fields
    num_cols, support = Counter(len(r) for r in run).most_common(1)[0]
    if support < 2:
        return None
    reference = [r for r in run if len(r) == num_cols]
    spans = [(np.median([r[j]['bbox'][0] for r in reference]), np.median([r[j]['bbox'][2] for r in reference]))
             for j in range(num_cols)]

    grid, ids = [], []
    for row in run:
        cells = [''] * num_cols
        for cell in row:
            x0, _, x1, _ = cell['bbox']
            overlaps = [min(x1, s1) - max(x0, s0) for s0, s1 in spans]
            centre = (x0 + x1) / 2
            j = int(np.argmax(overlaps)) if max(overlaps) > 0 else \
                int(np.argmin([abs(centre - (s0 + s1) / 2) for s0, s1 in spans]))
            cells[j] = f"{cells[j]} {cell['text']}".strip()
            ids.append(cell['id'])
        grid.append(cells)
    return {'columns': num_cols, 'rows': grid, 'line_ids': ids}


def find_entities(lines: List[dict]) -> List[dict]:
    found = []
    for line in lines:
        taken: List[tuple] = []
        for kind, pattern in ENTITY_PATTERNS:
            for m in re.finditer(pattern, line['text'], flags=re.IGNORECASE if kind == 'time' else 0):
                span = m.span()
                if any(span[0] < e and s < span[1] for s, e in taken):
                    continue
                if kind == 'phone' and len(re.sub(r'\D', '', m.group())) < 9:
                    continue
                taken.append(span)
                found.append({'type': kind, 'text': m.group().strip(), 'line_id': line['id']})
    return found


def build(lines: List[dict]) -> Dict:
    """Assign reading-order ids to lines and derive text, rows, fields, tables and entities."""
    rows = group_rows(lines)
    ordered = [cell for row in rows for cell in row]
    for i, line in enumerate(ordered):
        line['id'] = i
    row_ids = [[c['id'] for c in r] for r in rows]
    fields, tables = find_fields(rows), find_tables(rows)
    blocks = document.build_blocks(ordered, row_ids, fields, tables)
    return {
        'text': '\n'.join(row_text(r) for r in rows),           # plain reading order
        'formatted_text': document.render_text(blocks),          # headings, aligned tables, paragraphs
        'document': blocks,                                      # the same, as blocks for the web UI / HTML
        'lines': ordered,
        'rows': row_ids,
        'fields': fields,
        'tables': tables,
        'entities': find_entities(ordered),
    }
