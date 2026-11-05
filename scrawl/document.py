"""Formatted version of a page: headings, label/value rows, paragraphs and tables, in reading order.

Blocks (JSON-friendly, also rendered by the web UI):
  {'type': 'heading',   'level': 1|2, 'text', 'line_ids'}
  {'type': 'row',       'items': [{'label' (or None), 'text', 'line_ids', 'handwritten', 'confidence'}]}
  {'type': 'paragraph', 'text', 'line_ids', 'handwritten', 'confidence', 'lines' (how many lines it joins)}
  {'type': 'table',     'index', 'rows', 'line_ids'}
"""
import html
import re
import textwrap
from typing import Dict, List

import numpy as np

LOW_CONFIDENCE = 0.6


def _looks_like_heading(text: str) -> bool:
    """Section titles on forms are short and in capitals: "MEMBER PARTICULARS", "TREATMENT DETAILS (...)"."""
    core = re.sub(r'\(.*?\)?$', '', text).strip()
    letters = [c for c in core if c.isalpha()]
    if len(letters) < 4 or len(core.split()) > 8 or core.endswith(':'):
        return False
    digits = sum(c.isdigit() for c in core)
    return sum(c.isupper() for c in letters) / len(letters) >= 0.8 and digits <= 2


def build_blocks(lines: List[dict], rows: List[List[int]], fields: List[dict], tables: List[dict]) -> List[dict]:
    by_id = {line['id']: line for line in lines}
    heights = [l['bbox'][3] - l['bbox'][1] for l in lines if l['engine'] == 'printed'] or \
              [l['bbox'][3] - l['bbox'][1] for l in lines] or [20]
    line_height = float(np.median(heights))
    field_of = {i: n for n, f in enumerate(fields) for i in f['line_ids']}
    table_of = {i: n for n, t in enumerate(tables) for i in t['line_ids']}
    used, done_fields, done_tables = set(), set(), set()

    blocks: List[dict] = []
    for row in rows:
        ids = [i for i in row if i not in used]
        if not ids:
            continue
        if all(i in table_of for i in ids):
            for n in sorted({table_of[i] for i in ids} - done_tables):
                blocks.append({'type': 'table', 'index': n, 'rows': tables[n]['rows'], 'line_ids': tables[n]['line_ids']})
                done_tables.add(n)
            used.update(ids)
            continue

        if len(ids) == 1 and ids[0] not in field_of:
            line = by_id[ids[0]]
            height = line['bbox'][3] - line['bbox'][1]
            big = height >= 1.2 * line_height and len(line['text'].split()) <= 8 and \
                sum(c.isdigit() for c in line['text']) <= 2 and not re.search(r'@|www\.|https?:', line['text'])
            known = line.get('heading')  # readers that find titles themselves (MinerU) give a level, or False
            if known or (known is None and line['engine'] == 'printed' and (big or _looks_like_heading(line['text']))):
                blocks.append({'type': 'heading', 'level': known or (1 if big else 2), 'text': line['text'], 'line_ids': ids})
                used.update(ids)
                continue

        items = []
        for i in ids:
            if i in used:
                continue
            if i in field_of:
                n = field_of[i]
                if n in done_fields:
                    continue
                field, done_fields = fields[n], done_fields | {n}
                value_lines = [by_id[j] for j in field['line_ids'][1:]] or [by_id[field['line_ids'][0]]]
                items.append({'label': field['label'], 'text': field['value'], 'line_ids': field['line_ids'],
                              'handwritten': any(l['engine'] == 'handwriting' for l in value_lines),
                              'confidence': min(by_id[j]['confidence'] for j in field['line_ids'])})
                used.update(field['line_ids'])
            else:
                line = by_id[i]
                items.append({'label': None, 'text': line['text'], 'line_ids': [i],
                              'handwritten': line['engine'] == 'handwriting', 'confidence': line['confidence']})
                used.add(i)
        if items and _continues_field(blocks, items, by_id, line_height):
            field = blocks[-1]['items'][-1]  # a handwritten answer running on to the next line
            field['text'] = f"{field['text']} {' '.join(i['text'] for i in items)}"
            field['line_ids'] = field['line_ids'] + [j for i in items for j in i['line_ids']]
            field['confidence'] = min([field['confidence']] + [i['confidence'] for i in items])
            blocks[-1]['_bottom'] = max(by_id[j]['bbox'][3] for i in items for j in i['line_ids'])
            continue
        if items:
            boxes = [by_id[j]['bbox'] for it in items for j in it['line_ids']]
            blocks.append({'type': 'row', 'items': items, '_top': min(b[1] for b in boxes),
                           '_bottom': max(b[3] for b in boxes), '_left': min(b[0] for b in boxes)})
    blocks = _paragraphs(blocks, line_height)
    heads = [b for b in blocks[:3] if b['type'] == 'heading']
    if heads and not any(h['level'] == 1 for h in heads):
        heads[0]['level'] = 1  # a heading at the very top of the page is its title
    return blocks


def _continues_field(blocks: List[dict], items: List[dict], by_id: Dict[int, dict], line_height: float) -> bool:
    """Only handwriting on this row, right under a handwritten field value: the answer's second line."""
    if not blocks or blocks[-1]['type'] != 'row' or any(i['label'] or not i['handwritten'] for i in items):
        return False
    last = blocks[-1]['items'][-1]
    top = min(by_id[j]['bbox'][1] for i in items for j in i['line_ids'])
    return bool(last['label']) and last['handwritten'] and top - blocks[-1]['_bottom'] < 1.0 * line_height


def _paragraphs(blocks: List[dict], line_height: float) -> List[dict]:
    """Consecutive single, unlabelled lines that sit close together and line up on the left form a paragraph."""
    out: List[dict] = []
    for block in blocks:
        if block['type'] == 'row' and len(block['items']) == 1 and block['items'][0]['label'] is None:
            item = block['items'][0]
            prev = out[-1] if out else None
            if (prev is not None and prev['type'] == 'paragraph' and prev['handwritten'] == item['handwritten']
                    and block['_top'] - prev['_bottom'] < 0.8 * line_height
                    and abs(block['_left'] - prev['_left']) < 3 * line_height):
                prev['text'] = f"{prev['text']} {item['text']}"
                prev['line_ids'] = prev['line_ids'] + item['line_ids']
                prev['confidence'] = min(prev['confidence'], item['confidence'])
                prev['lines'] += 1
                prev['_bottom'] = block['_bottom']
                continue
            block = {'type': 'paragraph', 'lines': 1, **{k: v for k, v in item.items() if k != 'label'},
                     '_top': block['_top'], '_bottom': block['_bottom'], '_left': block['_left']}
        out.append(block)
    return [{k: v for k, v in b.items() if not k.startswith('_')} for b in out]


# ---------------------------------------------------------------- renderers

def render_text(blocks: List[dict], width: int = 100) -> str:
    out: List[str] = []

    def gap():
        if out and out[-1] != '':
            out.append('')

    for b in blocks:
        if b['type'] == 'heading':
            gap()
            out += [b['text'], ('=' if b['level'] == 1 else '-') * len(b['text'])]
        elif b['type'] == 'row':
            out.append('    '.join(f"{i['label']}: {i['text']}" if i['label'] else i['text'] for i in b['items']))
        elif b['type'] == 'paragraph':
            if b['lines'] > 1:
                gap()
            out += textwrap.wrap(b['text'], width) or ['']
            if b['lines'] > 1:
                out.append('')
        elif b['type'] == 'table':
            rows = b['rows']
            widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
            gap()
            for k, r in enumerate(rows):
                out.append('  '.join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip())
                if k == 0:
                    out.append('  '.join('-' * w for w in widths))
            out.append('')
    text = '\n'.join(out)
    return re.sub(r'\n{3,}', '\n\n', text).strip() + '\n'


def _md(text: str) -> str:
    return re.sub(r'([\\`*_|<>])', r'\\\1', text)


def render_markdown(blocks: List[dict]) -> str:
    out: List[str] = []
    for b in blocks:
        if b['type'] == 'heading':
            out += ['', f"{'##' if b['level'] == 1 else '###'} {_md(b['text'])}", '']
        elif b['type'] == 'row':
            out.append(' · '.join(f"**{_md(i['label'])}:** {_md(i['text'])}" if i['label'] else _md(i['text'])
                                  for i in b['items']) + '  ')
        elif b['type'] == 'paragraph':
            out += (['', _md(b['text']), ''] if b['lines'] > 1 else [_md(b['text']) + '  '])
        elif b['type'] == 'table':
            rows = b['rows']
            out += ['', '| ' + ' | '.join(_md(c) for c in rows[0]) + ' |', '|' + '---|' * len(rows[0])]
            out += ['| ' + ' | '.join(_md(c) for c in r) + ' |' for r in rows[1:]]
            out.append('')
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(out)).strip() + '\n'


HTML_STYLE = """
body { font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: #1c1d21; background: #fff;
       max-width: 860px; margin: 32px auto; padding: 0 20px; }
h1 { font-size: 20px; } h2 { font-size: 19px; margin: 22px 0 6px; } h3 { font-size: 14px; letter-spacing: .04em;
     text-transform: uppercase; color: #555; margin: 20px 0 6px; border-bottom: 1px solid #ddd; padding-bottom: 3px; }
.row { display: flex; flex-wrap: wrap; gap: 4px 28px; margin: 3px 0; }
.label { font-weight: 600; color: #444; }
p { margin: 3px 0; } p.para { margin: 10px 0; }
.hand { color: #1d4ed8; }
.low { text-decoration: underline dotted #d9480f 2px; text-underline-offset: 3px; }
table { border-collapse: collapse; margin: 12px 0; width: 100%; }
th, td { border: 1px solid #ccc; padding: 5px 9px; text-align: left; vertical-align: top; }
th { background: #f3f3f3; }
.page { page-break-after: always; } .page:last-child { page-break-after: auto; }
.meta { color: #666; font-size: 13px; margin-bottom: 18px; }
.legend { color: #666; font-size: 12.5px; }
"""


def _span(item: dict) -> str:
    classes = ' '.join(c for c, on in (('hand', item['handwritten']),
                                        ('low', item['confidence'] < LOW_CONFIDENCE)) if on)
    title = f' title="read with {round(item["confidence"] * 100)}% confidence"' if item['confidence'] < LOW_CONFIDENCE else ''
    return f'<span class="{classes}"{title}>{html.escape(item["text"])}</span>'


def render_html_blocks(blocks: List[dict]) -> str:
    out = []
    for b in blocks:
        if b['type'] == 'heading':
            tag = 'h2' if b['level'] == 1 else 'h3'
            out.append(f'<{tag}>{html.escape(b["text"])}</{tag}>')
        elif b['type'] == 'row':
            cells = ''.join(f'<div>{"<span class=label>" + html.escape(i["label"]) + ":</span> " if i["label"] else ""}'
                            f'{_span(i)}</div>' for i in b['items'])
            out.append(f'<div class="row">{cells}</div>')
        elif b['type'] == 'paragraph':
            out.append(f'<p class="{"para" if b["lines"] > 1 else ""}">{_span(b)}</p>')
        elif b['type'] == 'table':
            head, *body = b['rows']
            out.append('<table><thead><tr>' + ''.join(f'<th>{html.escape(c)}</th>' for c in head) + '</tr></thead><tbody>'
                       + ''.join('<tr>' + ''.join(f'<td>{html.escape(c)}</td>' for c in r) + '</tr>' for r in body)
                       + '</tbody></table>')
    return '\n'.join(out)


def render_suggestions_text(suggestions: List[dict]) -> str:
    if not suggestions:
        return ''
    out = ['', 'Suggested by the vision model - NOT applied, please check against the document:']
    out += [f"  {s['label']}: {s['value']}  ({s['reason']})" for s in suggestions]
    return '\n'.join(out) + '\n'


def render_suggestions_html(suggestions: List[dict]) -> str:
    if not suggestions:
        return ''
    rows = ''.join(f'<tr><td>{html.escape(s["label"])}</td><td class="low">{html.escape(s["value"])}</td>'
                   f'<td>{html.escape(s["reason"])}</td></tr>' for s in suggestions)
    return ('<h3>Suggested by the vision model: not applied, please check</h3>'
            f'<table><thead><tr><th>Field</th><th>Suggested value</th><th>Why it is only a suggestion</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def render_html(result: Dict) -> str:
    title = html.escape(result.get('source') or 'Scanned document')
    pages = result['pages']
    body = ''.join(
        f'<section class="page">{"<h1>Page " + str(p["page"]) + "</h1>" if len(pages) > 1 else ""}'
        f'{render_html_blocks(p["document"])}{render_suggestions_html(p.get("suggestions", []))}</section>' for p in pages)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><style>{HTML_STYLE}</style></head>'
            f'<body><div class="meta">{title} · read locally by Scrawl</div>'
            f'<div class="legend"><span class="hand">Blue</span> = handwritten · '
            f'<span class="low">dotted</span> = low confidence, please check</div>{body}</body></html>\n')
