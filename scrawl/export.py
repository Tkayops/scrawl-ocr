"""Render a result as formatted text, Markdown, HTML, CSV or JSON."""
import csv
import io
import json

from . import document


def to_text(result: dict) -> str:
    """Formatted plain text: headings underlined, fields as "Label: value", tables in aligned columns."""
    pages = result['pages']
    texts = [(p.get('formatted_text') or p['text'] + '\n') + document.render_suggestions_text(p.get('suggestions', []))
             for p in pages]
    if len(pages) == 1:
        return texts[0]
    return '\n'.join(f"--- page {p['page']} ---\n{t}" for p, t in zip(pages, texts))


def to_csv(result: dict) -> str:
    """One row per recognised line."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['page', 'line', 'engine', 'confidence', 'x0', 'y0', 'x1', 'y1', 'text'])
    for page in result['pages']:
        for line in page['lines']:
            writer.writerow([page['page'], line['id'], line['engine'], line['confidence'], *line['bbox'], line['text']])
    return buf.getvalue()


def to_markdown(result: dict) -> str:
    out = []
    for page in result['pages']:
        if len(result['pages']) > 1:
            out.append(f"# Page {page['page']}\n")
        out.append(document.render_markdown(page['document']))
        if page.get('suggestions'):
            out.append('#### Suggested by the vision model (not applied, please check)\n')
            out += [f"- **{s['label']}:** {s['value']} _({s['reason']})_" for s in page['suggestions']]
            out.append('')
        if page['entities']:
            out.append('#### Detected values\n')
            out += [f"- **{e['type'].replace('_', ' ')}**: {e['text']}" for e in page['entities']]
            out.append('')
    return '\n'.join(out)


def to_html(result: dict) -> str:
    return document.render_html(result)


def to_json(result: dict) -> str:
    slim = {**result, 'pages': [{k: v for k, v in p.items() if k != 'image'} for p in result['pages']]}
    return json.dumps(slim, indent=2, ensure_ascii=False) + '\n'


FORMATS = {'txt': to_text, 'md': to_markdown, 'html': to_html, 'csv': to_csv, 'json': to_json}
