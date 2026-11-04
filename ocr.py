#!/usr/bin/env python3
"""Read handwriting and printed documents locally.

  ./ocr.py note.jpg                         plain text to stdout
  ./ocr.py form.pdf --format json -o out.json
  ./ocr.py form.pdf --format html -o form.html  formatted, printable document
  ./ocr.py scans/*.png --format md -o results/   one file per input
  ./ocr.py note.jpg --overlay boxes.png     also save the page with detected lines drawn on it
"""
import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Local OCR for messy handwriting and printed/structured documents.')
    parser.add_argument('inputs', nargs='+', type=Path, help='images (png, jpg, tif, ...) or PDFs')
    parser.add_argument('--mode', choices=['auto', 'handwriting', 'printed'], default='auto',
                        help='auto decides per line; force one engine if you know what the page is')
    parser.add_argument('--format', choices=['txt', 'md', 'html', 'json', 'csv'], default='txt',
                        help='txt/md/html are formatted (headings, fields, tables); json has everything')
    parser.add_argument('-o', '--output', type=Path, help='output file (one input) or directory (several inputs)')
    parser.add_argument('--lexicon-strength', type=float, default=3.0,
                        help='how strongly to prefer dictionary words for handwriting (0 = off, default 3)')
    parser.add_argument('--printed-threshold', type=float, default=0.9,
                        help='auto mode: confidence the printed model needs before its reading is used')
    parser.add_argument('--fast', action='store_true',
                        help='read handwriting with SimpleHTR only (~10x faster, less accurate than TrOCR)')
    parser.add_argument('--no-tta', action='store_true', help='with --fast: read each handwritten line once')
    parser.add_argument('--no-context', action='store_true',
                        help="don't reconsider unclear handwritten words using the surrounding text")
    parser.add_argument('--no-rescue', action='store_true', help="don't search for handwriting the detector missed")
    parser.add_argument('--overlay', type=Path, help='save the first page with detected lines drawn on it (PNG)')
    args = parser.parse_args()

    from scrawl import Options, Scrawl
    from scrawl.export import FORMATS

    options = Options(mode=args.mode, tta=not args.no_tta, lexicon_strength=args.lexicon_strength,
                      printed_threshold=args.printed_threshold, rescue=not args.no_rescue,
                      handwriting_model='simplehtr' if args.fast else 'trocr', context=not args.no_context,
                      include_image=bool(args.overlay))
    engine = Scrawl()
    many = len(args.inputs) > 1
    if many and args.output:
        args.output.mkdir(parents=True, exist_ok=True)

    for path in args.inputs:
        result = engine.read_file(path, options)
        rendered = FORMATS[args.format](result)
        if args.output:
            target = args.output / f'{path.stem}.{args.format}' if many else args.output
            target.write_text(rendered, encoding='utf-8')
            print(f'{path} -> {target}  ({result["seconds"]}s)', file=sys.stderr)
        else:
            if many:
                print(f'==> {path} <==')
            sys.stdout.write(rendered)
        if args.overlay:
            save_overlay(result['pages'][0], args.overlay if not many else args.overlay.with_stem(
                f'{args.overlay.stem}_{path.stem}'))


def save_overlay(page: dict, target: Path) -> None:
    import base64

    import cv2
    import numpy as np

    data = base64.b64decode(page['image'].split(',', 1)[1])
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    scale = img.shape[1] / page['width']
    colours = {'handwriting': (40, 120, 230), 'printed': (60, 160, 60)}
    for line in page['lines']:
        x0, y0, x1, y1 = (int(v * scale) for v in line['bbox'])
        cv2.rectangle(img, (x0, y0), (x1, y1), colours[line['engine']], 2)
        cv2.putText(img, str(line['id']), (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colours[line['engine']], 1, cv2.LINE_AA)
    cv2.imwrite(str(target), img)
    print(f'overlay -> {target}', file=sys.stderr)


if __name__ == '__main__':
    main()
