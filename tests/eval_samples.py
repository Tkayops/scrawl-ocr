#!/usr/bin/env python3
"""Character error rate of the full pipeline on the bundled sample pages (ground truth typed by hand).

  .venv/bin/python tests/eval_samples.py [--mode handwriting] [--no-tta] [--lexicon-strength 0]
"""
import argparse
import sys
import time
from pathlib import Path

import editdistance

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GROUND_TRUTH = {
    'sample_1.png': """This is a handwritten text recognition
pipeline that operates on scanned pages
and applies the following operations
Detect words
Read words""",
    'sample_2.png': """The doorman turned his attention
to the next red-eyed emerger from the
dark; and we went on together to
the station, the children silent because
of the cruelty of the world. Finally Catherine
said, her eyes wet again: 'I think its all
absolutely beastly, and I can't bear to think
about it.' And Philip said: 'But we've got
to think about it, don't you see, because
if we don't it'll just go on and on, don't
you see?'""",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='auto')
    parser.add_argument('--no-tta', action='store_true')
    parser.add_argument('--lexicon-strength', type=float, default=3.0)
    parser.add_argument('--show', action='store_true', help='print every line')
    args = parser.parse_args()

    from scrawl import Options, Scrawl
    engine = Scrawl()
    options = Options(mode=args.mode, tta=not args.no_tta, lexicon_strength=args.lexicon_strength)
    total_err = total_len = 0
    for name, truth in GROUND_TRUTH.items():
        start = time.time()
        page = engine.read_file(ROOT / 'samples' / name, options)['pages'][0]
        got = page['text'].replace('\t', ' ')
        err = editdistance.eval(' '.join(got.split()), ' '.join(truth.split()))
        total_err += err
        total_len += len(truth)
        print(f'{name}: CER {100 * err / len(truth):.1f}%  ({time.time() - start:.1f}s)')
        if args.show:
            for g, t in zip(got.split('\n'), truth.split('\n') + [''] * 99):
                print(f'   {"  " if g == t else "x "}{g!r}')
    print(f'overall CER {100 * total_err / total_len:.1f}%')


if __name__ == '__main__':
    main()
