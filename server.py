#!/usr/bin/env python3
"""Local web UI for Scrawl.  ./server.py  then open http://127.0.0.1:8765

Everything runs on this machine: the server only listens on 127.0.0.1 and the page loads nothing from
the internet, so documents never leave the computer.
"""
import argparse
import threading

from flask import Flask, Response, jsonify, request, send_from_directory

from scrawl.paths import CUSTOM_VOCAB, ROOT, WEB_DIR

app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = 60 * 1024 * 1024

_engine = None
_engine_error = None
_engine_lock = threading.Lock()


def engine():
    global _engine, _engine_error
    with _engine_lock:
        if _engine is None:
            try:
                from scrawl import Scrawl
                engine_ = Scrawl()
                engine_.htr._model('line')  # load TensorFlow weights now rather than on the first page
                engine_.trocr()  # and TrOCR, if installed (slowest to load)
                engine_.context()  # and the clinical language model, if installed
                _engine = engine_  # only now does /api/status report ready
            except Exception as exc:  # surfaced by /api/status
                _engine_error = str(exc)
                raise
        return _engine


@app.get('/')
def index():
    return send_from_directory(WEB_DIR, 'index.html')


@app.get('/samples/<path:name>')
def sample(name: str):
    return send_from_directory(ROOT / 'samples', name)


@app.get('/api/status')
def status():
    ready = _engine is not None
    return jsonify({'ready': ready, 'error': _engine_error,
                    'vocabulary_words': len(_engine.htr.lexicon) if ready else None,
                    'trocr': ready and _engine._trocr is not None,
                    'vision_model': _engine.vlm() is not None if ready else False,
                    'azure': {'configured': _engine.azure.configured(), 'description': _engine.azure.describe()}
                    if ready else None,
                    'mindee': {'configured': _engine.mindee.configured(), 'description': _engine.mindee.describe()}
                    if ready else None,
                    'mineru': {'available': _engine.mineru.available(), 'description': _engine.mineru.describe()}
                    if ready else None,
                    'hardware': _engine.hardware() if ready else None})


@app.post('/api/ocr')
def ocr():
    from scrawl import Options
    upload = request.files.get('file')
    if upload is None:
        return jsonify({'error': 'no file uploaded'}), 400
    form = request.form
    options = Options(engine=form.get('engine', 'auto'), mode=form.get('mode', 'auto'),
                      tta=form.get('tta', 'true') == 'true',
                      lexicon_strength=float(form.get('lexicon_strength', 3.0)),
                      printed_threshold=float(form.get('printed_threshold', 0.9)),
                      handwriting_model=form.get('handwriting_model', 'trocr'),
                      context=form.get('context', 'true') == 'true',
                      vlm=form.get('vlm', 'true') == 'true',
                      include_image=True)
    try:
        return jsonify(engine().read_bytes(upload.read(), upload.filename or '', options))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


@app.post('/api/restructure')
def restructure():
    """Recompute text / fields / tables after the user corrected some lines."""
    from scrawl import structure
    lines = request.get_json()['lines']
    return jsonify(structure.build(lines))


@app.post('/api/export/<fmt>')
def export(fmt: str):
    from scrawl.export import FORMATS
    if fmt not in FORMATS:
        return jsonify({'error': f'unknown format {fmt}'}), 400
    mime = {'json': 'application/json', 'csv': 'text/csv', 'md': 'text/markdown', 'html': 'text/html'}.get(fmt, 'text/plain')
    return Response(FORMATS[fmt](request.get_json()), mimetype=f'{mime}; charset=utf-8')


@app.get('/api/vocabulary')
def get_vocabulary():
    return Response(CUSTOM_VOCAB.read_text(encoding='utf-8') if CUSTOM_VOCAB.exists() else '', mimetype='text/plain')


@app.put('/api/vocabulary')
def put_vocabulary():
    CUSTOM_VOCAB.write_text(request.get_data(as_text=True), encoding='utf-8')
    return jsonify({'vocabulary_words': engine().reload_vocabulary()})


def main() -> None:
    parser = argparse.ArgumentParser(description='Scrawl web UI')
    parser.add_argument('--host', default='127.0.0.1', help='keep 127.0.0.1 unless you mean to share it on your network')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    threading.Thread(target=engine, daemon=True).start()  # warm up models in the background
    print(f'Scrawl running at http://{args.host}:{args.port}  (models load in the background, ~15s)')
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
