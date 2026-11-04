# Scrawl: local OCR for messy handwriting and structured documents

Reads handwritten notes, typed letters and forms. It returns plain text and structured data: lines with
confidence, `Label: value` fields, tables, and typed values such as dates, doses, blood pressure, phone
numbers and emails. **Everything runs on this computer.** No GPU is needed, nothing is uploaded, and the
web UI loads nothing from the internet.

| Part | Engine |
|---|---|
| Handwriting, **Accurate** (default) | Microsoft [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten) base-handwritten (PyTorch, CPU), cross-checked with SimpleHTR |
| Handwriting, **Fast** | [SimpleHTR](https://github.com/githubharald/SimpleHTR) line model (TensorFlow, IAM-trained), plus the accuracy boosters below |
| Reconstructing unclear words (Accurate mode) | [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT), a clinical language model, together with TrOCR |
| Printed text and text detection | RapidOCR (PaddleOCR PP-OCRv4 via ONNX Runtime, models bundled in the pip wheel) |
| PDFs | pypdfium2 (each page rendered at 200 dpi) |

## Quick start

```bash
./setup.sh
```

```bash
.venv/bin/python server.py
```

Then open http://127.0.0.1:8765. Drop in a scan, photo or PDF, or click one of the samples.

* **Accurate / Fast:** Accurate (TrOCR) is the default and roughly halves the errors on real-world
  handwriting. It takes about 1 minute per handwritten page on a 4-core CPU. Fast (SimpleHTR) takes about
  10–20 s.

* **Boxes on the page:** purple is handwriting, teal is printed text. Hover or click one to find its line.
* **Lines tab:** dotted words were hard to read. Click one to pick an alternative, or use ✎ to correct
  the line. Fields and tables update as you edit.
* **Text tab:** shows the page laid out like the document. Titles and section headers become headings,
  form fields become `Label: value` (side by side as on the paper), and tables stay in place. Free-text
  lines are joined into paragraphs, and a handwritten answer that runs onto a second line stays with its
  field. Handwritten parts are coloured and low-confidence parts are underlined. Click any part to find
  it on the page. "Plain text" shows the same as text.
* **Downloads:**
  * Formatted document (HTML, printable)
  * Formatted text (TXT)
  * Markdown
  * CSV (one row per line)
  * The full JSON, which includes the formatted blocks as `document`
* **Vocabulary tab:** a clinical vocabulary is built in. Add your own names, facilities and jargon here
  (see the tips below).

### Hardware: GPU and CPU threads

`./setup.sh` detects the machine and installs the matching builds:

| Machine | TrOCR (handwriting) | Text detection and printed text | SimpleHTR |
|---|---|---|---|
| NVIDIA GPU (Linux, or Windows via WSL2) | GPU (CUDA) | GPU (onnxruntime-gpu) | CPU |
| AMD GPU with ROCm (Linux) | GPU (ROCm) | CPU | CPU |
| Apple Silicon Mac | GPU (Metal) | CPU | CPU |
| Intel GPU (Arc and similar), on request: `GPU=intel ./setup.sh` | GPU (XPU) | CPU | CPU |
| No GPU | CPU | CPU | CPU |

* **Detection:** at start-up the best device is picked automatically. The web UI's status line shows
  where handwriting runs; hover over it for every engine.
* **CPU threads:** every engine uses all CPU threads (PyTorch, ONNX Runtime, TensorFlow, OpenCV and
  numpy).
* **Why SimpleHTR stays on the CPU:** TensorFlow's GPU build needs different CUDA libraries from
  PyTorch's. SimpleHTR is the smallest model, so this costs little.
* **Overrides:**
  * `GPU=nvidia|amd|intel|apple|cpu ./setup.sh` forces a build, and `TORCH_INDEX=<url>` picks another
    PyTorch build (e.g. a different CUDA version).
  * `SCRAWL_THREADS=8` uses fewer CPU threads. On machines with hyper-threading, the real core count can
    be as fast.
  * `SCRAWL_DEVICE=cpu` (or `cuda:1`, `mps`, ...) forces the PyTorch device.

On the 2-core / 4-thread laptop this was built on, using all 4 threads made text detection 2.3× faster
(4.0 s to 1.7 s per page). TrOCR stayed about the same there, because its 2 real cores were already busy.

### Command line

```bash
.venv/bin/python ocr.py note.jpg                          # text to stdout
.venv/bin/python ocr.py form.pdf --format json -o form.json
.venv/bin/python ocr.py form.pdf --format html -o form.html  # formatted, printable
.venv/bin/python ocr.py scans/*.png --format md -o out/    # one file per input
.venv/bin/python ocr.py note.jpg --mode handwriting --overlay boxes.png
```

The main options:

* `--mode auto|handwriting|printed`
* `--lexicon-strength 0..6` (default 3)
* `--printed-threshold` (default 0.9)
* `--fast`: SimpleHTR instead of TrOCR
* `--no-context`: skip reconstructing unclear words from the surrounding text
* `--no-tta`: with `--fast`, faster still
* `--no-rescue`

Run `--help` for the full list.

### Python

```python
from scrawl import Scrawl, Options
engine = Scrawl()                       # load once (about 15 s), reuse for many pages
result = engine.read_file('form.png', Options(mode='auto'))
page = result['pages'][0]
page['text'], page['fields'], page['tables'], page['entities'], page['lines']
```

## How it works

1. **Clean the page:** divide out shadows and paper colour, then erase ruled lines, form underlines and
   table borders. Pen strokes that cross those lines are kept.
   * **Pen layer:** when a page has coloured ink (forms printed in black, filled in with a blue pen), it
     is split into a print layer and a pen layer, and each is read separately. The answers no longer get
     mixed up with labels, dotted lines and boxes. Stamps are coloured too; they are skipped because
     their text is tilted.
2. **Find text:** use RapidOCR's detector with lowered thresholds, so faint scrawl is still found. The page
   is straightened using the angle of the detected lines.
3. **Printed or handwritten?** In auto mode each box is first read by the printed-text model:
   * Confident readings are kept.
   * The rest go to SimpleHTR, along with uncertain "printed" boxes that continue a handwritten line.
   * A printed `Label:` fused with a handwritten answer in one box is split at the gap.
4. **Handwriting:**
   * Boxes on the same line are merged, and ink the detector missed is rescued.
   * Each line is cropped, with neighbouring lines' ink removed and letters cut by the box edge kept whole.
   * The crop is rescaled to the resolution SimpleHTR was trained on, and very wide word gaps are
     narrowed.
   * **Accurate mode:** TrOCR reads every handwritten line. When it is unsure (it tends to invent words
     for stamps and scribbles), the reading falls back to SimpleHTR or the printed model. When TrOCR and
     SimpleHTR agree, the line is marked as such. Readings from the other models stay available as
     alternatives in the Lines tab.
   * **Vocabulary steering (both readers):**
     * Each word the reader is unsure of, or that isn't a known word, is compared with nearby words: the
       English dictionary, the built-in clinical vocabulary (`vocab/medical_terms.txt`, about 800
       diagnoses, drugs, abbreviations and form terms) and your `vocab/custom_words.txt`.
     * Badly mangled long words are also matched against clinical words that merely look alike, so
       "opportunationology" reaches "ophthalmology".
     * The reader then scores each candidate against the image: TrOCR by how likely it finds that exact
       line, SimpleHTR through its character probabilities. A known word wins only when the handwriting
       supports it. Clinical and custom words get a small extra preference.
     * Short capitalised abbreviations (LE, BP, TSC) are left alone. In Accurate mode, names from the
       general dictionary are never offered, so a patient's name isn't turned into a place name.
   * Two boosters run on top of SimpleHTR, which is otherwise unchanged:
     * **Extra passes (test-time augmentation):** the line is read at three stroke thicknesses and the
       per-character probabilities are averaged.
     * **Dictionary correction:** each word is compared with nearby dictionary words (~100k English words
       plus your vocabulary). Each candidate is scored by how well the network's output supports it. A
       dictionary word wins only when the pixels allow it, so names and numbers that are read clearly are
       left alone.
5. **Reconstructing unclear words from context** (Accurate mode):
   * **Which words:** after the page is read, handwritten words that aren't real words, or that TrOCR was
     unsure of, get a second look.
   * **Candidates:** similar dictionary and clinical words, plus words the clinical language model
     (Bio_ClinicalBERT) expects there. Its context is the field label and the lines around the word.
   * **Scoring:** each candidate is scored on three things: how well it matches the handwriting (TrOCR),
     whether it's a known word, and how well it fits the sentence (Bio_ClinicalBERT).
   * **Safety:**
     * Names (any field labelled name, doctor, signature…), numbers, dates and short abbreviations are
       never guessed.
     * A medicine name that only context favours is shown as a suggestion and not filled in.
   * **In the Lines tab:** reconstructed words are shown in *wavy italic* with the original reading, and
     suggestions are dashed.
6. **Structure:**
   * Rows and reading order.
   * Fields: `Label: value` inline, in the next cell or on the line below, and a printed caption
     followed by handwriting even without a colon.
   * Tables: aligned columns, but not label/answer layouts or question rows.
   * Typed values.
7. **Formatting** (`scrawl/document.py`): headings (larger or all-capital printed lines), field rows,
   paragraphs and tables in reading order. These are rendered as text, Markdown, HTML and in the web UI.

### Measured accuracy

Character error rate (share of characters wrong).

**A real scanned medical claim form and referral letter.** These have blue-pen handwriting, including a
doctor's cursive, over a black printed form. The test is 20 handwritten fields, 232 characters, with ground
truth typed by hand. They are not bundled because they contain patient data.

| Setup | Errors | Time per form page |
|---|---|---|
| SimpleHTR, first version | 51% | ~10–20 s |
| Fast: pen layer and better engine choice | 38% | 12–24 s |
| Accurate: plus TrOCR | 22% | 64–82 s |
| Accurate: plus clinical vocabulary steering TrOCR | 17.2% | 79–96 s |
| **Accurate: plus context reconstruction (Bio_ClinicalBERT)** | **15.9%** | about the same; page 3 measured 92 s with it, 115 s without (run-to-run noise) |

The context step's one change on this PDF was "pentment" → "patient". It made no change on the sample
pages (2.3% either way).

**Bundled handwritten sample pages** (`tests/eval_samples.py`). These are clean, IAM-style writing:

| Setup | sample_1 | sample_2 | overall |
|---|---|---|---|
| SimpleHTR without boosters | 8.6% | 3.6% | 4.8% |
| Fast: SimpleHTR with extra passes and dictionary correction | 6.5% | 2.9% | 3.8% |
| **Accurate: TrOCR + SimpleHTR** | 2.9% | 2.2% | **2.3%** |

`samples/form_mixed.png` is a tilted, unevenly lit printed form with handwritten answers. It comes out
fully correct: 9 fields, 1 table and 8 typed values.

## Getting the most out of terrible handwriting

* **Capture:** scan at 300 dpi, or photograph flat in even light, filling the frame. Dark ink beats
  pencil.
* **Add your vocabulary:** put the words you expect in `vocab/custom_words.txt` (or the Vocabulary tab).
  Clinical terms and common drugs are built in, so focus on what's specific to you: doctors' and common
  patient names, facility names, local brand names, in-house abbreviations.
* **Pen colour:** forms filled in with a blue or black pen on a black-printed form read best in blue,
  because the pen layer can then separate them.
* **Force the mode:** use **Handwriting** mode for pages that are all handwriting, so a confident but
  wrong printed reading can't win.
* **Tune the dictionary:** raise **Dictionary** (4–6) for prose full of ordinary words. Lower it (1–2)
  for lists of names and codes.
* **Check the Lines tab:** the confidence bars show which lines need a human look.

## Limits

* **English handwriting models.** Both TrOCR and SimpleHTR were trained on English handwriting (IAM).
  Doctors' cursive is still hard: about 1 in 5 characters is wrong on the test form, so check the Lines
  tab.
* **Characters in boxes.** Digits written one per box (dates, IDs) are often only partly read.
* **Limited characters in Fast mode.** SimpleHTR can only produce
  `A–Z a–z 0–9` and `space ! " # & ' ( ) * + , - . / : ; ?`. Printed text has the full character set.
* **Commas and quotes get dropped.** Small punctuation in handwriting is often missed.
* **Multi-column layouts read across.** Reading order runs row by row, so magazine-style columns are
  interleaved.
* **Speed:**
  * Accurate: about 1 minute per handwritten page (about 3 s per handwritten line). Printed pages stay
    fast.
  * Fast: about 10–20 s per page.
  * Memory: about 3 GB with both models loaded.

## Next step: train on your own handwriting

SimpleHTR can be fine-tuned. Collect line images with their transcriptions in IAM format and see
`vendor/SimpleHTR/README.md`, "Train model on IAM dataset". Copy the resulting checkpoint into
`models/line/`. A few hundred lines from the writers you care about usually help more than any setting.

## Credits and licences

* [SimpleHTR](https://github.com/githubharald/SimpleHTR) by Harald Scheidl (MIT). Its code and
  pretrained line model are downloaded by `setup.sh`, pinned to the tested commit.
* `samples/sample_1.png` and `samples/sample_2.png` come from Harald Scheidl's
  [HTRPipeline](https://github.com/githubharald/HTRPipeline) (MIT). `samples/form_mixed.png` is generated
  by `tests/make_test_form.py` from them.
* [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten) by Microsoft (MIT), downloaded by
  `setup.sh`.
* [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) by Alsentzer et al. (MIT),
  trained on MIMIC-III clinical notes, downloaded by `setup.sh`.
* [RapidOCR](https://github.com/RapidAI/RapidOCR) (Apache-2.0) with PaddleOCR PP-OCRv4 models
  (Apache-2.0).
* The handwriting models were trained on the IAM Handwriting Database, which has its own terms for
  commercial use.

## Layout

```
ocr.py               command line
server.py, web/      local web UI (127.0.0.1 only)
scrawl/              pipeline: hardware (GPU / threads), imaging (incl. pen layer), layout, htr (SimpleHTR),
                     trocr, context (clinical language model), ctc, lexicon, printed, structure,
                     document (formatting), export
vocab/               your custom words
vendor/SimpleHTR/    upstream SimpleHTR code (unchanged)
models/line/         SimpleHTR pretrained line model
models/hf/           TrOCR and Bio_ClinicalBERT (downloaded once by setup.sh, then used offline)
samples/, tests/     sample pages, test-form generator and accuracy check
```
