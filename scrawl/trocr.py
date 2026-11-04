"""Stronger handwriting reader: Microsoft TrOCR (base, handwritten), run locally on the CPU with PyTorch.

Much better than SimpleHTR on real-world handwriting (cursive, capitals, digits), but ~3 s per line on a
4-core CPU. It is optional: without torch/transformers or the downloaded model, Scrawl uses SimpleHTR only.

Vocabulary steering works like it does for SimpleHTR: each word TrOCR is unsure of is compared with nearby
dictionary and clinical words, and every candidate line is scored by TrOCR itself (how likely it finds that
exact text given the image). A known word wins only when the handwriting supports it.
"""
import os
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import hardware
from .htr import tidy_punctuation
from .lexicon import Lexicon
from .paths import MODELS_DIR

MODEL_NAME = 'microsoft/trocr-base-handwritten'
CACHE_DIR = MODELS_DIR / 'hf'
MIN_CONFIDENCE = 0.5  # below this TrOCR is usually describing a stamp, box or scribble, not reading text
SURE_WORD = 0.5       # dictionary words read at least this confidently are not second-guessed (speed)


def available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return (CACHE_DIR / ('models--' + MODEL_NAME.replace('/', '--'))).exists()


def _tight(gray: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(gray < 200)
    if len(ys) < 5:
        return None
    m = max(4, int(0.1 * (ys.max() - ys.min())))
    return gray[max(0, ys.min() - m):ys.max() + m + 1, max(0, xs.min() - m):xs.max() + m + 1]


def _is_mark(crop: np.ndarray) -> bool:
    """Ticks, crosses, single boxed digits: not worth 3 s of TrOCR, which tends to invent words for them."""
    import cv2
    h, w = crop.shape
    n = cv2.connectedComponents((crop < 200).astype(np.uint8))[0] - 1
    return w < 1.3 * h and n <= 2


class TrOCRReader:
    def __init__(self, model_name: str = MODEL_NAME, cache_dir: Path = CACHE_DIR) -> None:
        os.environ.setdefault('HF_HUB_OFFLINE', '1')  # use the downloaded copy; never call out at run time
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from transformers.utils import logging
        logging.set_verbosity_error()
        self.torch = torch
        self.device = hardware.torch_device()  # NVIDIA / AMD / Intel / Apple GPU when present
        if self.device.type == 'cpu':
            hardware.configure_torch_cpu()
        self.device_name = hardware.torch_device_name(self.device)
        self.batch_size = 4 if self.device.type == 'cpu' else 16
        self.processor = TrOCRProcessor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=cache_dir,
                                                               local_files_only=True).eval().to(self.device)
        self.tokenizer = self.processor.tokenizer
        self.start_id = self.model.generation_config.decoder_start_token_id
        self._lock = threading.Lock()

    def read(self, grays: List[np.ndarray], lexicon: Optional[Lexicon] = None, lexicon_strength: float = 3.0,
             batch_size: Optional[int] = None) -> List[Optional[dict]]:
        """Text, confidence (mean token probability) and per-word details for each line crop."""
        import cv2
        crops = [_tight(g) for g in grays]
        todo = [i for i, c in enumerate(crops) if c is not None and not _is_mark(c)]
        results: List[Optional[dict]] = [None] * len(grays)
        batch_size = batch_size or self.batch_size
        for start in range(0, len(todo), batch_size):
            idx = todo[start:start + batch_size]
            images = [cv2.cvtColor(crops[i], cv2.COLOR_GRAY2RGB) for i in idx]
            pixels = self.processor(images=images, return_tensors='pt').pixel_values.to(self.device)
            with self._lock, self.torch.no_grad():
                encoded = self.model.encoder(pixel_values=pixels)
                out = self.model.generate(encoder_outputs=encoded, max_new_tokens=48, num_beams=1,
                                          output_scores=True, return_dict_in_generate=True)
                step_logp = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            sequences, step_logp = out.sequences.cpu(), step_logp.float().cpu()
            for row, i in enumerate(idx):
                features = encoded.last_hidden_state[row:row + 1]  # stays on the GPU for vocabulary scoring
                results[i] = self._finish(sequences[row, 1:], step_logp[row], features, lexicon, lexicon_strength)
        return results

    def _finish(self, seq, logp, features, lexicon: Optional[Lexicon], strength: float) -> dict:
        seq = seq[:len(logp)]
        valid = seq != self.tokenizer.pad_token_id  # generated tokens, including the end-of-text token
        confidence = float(logp[valid].mean().exp()) if valid.any() else 0.0
        keep = valid & (seq != self.tokenizer.eos_token_id)
        ids, token_logp = seq[keep], logp[keep]
        raw = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        text, words = raw, [{'text': w, 'confidence': None, 'alternatives': []} for w in raw.split()]
        if lexicon is not None and strength > 0 and raw:
            text, words = lexicon.correct(
                raw, lambda lines: self._score(lines, features), charset=None, strength=strength,
                max_candidates=8, word_probs=self._word_probs(ids, token_logp), sure=SURE_WORD, match_case=True)
        return {'text': tidy_punctuation(text), 'raw_text': tidy_punctuation(raw),
                'confidence': round(confidence, 3), 'words': _attach(words)}

    def _word_probs(self, ids, token_logp) -> List[float]:
        """Probability of each whitespace-separated word = product of its tokens' probabilities."""
        probs: List[float] = []
        for token, lp in zip(self.tokenizer.convert_ids_to_tokens(ids.tolist()), token_logp.tolist()):
            if token.startswith('Ġ') or not probs:  # byte-level BPE marks a leading space with Ġ
                if token == 'Ġ':
                    continue
                probs.append(0.0)
            probs[-1] += lp
        return [float(np.exp(p)) for p in probs]

    def _score(self, lines: List[str], features) -> np.ndarray:
        """log p(line | image) for each candidate line, reusing the image features and their cross-attention."""
        torch = self.torch
        labels = self.tokenizer(lines, padding=True, return_tensors='pt').input_ids[:, 1:].to(self.device)  # no <s>
        n = len(lines)
        with self._lock, torch.no_grad():
            first = self.model.decoder(input_ids=torch.full((1, 1), self.start_id, device=self.device),
                                       encoder_hidden_states=features,
                                       use_cache=True)
            cache = first.past_key_values
            cache.batch_repeat_interleave(n)  # cross-attention keys/values computed once, shared by all candidates
            rest = self.model.decoder(input_ids=labels[:, :-1], encoder_hidden_states=features.expand(n, -1, -1),
                                      past_key_values=cache, use_cache=True)
            logits = torch.cat([first.logits.expand(n, -1, -1), rest.logits], dim=1)
            logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            mask = labels != self.tokenizer.pad_token_id
            return (logp * mask).sum(dim=1).float().cpu().numpy()


def _attach(words: List[dict]) -> List[dict]:
    """TrOCR writes "stable ." - glue stand-alone punctuation to the previous word, as the line text does."""
    out: List[dict] = []
    for w in words:
        if out and w['text'] and all(not c.isalnum() for c in w['text']) and w['text'][0] in ',.;:!?)':
            out[-1] = {**out[-1], 'text': out[-1]['text'] + w['text']}
        else:
            out.append(w)
    return out
