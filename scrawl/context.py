"""Context model: a clinical language model (Bio_ClinicalBERT) that judges which word fits the sentence.

Used to reconstruct handwriting the readers could not make out: candidates come from the dictionary and from
the model's own predictions, and each is scored by context (this model) plus how well it matches the image
(TrOCR). Runs locally; optional - without the downloaded model the step is skipped.
"""
import os
import threading
from pathlib import Path
from typing import List

import numpy as np

from . import hardware
from .paths import MODELS_DIR

MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
CACHE_DIR = MODELS_DIR / 'hf'
CONTEXT_TOKENS = 40  # word pieces of context kept on each side


def available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return (CACHE_DIR / ('models--' + MODEL_NAME.replace('/', '--'))).exists()


class ContextModel:
    def __init__(self, model_name: str = MODEL_NAME, cache_dir: Path = CACHE_DIR) -> None:
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        from transformers.utils import logging
        logging.set_verbosity_error()
        self.torch = torch
        self.device = hardware.torch_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir,
                                                          local_files_only=True).eval().to(self.device)
        self._lock = threading.Lock()

    def _ids(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids if text.strip() else []

    def score(self, left: str, right: str, words: List[str]) -> np.ndarray:
        """How well each word fits between left and right context: pseudo-log-likelihood (each word piece
        predicted with the others visible), so long medical words are not penalised for being split up."""
        tok, torch = self.tokenizer, self.torch
        left_ids, right_ids = self._ids(left)[-CONTEXT_TOKENS:], self._ids(right)[:CONTEXT_TOKENS]
        seqs, owner, target, position = [], [], [], []
        for w, word in enumerate(words):
            pieces = self._ids(word) or [tok.unk_token_id]
            for j in range(len(pieces)):
                seqs.append([tok.cls_token_id] + left_ids + pieces[:j] + [tok.mask_token_id] + pieces[j + 1:]
                            + right_ids + [tok.sep_token_id])
                owner.append(w)
                target.append(pieces[j])
                position.append(1 + len(left_ids) + j)
        width = max(map(len, seqs))
        ids = torch.full((len(seqs), width), tok.pad_token_id)
        mask = torch.zeros_like(ids)
        for k, s in enumerate(seqs):
            ids[k, :len(s)] = torch.tensor(s)
            mask[k, :len(s)] = 1
        rows = torch.arange(len(seqs))
        with self._lock, torch.no_grad():
            logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits
            logp = torch.log_softmax(logits[rows, torch.tensor(position)].float(), dim=-1)
            picked = logp[rows, torch.tensor(target)].cpu().numpy()
        out = np.zeros(len(words))
        np.add.at(out, owner, picked)
        return out

    def propose(self, left: str, right: str, k: int = 30) -> List[str]:
        """Whole words the model expects in the gap (single word pieces only)."""
        tok, torch = self.tokenizer, self.torch
        ids = [tok.cls_token_id] + self._ids(left)[-CONTEXT_TOKENS:] + [tok.mask_token_id] + \
            self._ids(right)[:CONTEXT_TOKENS] + [tok.sep_token_id]
        at = ids.index(tok.mask_token_id)
        with self._lock, torch.no_grad():
            logits = self.model(input_ids=torch.tensor([ids], device=self.device)).logits[0, at]
        words = tok.convert_ids_to_tokens(logits.topk(k).indices.tolist())
        return [w for w in words if w.isalpha() and len(w) >= 3]
