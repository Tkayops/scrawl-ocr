"""CTC helpers that operate on the raw network output with numpy (no TensorFlow needed)."""
from typing import List, Sequence

import numpy as np


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def best_path(log_probs: np.ndarray, blank: int) -> List[int]:
    """Greedy decoding: most likely class per frame, collapse repeats, drop blanks."""
    labels, prev = [], -1
    for idx in log_probs.argmax(axis=1):
        if idx != prev and idx != blank:
            labels.append(int(idx))
        prev = idx
    return labels


def log_likelihood(log_probs: np.ndarray, label_seqs: Sequence[Sequence[int]], blank: int) -> np.ndarray:
    """log p(labels | image) for several label sequences at once (CTC forward algorithm).

    log_probs: T x C matrix of per-frame log probabilities.
    Returns one log-likelihood per sequence (-inf if the sequence cannot fit in T frames).
    """
    num_frames = log_probs.shape[0]
    num_seqs = len(label_seqs)
    max_len = max((len(s) for s in label_seqs), default=0)
    num_states = 2 * max_len + 1

    # extended label sequence: blank, l1, blank, l2, ..., blank
    ext = np.full((num_seqs, num_states), blank, dtype=np.int64)
    for k, seq in enumerate(label_seqs):
        if len(seq):
            ext[k, 1:2 * len(seq):2] = seq
    # a state may be reached by skipping the preceding blank if it differs from the label two states back
    can_skip = np.zeros((num_seqs, num_states), dtype=bool)
    can_skip[:, 2:] = (ext[:, 2:] != blank) & (ext[:, 2:] != ext[:, :-2])

    emit = log_probs[:, ext]  # T x K x S
    alpha = np.full((num_seqs, num_states), -np.inf)
    alpha[:, 0] = emit[0, :, 0]
    if num_states > 1:
        alpha[:, 1] = emit[0, :, 1]

    neg1 = np.full((num_seqs, 1), -np.inf)
    neg2 = np.full((num_seqs, 2), -np.inf)
    with np.errstate(invalid='ignore', divide='ignore'):
        for t in range(1, num_frames):
            from_prev = np.concatenate([neg1, alpha[:, :-1]], axis=1)
            from_skip = np.where(can_skip, np.concatenate([neg2, alpha[:, :-2]], axis=1), -np.inf)
            alpha = np.logaddexp(np.logaddexp(alpha, from_prev), from_skip) + emit[t]

        result = np.empty(num_seqs)
        for k, seq in enumerate(label_seqs):
            last = 2 * len(seq)
            result[k] = alpha[k, last] if last == 0 else np.logaddexp(alpha[k, last], alpha[k, last - 1])
    return result
