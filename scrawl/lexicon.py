"""Dictionary-guided correction of recognised handwriting.

Every word the network reads is compared with nearby dictionary words (by edit distance).
Each candidate is then scored by how well the *network output itself* supports it (CTC
likelihood of the whole line with that word swapped in), plus a bonus for being a known word.
So "hal" can become "that" when the pixels agree, while names and numbers the network is
sure about are left alone.
"""
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
CUSTOM_WORD_COUNT = 1000  # custom vocabulary counts as a very frequent word


def _read_lines(path: Path) -> Iterable[str]:
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                yield line


class Lexicon:
    def __init__(self, corpus_files: Sequence[Path] = (), word_list_files: Sequence[Path] = (),
                 custom_files: Sequence[Path] = ()) -> None:
        self.freq: Counter = Counter()
        self.forms: Dict[str, set] = defaultdict(set)  # lower-case word -> spellings seen ("May", "may")
        self.domain: set = set()  # words from the clinical / custom vocabulary files (lower case)
        for path in corpus_files:  # running text: counts give word frequencies
            with open(path, encoding='utf-8', errors='ignore') as f:
                self._add(WORD_RE.findall(f.read()), 1)
        for path in word_list_files:  # one word per line
            self._add((w for w in _read_lines(path) if WORD_RE.fullmatch(w)), 1)
        self.drugs: set = set()  # medicine names (sections marked [drugs]): never filled in from context alone
        for path in custom_files:  # user vocabulary, may contain phrases
            drugs_section = False
            with open(path, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#'):
                        drugs_section = '[drugs]' in line if line.startswith('# ---') else drugs_section
                        continue
                    words = WORD_RE.findall(line)
                    self._add(words, CUSTOM_WORD_COUNT)
                    self.domain.update(w.lower() for w in words if len(w) >= 3)
                    if drugs_section:
                        self.drugs.update(w.lower() for w in words)
        self.domain_list = sorted(self.domain)
        # names from the general dictionary (Maine, Tacoma): never offered when common_only is set, so a
        # patient's name is not "corrected" into a place; names in the clinical / custom files still are
        self.proper = {low for low, forms in self.forms.items()
                       if low not in self.domain and all(f[0].isupper() for f in forms)}

        self.by_len: Dict[int, List[str]] = defaultdict(list)
        for word in self.freq:
            self.by_len[len(word)].append(word)

    def _add(self, words: Iterable[str], count: int) -> None:
        for word in words:
            low = word.lower()
            self.freq[low] += count
            self.forms[low].add(word)

    def __len__(self) -> int:
        return len(self.freq)

    def __contains__(self, word: str) -> bool:
        return word.lower() in self.freq

    def candidates(self, word: str, limit: int = 15, common_only: bool = False) -> List[str]:
        """Dictionary words within a small edit distance (closest, clinical, frequent first), plus clinical
        vocabulary words that merely look alike: long medical words get mangled beyond a few edits."""
        low = word.lower()
        max_dist = 1 if len(low) <= 3 else 2 if len(low) <= 6 else 3 if len(low) <= 9 else 4
        found = []
        for length in range(max(1, len(low) - max_dist), len(low) + max_dist + 1):
            choices = self.by_len.get(length)
            if choices:
                found += process.extract(low, choices, scorer=Levenshtein.distance,
                                         score_cutoff=max_dist, limit=None)
        if common_only:
            found = [r for r in found if r[0] not in self.proper]
        found.sort(key=lambda r: (r[1], r[0] not in self.domain, -self.freq[r[0]]))
        out = [w for w, _, _ in found[:limit]]
        if len(low) >= 5 and self.domain_list:
            similar = [w for w in self.domain_list if 0.6 <= len(w) / len(low) <= 1.6 and w not in out]
            out += [w for w, _, _ in process.extract(low, similar, scorer=fuzz.ratio, score_cutoff=55, limit=3)]
        return out

    def bonus(self, word: str, strength: float, domain_bonus: float = 1.0) -> float:
        """Preference for known words, added to a reader's log-likelihood (0 for unknown words)."""
        low = word.lower()
        if low not in self.freq:
            return 0.0
        return strength + 0.1 * np.log1p(self.freq[low]) + (domain_bonus if low in self.domain else 0.0)

    def _matching_case(self, low: str, original: str) -> str:
        """The candidate written the way the original was (TrOCR gets letter case right far more than letters)."""
        if len(original) > 1 and original.isupper():
            return low.upper()
        special = [f for f in self.forms[low] if f != low and f != low.capitalize()]  # NHIF, McDonald
        if special:
            return special[0]
        return low.capitalize() if original[0].isupper() else low

    def _spellings(self, low: str, original: str) -> set:
        out = set(self.forms[low])
        out |= {f[0].upper() + f[1:] for f in out}  # sentence-initial capital
        if sum(c.isupper() for c in original) >= 2:
            out.add(low.upper())
        return out

    def correct(self, text: str, score_texts: Callable[[List[str]], np.ndarray], charset: Optional[Dict[str, int]] = None,
                strength: float = 3.0, max_candidates: int = 15, word_probs: Optional[List[float]] = None,
                sure: float = 0.9, match_case: bool = False, domain_bonus: float = 1.0) -> Tuple[str, List[dict]]:
        """Correct each word of `text`; score_texts returns log p(text | image) for candidate lines.

        word_probs (one per whitespace-separated word) lets confidently read dictionary words be skipped;
        match_case tries each candidate only in the original word's letter case (fewer, cheaper candidates)."""
        parts = re.split(r'(\s+)', text)
        if word_probs is not None and len(word_probs) != sum(1 for p in parts if p and not p.isspace()):
            word_probs = None
        words, k = [], -1
        for i, part in enumerate(parts):
            if not part or part.isspace():
                continue
            k += 1
            match = WORD_RE.search(part)
            if not match or len(match.group()) < 2:
                words.append({'text': part, 'confidence': None, 'alternatives': []})
                continue
            core, pre, post = match.group(), part[:match.start()], part[match.end():]
            if word_probs is not None and word_probs[k] >= sure and core in self:
                words.append({'text': part, 'confidence': round(word_probs[k], 3), 'alternatives': []})
                continue
            if core.isupper() and len(core) <= 4:  # LE, BP, OIE, TSC: abbreviations, not misspelt words
                words.append({'text': part, 'confidence': None, 'alternatives': []})
                continue

            options = {core}
            for cand in self.candidates(core, max_candidates, common_only=match_case):
                options |= {self._matching_case(cand, core)} if match_case else self._spellings(cand, core)
            options = [o for o in options if charset is None or all(c in charset for c in o)]

            lines = [''.join(parts[:i] + [pre + o + post] + parts[i + 1:]) for o in options]
            scores = score_texts(lines)
            scores = scores + np.array([self.bonus(o, strength, domain_bonus) for o in options])
            order = np.argsort(-scores)
            probs = np.exp(scores - scores[order[0]])
            probs /= probs.sum()

            best = options[order[0]]
            parts[i] = pre + best + post
            words.append({
                'text': parts[i],
                'confidence': round(float(probs[order[0]]), 3),
                'read_as': part if best != core else None,
                'alternatives': [pre + options[j] + post for j in order[1:4] if probs[j] >= 0.02],
            })
        return ''.join(parts), words


def default_lexicon(simplehtr_dir: Path, custom_files: Sequence[Path] = ()) -> Optional[Lexicon]:
    corpus = [simplehtr_dir / 'data' / 'corpus.txt']
    word_lists = [p for p in (Path('/usr/share/dict/american-english'), Path('/usr/share/dict/british-english'),
                              Path('/usr/share/dict/words')) if p.exists()][:1]
    return Lexicon([p for p in corpus if p.exists()], word_lists, [p for p in custom_files if p.exists()])
