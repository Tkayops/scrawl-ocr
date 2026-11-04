"""Scrawl: local OCR for messy handwriting (SimpleHTR) and printed / structured documents (RapidOCR)."""
from . import hardware  # noqa: F401  (first: thread counts must be set before the ML libraries load)
from .pipeline import Options, Scrawl

__all__ = ['Options', 'Scrawl']
