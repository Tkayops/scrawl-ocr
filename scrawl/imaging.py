"""Loading pages and cleaning up scans / phone photos."""
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

PDF_DPI = 200


def load_pages(data: bytes, filename: str = '') -> List[np.ndarray]:
    """Decode an image or PDF into BGR pages."""
    if data[:5] == b'%PDF-' or filename.lower().endswith('.pdf'):
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(data)
        pages = []
        for page in pdf:
            rgb = np.asarray(page.render(scale=PDF_DPI / 72).to_pil().convert('RGB'))
            pages.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return pages

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'Cannot read {filename or "image"}: unsupported or corrupt file')
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(1, int(img.max())))
    if img.ndim == 2:
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)]
    if img.shape[2] == 4:  # transparent background -> white paper
        alpha = img[:, :, 3:4].astype(np.float32) / 255
        img = (img[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return [img]


def load_file(path: Path) -> List[np.ndarray]:
    return load_pages(Path(path).read_bytes(), str(path))


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide out the paper background (shadows, yellow paper, uneven light) -> white paper, dark ink."""
    h, w = gray.shape
    scale = min(1.0, 800 / max(h, w))  # estimate the background on a small copy, it is smooth anyway
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    k = max(9, int(min(small.shape) / 25) | 1)
    background = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    background = cv2.GaussianBlur(background, (0, 0), k / 3)
    if scale < 1:
        background = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.divide(gray, np.maximum(background, 1), scale=255)


def ink_mask(norm_gray: np.ndarray) -> np.ndarray:
    """Binary mask of ink (True = ink) on an illumination-normalised page, specks removed."""
    thresh, _ = cv2.threshold(norm_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (norm_gray < min(thresh, 200)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(4, int(norm_gray.size * 2e-6))
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def skew_angle(quads: List[np.ndarray]) -> float:
    """Median text-line angle (degrees) from detected quadrilaterals (only clearly horizontal-ish, long boxes)."""
    angles, weights = [], []
    for q in quads:
        top = q[1] - q[0]
        length = float(np.hypot(*top))
        height = float(np.hypot(*(q[3] - q[0])))
        if length > 2 * height:
            angle = np.degrees(np.arctan2(top[1], top[0]))
            if abs(angle) < 20:
                angles.append(angle)
                weights.append(length)
    if len(angles) < 2:
        return 0.0
    order = np.argsort(angles)
    cumulative = np.cumsum(np.array(weights)[order])
    return float(np.array(angles)[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def rotate(img: np.ndarray, angle: float, quads: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Rotate the page (and the quads) so text runs horizontally; the canvas grows to keep everything."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    border = (255, 255, 255) if img.ndim == 3 else 255
    out = cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=border)
    moved = [cv2.transform(q[None], m)[0] for q in quads]
    return out, moved


def pen_weight(bgr: np.ndarray) -> np.ndarray:
    """0..1 per pixel: how clearly it is coloured ink (blue/red/green pen, stamps) rather than black print or paper.

    Forms are usually printed in black and filled in with a coloured pen, so this separates the answers from the
    form even where they overlap."""
    channels = bgr.astype(np.int16)
    chroma = (channels.max(axis=2) - channels.min(axis=2)).astype(np.float32)
    darkest = normalize_illumination(bgr.min(axis=2))  # coloured ink is dark in at least one channel
    dark = np.clip((215 - darkest.astype(np.float32)) / 60, 0, 1)
    return np.clip((chroma - 30) / 35, 0, 1) * dark


def split_layers(bgr: np.ndarray, norm: np.ndarray, weight: np.ndarray):
    """(print layer, pen layer) as greyscale pages: print with the pen ink removed, pen ink alone on white."""
    darkest = normalize_illumination(bgr.min(axis=2)).astype(np.float32)
    pen = 255 - (255 - darkest) * weight
    printed = norm.copy()
    printed[cv2.dilate((weight > 0.1).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 255  # no ghosts of the pen
    return printed, pen.astype(np.uint8)
