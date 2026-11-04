"""From detected text boxes to clean line crops: rule removal, merging, missed-ink rescue, neighbour clean-up."""
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np


@dataclass
class Region:
    x0: int
    y0: int
    x1: int
    y1: int
    printed_text: str = ''
    printed_score: float = 0.0
    parts: List['Region'] = field(default_factory=list)
    rescued: bool = False  # found from leftover ink, not by the detector

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def bbox(self) -> List[int]:
        return [int(self.x0), int(self.y0), int(self.x1), int(self.y1)]


def quad_to_region(quad: np.ndarray, width: int, height: int) -> Region:
    x0, y0 = np.floor(quad.min(axis=0)).astype(int)
    x1, y1 = np.ceil(quad.max(axis=0)).astype(int)
    return Region(max(0, x0), max(0, y0), min(width, x1), min(height, y1))


def remove_rules(norm_gray: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """White out long straight lines (ruled paper, form underlines, table borders) that confuse both models."""
    h, w = ink.shape
    mask = ink.astype(np.uint8)
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, max(40, w // 15)), np.uint8))
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((max(40, h // 10), 1), np.uint8))
    horizontal = cv2.dilate(horizontal, np.ones((3, 3), np.uint8)) > 0
    vertical = cv2.dilate(vertical, np.ones((3, 3), np.uint8)) > 0
    strokes = ink & ~horizontal & ~vertical
    erase = (horizontal & ~_crossings(strokes, horizontal, axis=0)) | (vertical & ~_crossings(strokes, vertical, axis=1))
    cleaned = norm_gray.copy()
    cleaned[erase] = 255
    return cleaned


def _crossings(strokes: np.ndarray, rules: np.ndarray, axis: int, reach: int = 7) -> np.ndarray:
    """Rule pixels with pen strokes on both sides, i.e. where a letter (a 'y' descender) crosses the line."""
    before, after = np.zeros_like(strokes), np.zeros_like(strokes)
    for d in range(1, reach + 1):
        before |= np.roll(strokes, d, axis=axis)
        after |= np.roll(strokes, -d, axis=axis)
    return rules & before & after


def same_line(a: Region, b: Region) -> bool:
    if b.y0 <= a.cy <= b.y1 and a.y0 <= b.cy <= a.y1:
        return True
    small, big = sorted((a, b), key=lambda r: r.h)
    is_mark = small.h < 0.7 * big.h and small.x1 - small.x0 <= 2 * small.h
    return is_mark and min(small.y1, big.y1) - max(small.y0, big.y0) >= 0.3 * small.h


def merge_lines(regions: List[Region], gap_factor: float) -> List[Region]:
    """Join boxes on the same text line whose horizontal gap is below gap_factor x line height."""
    while True:  # repeat: a merged line can now reach pieces its parts could not
        merged = _merge_once(regions, gap_factor)
        if len(merged) == len(regions):
            return merged
        regions = merged


def _merge_once(regions: List[Region], gap_factor: float) -> List[Region]:
    parent = list(range(len(regions)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(regions):
        for j in range(i + 1, len(regions)):
            b = regions[j]
            gap = max(a.x0, b.x0) - min(a.x1, b.x1)
            if same_line(a, b) and gap <= gap_factor * min(a.h, b.h):
                parent[find(i)] = find(j)

    groups = {}
    for i, r in enumerate(regions):
        groups.setdefault(find(i), []).append(r)
    merged = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(members[0])
            continue
        members.sort(key=lambda r: r.x0)
        widths = np.array([r.x1 - r.x0 for r in members], float)
        merged.append(Region(
            min(r.x0 for r in members), min(r.y0 for r in members),
            max(r.x1 for r in members), max(r.y1 for r in members),
            printed_text=' '.join(r.printed_text for r in members if r.printed_text),
            printed_score=float(np.average([r.printed_score for r in members], weights=widths)),
            parts=[p for r in members for p in (r.parts or [r])], rescued=all(r.rescued for r in members)))
    return merged


def rescue_missed_ink(ink: np.ndarray, regions: List[Region], line_height: float) -> List[Region]:
    """Word-sized blobs of ink that no detected box covers (detectors often skip very messy writing)."""
    mask = ink.astype(np.uint8)
    for r in regions:
        mask[r.y0:r.y1, r.x0:r.x1] = 0
    kw, kh = max(3, int(0.6 * line_height)), max(1, int(0.15 * line_height))
    blobs = cv2.dilate(mask, np.ones((kh, kw), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(blobs)
    found = []
    for x, y, w, h, _ in stats[1:]:
        x0, y0, x1, y1 = x + kw // 2, y + kh // 2, x + w - kw // 2, y + h - kh // 2
        if not (0.4 * line_height <= y1 - y0 <= 3 * line_height and x1 - x0 >= 0.5 * line_height):
            continue
        if mask[y0:y1, x0:x1].mean() < 0.02:
            continue
        found.append(Region(int(x0), int(y0), int(x1), int(y1), rescued=True))
    return found


def line_crop(norm_gray: np.ndarray, ink: np.ndarray, region: Region, pad_frac: float = 0.25) -> np.ndarray:
    """Crop a line with some margin. Letters cut by the box edge are kept whole; ink belonging to the
    lines above/below or to neighbouring text on the left/right is whitened."""
    h, w = norm_gray.shape
    pad_y, pad_x = int(pad_frac * region.h), region.h
    y0, y1 = max(0, region.y0 - pad_y), min(h, region.y1 + pad_y)
    x0, x1 = max(0, region.x0 - pad_x), min(w, region.x1 + pad_x)
    crop = norm_gray[y0:y1, x0:x1].copy()

    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink[y0:y1, x0:x1].astype(np.uint8), connectivity=8)
    if n > 1:
        top, height = stats[:, cv2.CC_STAT_TOP], stats[:, cv2.CC_STAT_HEIGHT]
        left, width = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_WIDTH]
        core_top, core_bottom = region.y0 - y0, region.y1 - y0
        slack = 0.15 * region.h
        core_left, core_right = region.x0 - x0 - slack, region.x1 - x0 + slack
        v_overlap = np.clip(np.minimum(top + height, core_bottom) - np.maximum(top, core_top), 0, None)
        h_overlap = np.minimum(left + width, core_right) - np.maximum(left, core_left)
        foreign = (v_overlap < 0.5 * height) | (h_overlap <= 0)
        foreign[0] = False
        own = (labels > 0) & ~foreign[labels]
        halo = cv2.dilate(foreign[labels].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # anti-aliased edges too
        crop[halo & ~own] = 255
        # nothing but white outside the core box, apart from letters that straddle its edge
        straddling = own.any(axis=0)
        outside = np.ones(crop.shape[1], bool)
        outside[max(0, int(core_left)):int(core_right)] = False
        crop[:, outside & ~straddling] = 255
    return crop
