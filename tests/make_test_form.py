#!/usr/bin/env python3
"""Build samples/form_mixed.png: a printed form with handwritten answers (cut from sample_2.png), a printed
table, a slight tilt and scanner noise. Used to test auto mode and structure extraction."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

# (x0, y0, x1, y1) of handwritten words in samples/sample_2.png
HANDWRITING = {
    'Catherine': (1570, 790, 1960, 915),
    'Philip': (640, 1340, 875, 1470),
    'absolutely beastly': (15, 1135, 885, 1285),
}


def handwriting(name: str, height: int) -> Image.Image:
    page = cv2.imread(str(ROOT / 'samples' / 'sample_2.png'), cv2.IMREAD_GRAYSCALE)
    x0, y0, x1, y1 = HANDWRITING[name]
    crop = page[y0:y1, x0:x1]
    f = height / crop.shape[0]
    return Image.fromarray(cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA))


def main() -> None:
    img = Image.new('L', (1700, 1250), 255)
    draw = ImageDraw.Draw(img)
    font, bold, title = ImageFont.truetype(FONT, 30), ImageFont.truetype(FONT_BOLD, 30), ImageFont.truetype(FONT_BOLD, 44)

    draw.text((80, 60), 'PATIENT INTAKE FORM', font=title, fill=0)
    draw.text((80, 150), 'Clinic: Riverside Family Practice', font=font, fill=0)
    draw.text((1100, 150), 'Date: 04/11/2026', font=font, fill=0)

    for y, label, word in [(240, 'Patient name:', 'Catherine'), (340, 'Referred by:', 'Philip'),
                           (440, 'Notes:', 'absolutely beastly')]:
        draw.text((80, y), label, font=font, fill=0)
        hw = handwriting(word, 80)
        x = 420
        img.paste(Image.fromarray(np.minimum(np.asarray(img.crop((x, y - 30, x + hw.width, y - 30 + hw.height))),
                                             np.asarray(hw))), (x, y - 30))
        draw.line((x - 10, y + 45, x + 600, y + 45), fill=0, width=2)  # answer line

    draw.text((80, 560), 'Blood pressure: 120/80 mmHg', font=font, fill=0)
    draw.text((1100, 560), 'Heart rate: 72 bpm', font=font, fill=0)
    draw.text((80, 630), 'Phone: +1 (555) 013-2244', font=font, fill=0)
    draw.text((1100, 630), 'Email: front.desk@example.com', font=font, fill=0)

    rows = [('Medication', 'Dose', 'Frequency'), ('Metformin', '500 mg', 'Twice daily'),
            ('Amlodipine', '5 mg', 'Once daily'), ('Salbutamol', '100 mcg', 'As needed')]
    for i, (a, b, c) in enumerate(rows):
        y = 760 + i * 80
        f = bold if i == 0 else font
        draw.text((100, y), a, font=f, fill=0)
        draw.text((700, y), b, font=f, fill=0)
        draw.text((1100, y), c, font=f, fill=0)
        draw.line((80, y + 55, 1620, y + 55), fill=0, width=2)

    page = np.asarray(img).astype(np.float32)
    h, w = page.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -1.5, 1.0)
    page = cv2.warpAffine(page, m, (w, h), borderValue=255)
    rng = np.random.default_rng(0)
    page = page * np.linspace(0.85, 1.0, w)[None, :] + rng.normal(0, 6, page.shape)  # uneven light + noise
    cv2.imwrite(str(ROOT / 'samples' / 'form_mixed.png'), np.clip(page, 0, 255).astype(np.uint8))
    print('wrote samples/form_mixed.png')


if __name__ == '__main__':
    main()
