"""
utils/preprocess.py
===================
Preprocessing helpers for the Logo Detection pipeline.
Handles dataset conversion, augmentation, and YOLO format utilities.
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image


# ── Class registry ─────────────────────────────────────────────────────────────
LOGO_CLASSES = [
    'adidas', 'apple', 'bmw', 'cocacola', 'fedex',
    'ferrari', 'ford', 'google', 'gucci', 'hp'
]
CLASS2IDX = {c: i for i, c in enumerate(LOGO_CLASSES)}
IDX2CLASS  = {i: c for c, i in CLASS2IDX.items()}


# ── Annotation converters ──────────────────────────────────────────────────────

def flickr32_to_yolo(
    src_root: str,
    dst_images: str,
    dst_labels: str,
    classes: Optional[List[str]] = None,
) -> int:
    """
    Convert FlickrLogos-32 annotations to YOLO format.

    FlickrLogos-32 structure:
        <root>/<class_name>/<img>.jpg
        <root>/<class_name>/<img>.gt_data.txt  (x1 y1 x2 y2 per line)

    YOLO format:
        <class_idx> <cx_norm> <cy_norm> <w_norm> <h_norm>

    Returns count of successfully converted images.
    """
    classes = classes or LOGO_CLASSES
    c2i     = {c: i for i, c in enumerate(classes)}
    Path(dst_images).mkdir(parents=True, exist_ok=True)
    Path(dst_labels).mkdir(parents=True, exist_ok=True)

    converted = 0
    for cls_name in classes:
        cls_dir = Path(src_root) / cls_name
        if not cls_dir.exists():
            continue
        cls_idx = c2i[cls_name]
        for ann_file in cls_dir.glob('*.gt_data.txt'):
            img_stem = ann_file.stem.replace('.gt_data', '')
            img_path = cls_dir / f'{img_stem}.jpg'
            if not img_path.exists():
                img_path = cls_dir / f'{img_stem}.png'
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]

            dst_img = Path(dst_images) / img_path.name
            shutil.copy(img_path, dst_img)

            lbl_path = Path(dst_labels) / f'{img_stem}.txt'
            with open(ann_file) as af, open(lbl_path, 'w') as lf:
                for line in af:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        x1, y1, x2, y2 = map(int, parts[1:5])
                    except ValueError:
                        continue
                    cx = ((x1 + x2) / 2) / iw
                    cy = ((y1 + y2) / 2) / ih
                    nw = (x2 - x1) / iw
                    nh = (y2 - y1) / ih
                    # Clamp to [0,1]
                    cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
                    nw, nh = max(0.01, min(1.0, nw)), max(0.01, min(1.0, nh))
                    lf.write(f'{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n')
            converted += 1

    return converted


def openlogo_to_yolo(
    src_root: str,
    dst_images: str,
    dst_labels: str,
    classes: Optional[List[str]] = None,
) -> int:
    """
    Convert OpenLogo dataset (Pascal VOC XML) to YOLO format.

    OpenLogo structure:
        <root>/JPEGImages/<img>.jpg
        <root>/Annotations/<img>.xml  (VOC format)
    """
    classes = classes or LOGO_CLASSES
    c2i     = {c.lower(): i for i, c in enumerate(classes)}
    Path(dst_images).mkdir(parents=True, exist_ok=True)
    Path(dst_labels).mkdir(parents=True, exist_ok=True)

    ann_dir = Path(src_root) / 'Annotations'
    img_dir = Path(src_root) / 'JPEGImages'
    converted = 0

    for xml_file in ann_dir.glob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        img_name = root.find('filename').text
        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        size = root.find('size')
        iw   = int(size.find('width').text)
        ih   = int(size.find('height').text)

        lbl_lines = []
        for obj in root.findall('object'):
            name = obj.find('name').text.lower().strip()
            if name not in c2i:
                continue
            bb   = obj.find('bndbox')
            x1   = int(float(bb.find('xmin').text))
            y1   = int(float(bb.find('ymin').text))
            x2   = int(float(bb.find('xmax').text))
            y2   = int(float(bb.find('ymax').text))
            cx   = ((x1 + x2) / 2) / iw
            cy   = ((y1 + y2) / 2) / ih
            nw   = (x2 - x1) / iw
            nh   = (y2 - y1) / ih
            lbl_lines.append(f'{c2i[name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')

        if not lbl_lines:
            continue

        shutil.copy(img_path, Path(dst_images) / img_name)
        lbl_path = Path(dst_labels) / xml_file.with_suffix('.txt').name
        lbl_path.write_text('\n'.join(lbl_lines) + '\n')
        converted += 1

    return converted


# ── Dataset splitting ──────────────────────────────────────────────────────────

def train_val_split(
    images_dir: str,
    labels_dir: str,
    out_root: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[int, int]:
    """
    Split images/labels into train and val directories.

    Returns (n_train, n_val).
    """
    images = sorted(Path(images_dir).glob('*.jpg'))
    images += sorted(Path(images_dir).glob('*.png'))
    random.seed(seed)
    random.shuffle(images)
    split    = int(len(images) * (1 - val_ratio))
    train_im = images[:split]
    val_im   = images[split:]

    for subset, img_list in [('train', train_im), ('val', val_im)]:
        img_dst = Path(out_root) / 'images' / subset
        lbl_dst = Path(out_root) / 'labels' / subset
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for img_p in img_list:
            shutil.copy(img_p, img_dst / img_p.name)
            lbl_p = Path(labels_dir) / img_p.with_suffix('.txt').name
            if lbl_p.exists():
                shutil.copy(lbl_p, lbl_dst / lbl_p.name)

    return len(train_im), len(val_im)


def write_yaml(out_path: str, yolo_root: str,
               classes: List[str], train: str = 'images/train',
               val: str = 'images/val') -> None:
    """Write a YOLO-compatible dataset.yaml file."""
    content = (
        f'path: {yolo_root}\n'
        f'train: {train}\n'
        f'val:   {val}\n\n'
        f'nc: {len(classes)}\n'
        f'names: {classes}\n'
    )
    Path(out_path).write_text(content)


# ── Image preprocessing ────────────────────────────────────────────────────────

def resize_and_pad(image: np.ndarray, target: int = 640) -> np.ndarray:
    """
    Letterbox-resize image to target × target while preserving aspect ratio.
    """
    h, w = image.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    pad_y  = (target - nh) // 2
    pad_x  = (target - nw) // 2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
    return canvas


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize to [0,1] float32."""
    return image.astype(np.float32) / 255.0


def augment_image(image: np.ndarray, flip_prob: float = 0.5,
                  jitter: float = 0.2) -> np.ndarray:
    """
    Basic augmentation: horizontal flip + colour jitter.
    """
    if random.random() < flip_prob:
        image = cv2.flip(image, 1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= random.uniform(1 - jitter, 1 + jitter)   # saturation
    hsv[..., 2] *= random.uniform(1 - jitter, 1 + jitter)   # value
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ── YOLO label utilities ───────────────────────────────────────────────────────

def yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                 img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Convert normalised YOLO box to pixel absolute coordinates."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def load_yolo_labels(label_path: str) -> List[Dict]:
    """Parse a YOLO label file into a list of dicts."""
    rows = []
    p    = Path(label_path)
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_idx = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        rows.append({'class': cls_idx, 'cx': cx, 'cy': cy,
                     'w': bw, 'h': bh})
    return rows


def visualise_labels(image_path: str, label_path: str,
                     classes: Optional[List[str]] = None,
                     save_path: Optional[str] = None) -> np.ndarray:
    """
    Draw ground-truth YOLO boxes on an image.
    Returns annotated BGR numpy array.
    """
    classes = classes or LOGO_CLASSES
    img     = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f'Image not found: {image_path}')
    ih, iw = img.shape[:2]

    for lbl in load_yolo_labels(label_path):
        x1, y1, x2, y2 = yolo_to_xyxy(
            lbl['cx'], lbl['cy'], lbl['w'], lbl['h'], iw, ih)
        cls_name = classes[lbl['class']] if lbl['class'] < len(classes) else '?'
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 100), 2)
        cv2.putText(img, cls_name, (x1, max(y1-5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 2)

    if save_path:
        cv2.imwrite(str(save_path), img)

    return img


# ── Dataset stats ──────────────────────────────────────────────────────────────

def dataset_stats(yolo_root: str, classes: Optional[List[str]] = None) -> Dict:
    """Return per-class counts for train and val splits."""
    classes = classes or LOGO_CLASSES
    stats   = {'train': {c: 0 for c in classes},
               'val':   {c: 0 for c in classes}}
    for split in ('train', 'val'):
        lbl_dir = Path(yolo_root) / 'labels' / split
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.glob('*.txt'):
            for lbl in load_yolo_labels(str(lbl_file)):
                idx = lbl['class']
                if idx < len(classes):
                    stats[split][classes[idx]] += 1
    return stats
