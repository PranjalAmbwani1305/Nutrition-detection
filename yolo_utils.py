"""
utils/yolo_utils.py
===================
YOLOv8 training helpers, evaluation utilities, and result visualisation
for the Logo Detection pipeline.
"""

import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ── Training ───────────────────────────────────────────────────────────────────

def train_yolo(
    yaml_path: str,
    run_name: str,
    project_dir: str,
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    model_size: str = 'n',
    device: Optional[str] = None,
    patience: int = 10,
    resume: bool = False,
) -> Path:
    """
    Train a YOLOv8 model and return the path to best.pt.

    Parameters
    ----------
    yaml_path   : Path to dataset.yaml
    run_name    : Subdirectory name inside project_dir
    project_dir : Parent directory for run outputs
    epochs      : Number of training epochs
    imgsz       : Input image size
    batch       : Batch size
    model_size  : YOLOv8 variant  n | s | m | l | x
    device      : '0' for GPU, 'cpu', or None (auto)
    patience    : Early-stopping patience
    resume      : Resume from last checkpoint

    Returns
    -------
    Path to best.pt weights file.
    """
    import torch
    from ultralytics import YOLO

    if device is None:
        device = '0' if torch.cuda.is_available() else 'cpu'

    model = YOLO(f'yolov8{model_size}.pt')
    model.train(
        data     = str(yaml_path),
        epochs   = epochs,
        imgsz    = imgsz,
        batch    = batch,
        device   = device,
        project  = str(project_dir),
        name     = run_name,
        exist_ok = True,
        patience = patience,
        resume   = resume,
        verbose  = True,
    )
    best = Path(project_dir) / run_name / 'weights' / 'best.pt'
    return best


# ── Metrics ────────────────────────────────────────────────────────────────────

def load_results_csv(run_dir: str) -> List[Dict]:
    """Parse ultralytics results.csv and return list of epoch dicts."""
    csv_path = Path(run_dir) / 'results.csv'
    if not csv_path.exists():
        return []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        return [{k.strip(): safe_float(v) for k, v in row.items()}
                for row in reader]


def safe_float(v: str) -> float:
    try:
        return float(v.strip())
    except (ValueError, AttributeError):
        return 0.0


def last_epoch_metrics(run_dir: str) -> Dict[str, float]:
    """Return metric dict for the final epoch of a training run."""
    rows = load_results_csv(run_dir)
    return rows[-1] if rows else {}


def compare_runs(
    orig_dir: str,
    aug_dir: str,
    metric_keys: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare two training runs and plot a grouped bar chart.

    Returns
    -------
    dict with keys 'original' and 'augmented', each mapping metric → value.
    """
    metric_keys = metric_keys or [
        'metrics/mAP50(B)',
        'metrics/mAP50-95(B)',
        'metrics/precision(B)',
        'metrics/recall(B)',
    ]
    labels_pretty = ['mAP@50', 'mAP@50-95', 'Precision', 'Recall']

    orig_m = last_epoch_metrics(orig_dir)
    aug_m  = last_epoch_metrics(aug_dir)

    orig_vals = [orig_m.get(k, 0.0) for k in metric_keys]
    aug_vals  = [aug_m.get(k, 0.0) for k in metric_keys]

    x, w = np.arange(len(labels_pretty)), 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, orig_vals, w, label='Original',
                color='#457b9d', alpha=0.9)
    b2 = ax.bar(x + w/2, aug_vals,  w, label='GAN-Augmented',
                color='#e63946', alpha=0.9)

    for bar in (*b1, *b2):
        ax.annotate(f'{bar.get_height():.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

    ax.set_title('YOLOv8: Original vs GAN-Augmented',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels_pretty)
    ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return {
        'original':  dict(zip(labels_pretty, orig_vals)),
        'augmented': dict(zip(labels_pretty, aug_vals)),
    }


def plot_training_curve(
    run_dir: str,
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot training curves (loss + mAP) from a run directory."""
    rows = load_results_csv(run_dir)
    if not rows:
        print(f'No results.csv found in {run_dir}')
        return

    metrics = metrics or [
        'train/box_loss', 'train/cls_loss',
        'val/box_loss',   'val/cls_loss',
        'metrics/mAP50(B)'
    ]
    epochs = [r.get('epoch', i+1) for i, r in enumerate(rows)]

    n    = len(metrics)
    cols = min(3, n)
    rows_plot = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_plot, cols,
                              figsize=(cols * 5, rows_plot * 4))
    axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]

    for ax, m in zip(axes_flat, metrics):
        vals = [r.get(m, 0.0) for r in rows]
        ax.plot(epochs, vals, lw=2, color='#e63946')
        ax.set_title(m, fontsize=10)
        ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3)

    for ax in list(axes_flat)[n:]:
        ax.axis('off')

    plt.suptitle(f'Training Curves — {Path(run_dir).name}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ── Inference ──────────────────────────────────────────────────────────────────

LOGO_CLASSES = [
    'adidas', 'apple', 'bmw', 'cocacola', 'fedex',
    'ferrari', 'ford', 'google', 'gucci', 'hp'
]

CLASS_COLORS = [
    (30, 30, 30), (180, 180, 180), (0, 80, 200), (220, 20, 20),
    (100, 0, 220), (220, 30, 30), (0, 40, 160), (0, 160, 60),
    (20, 90, 20), (0, 100, 200),
]


def predict_image(
    model_path: str,
    image_path: str,
    conf: float = 0.35,
    iou: float = 0.45,
    classes: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Run YOLOv8 inference on a single image.

    Returns
    -------
    annotated_img : BGR numpy array with bounding boxes drawn
    detections    : list of detection dicts
    """
    from ultralytics import YOLO
    classes = classes or LOGO_CLASSES

    model   = YOLO(model_path)
    results = model.predict(str(image_path), conf=conf, iou=iou, verbose=False)

    img = cv2.imread(str(image_path))
    if img is None:
        img = np.zeros((640, 640, 3), dtype=np.uint8)

    detections: List[Dict] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_idx  = int(box.cls[0].item())
            cls_name = (classes[cls_idx] if cls_idx < len(classes)
                        else f'class_{cls_idx}')
            conf_val = float(box.conf[0].item())
            detections.append({
                'box':        (x1, y1, x2, y2),
                'class':      cls_name,
                'class_idx':  cls_idx,
                'confidence': conf_val,
            })

            color = CLASS_COLORS[cls_idx % len(CLASS_COLORS)]
            color_bgr = (color[2], color[1], color[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
            label = f'{cls_name} {conf_val:.0%}'
            cv2.putText(img, label, (x1, max(y1-5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

    if save_path:
        cv2.imwrite(str(save_path), img)

    return img, detections


def batch_predict(
    model_path: str,
    image_dir: str,
    output_dir: str,
    conf: float = 0.35,
    iou: float = 0.45,
    n_images: Optional[int] = None,
) -> List[Dict]:
    """Run inference on a directory of images. Returns all detections."""
    from ultralytics import YOLO
    model    = YOLO(model_path)
    img_dir  = Path(image_dir)
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = (
        list(img_dir.glob('*.jpg')) +
        list(img_dir.glob('*.png'))
    )
    if n_images:
        image_paths = image_paths[:n_images]

    all_dets: List[Dict] = []
    for p in image_paths:
        img, dets = predict_image(
            model_path, str(p), conf=conf, iou=iou,
            save_path=str(out_dir / p.name)
        )
        for d in dets:
            d['image'] = p.name
        all_dets.extend(dets)

    print(f'Processed {len(image_paths)} images, {len(all_dets)} detections')
    return all_dets


def visualise_predictions_grid(
    model_path: str,
    image_paths: List[str],
    nrow: int = 3,
    conf: float = 0.35,
    save_path: Optional[str] = None,
) -> None:
    """Visualise inference results in a grid."""
    ncol = min(nrow, len(image_paths))
    nrow_plot = (len(image_paths) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow_plot, ncol,
                              figsize=(ncol * 5, nrow_plot * 5))
    axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]

    for ax, img_path in zip(axes_flat, image_paths):
        annotated, dets = predict_image(model_path, img_path, conf=conf)
        ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax.set_title(Path(img_path).name + f'  ({len(dets)} det.)', fontsize=8)
        ax.axis('off')

    for ax in list(axes_flat)[len(image_paths):]:
        ax.axis('off')

    plt.suptitle('YOLOv8 Logo Detection Results',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
