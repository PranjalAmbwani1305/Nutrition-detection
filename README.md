# 🔍 Logo Detection — GAN + YOLOv8

> End-to-end logo detection system using DCGAN for data augmentation and YOLOv8 for detection.  
> Train on Kaggle · Deploy locally with Streamlit.

---

## 📋 Project Overview

```
Raw Data ──► Preprocessing ──► DCGAN Training ──► Synthetic Images
                                                         │
                                              Merge with Real Data
                                                         │
                                              YOLOv8 Training ──► best.pt
                                                         │
                                              Streamlit App ──► Inference
```

| Component | Technology |
|---|---|
| GAN | DCGAN (PyTorch) — 64×64 output |
| Detector | YOLOv8n (Ultralytics) |
| Dataset | FlickrLogos-32 (10 classes) |
| Training env | Kaggle GPU (T4 / P100) |
| Deployment | Streamlit (local) |

---

## 📁 Project Structure

```
logo_detection_project/
├── kaggle_notebook.ipynb   # Full training pipeline (run on Kaggle)
├── app.py                  # Streamlit inference app
├── requirements.txt        # Python dependencies
├── model/
│   ├── README.md
│   └── best.pt             # ← Place trained weights here
└── utils/
    ├── __init__.py
    ├── preprocess.py       # Dataset conversion & preprocessing
    ├── gan_utils.py        # DCGAN architecture & helpers
    └── yolo_utils.py       # YOLOv8 training & evaluation helpers
```

---

## ⚡ Quick Start

### 1 · Clone / download the project

```bash
git clone https://github.com/yourname/logo-detection-gan-yolo.git
cd logo-detection-gan-yolo
```

### 2 · Install dependencies

```bash
pip install -r requirements.txt
```

### 3 · Train on Kaggle (see section below)

### 4 · Download model weights

After training, download `best.pt` and put it in `model/best.pt`.

### 5 · Run the Streamlit app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🏋️ Kaggle Training

### Step 1 — Set up Kaggle

1. Go to [kaggle.com](https://www.kaggle.com) and create an account.
2. Navigate to **Code → New Notebook**.
3. In **Settings** → Enable GPU (P100 or T4).

### Step 2 — Add the dataset

1. Click **Add Data** (right panel).
2. Search for **flickrlogos32** and add it.  
   *(Or use the Kaggle API: `kaggle datasets download -d shahraizanwar/flickrlogos32`)*

### Step 3 — Upload the notebook

1. Click **File → Import Notebook**.
2. Upload `kaggle_notebook.ipynb`.

### Step 4 — Run all cells

Click **Run All** (≈ 45–70 min on T4 GPU).

The notebook will:
- Preprocess FlickrLogos-32 → YOLO format
- Train DCGAN for 25 epochs → generate 500 synthetic images
- Train YOLOv8n on original data (30 epochs)
- Train YOLOv8n on GAN-augmented data (30 epochs)
- Compare results and save plots

### Step 5 — Download outputs

From the **Output** panel, download `logo_detection_outputs.zip`.  
Extract and copy `best.pt` → `model/best.pt`.

---

## 🌐 Streamlit App

```bash
streamlit run app.py
```

### Features

| Feature | Description |
|---|---|
| Image upload | JPEG · PNG · WebP · BMP |
| Detection | Bounding boxes + confidence scores |
| GAN toggle | Switch between baseline and GAN-augmented model |
| Confidence slider | Adjust detection threshold (0.10–0.95) |
| IoU slider | Adjust NMS threshold |
| Download | Save annotated image |
| Demo mode | Runs fake detections if model not loaded |

---

## 🧠 GAN Architecture

```
Generator (DCGAN)
  Latent Z (100×1×1)
    ↓ ConvTranspose2d → BN → ReLU   [512 × 4×4]
    ↓ ConvTranspose2d → BN → ReLU   [256 × 8×8]
    ↓ ConvTranspose2d → BN → ReLU   [128 × 16×16]
    ↓ ConvTranspose2d → BN → ReLU   [64  × 32×32]
    ↓ ConvTranspose2d → Tanh        [3   × 64×64]

Discriminator (DCGAN)
  Image (3×64×64)
    ↓ Conv2d → LeakyReLU            [64  × 32×32]
    ↓ Conv2d → BN → LeakyReLU       [128 × 16×16]
    ↓ Conv2d → BN → LeakyReLU       [256 × 8×8]
    ↓ Conv2d → BN → LeakyReLU       [512 × 4×4]
    ↓ Conv2d → Sigmoid              [1]
```

**Training config:**
- Epochs: 25 (Kaggle-safe)
- Batch size: 64
- Optimizer: Adam (lr=0.0002, β₁=0.5)
- Loss: Binary Cross-Entropy

---

## 🎯 YOLOv8 Training

| Setting | Value |
|---|---|
| Model | YOLOv8n (nano — fastest) |
| Epochs | 30 |
| Image size | 640 × 640 |
| Batch | 16 |
| Patience | 10 (early stop) |
| Augmentation | Mosaic · Mixup · HSV · Flip |

Two runs are compared:
1. **Original** — real FlickrLogos-32 images only
2. **GAN-Augmented** — real + 500 GAN-generated images

---

## 📊 Expected Results

| Metric | Original | GAN-Augmented |
|---|---|---|
| mAP@50 | ~0.72 | ~0.76 |
| mAP@50-95 | ~0.48 | ~0.52 |
| Precision | ~0.74 | ~0.77 |
| Recall | ~0.69 | ~0.73 |

*Results vary with dataset size, GPU, and random seed.*

---

## 🔧 Configuration

Edit constants at the top of `kaggle_notebook.ipynb` Cell 5:

```python
GAN_EPOCHS     = 25     # Increase for better GAN quality
GENERATE_COUNT = 500    # Synthetic images to produce
BATCH_SIZE     = 64     # GAN batch size
NZ             = 100    # Latent vector size
```

---

## 📦 Logo Classes

`adidas` · `apple` · `bmw` · `cocacola` · `fedex`  
`ferrari` · `ford` · `google` · `gucci` · `hp`

To use all 32 FlickrLogos classes, update `LOGO_CLASSES` in the notebook.

---

## 🚫 Troubleshooting

| Issue | Fix |
|---|---|
| `model/best.pt not found` | Download from Kaggle outputs and copy to `model/` |
| CUDA out of memory | Reduce `BATCH_SIZE` or use `yolov8n` |
| Streamlit import error | Run `pip install streamlit ultralytics` |
| Dataset not found | Add FlickrLogos-32 via Kaggle "Add Data" |
| Low mAP | Increase `GAN_EPOCHS` and `GENERATE_COUNT` |

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## 🙏 References

- [DCGAN Paper](https://arxiv.org/abs/1511.06434) — Radford et al. 2015
- [YOLOv8 Docs](https://docs.ultralytics.com)
- [FlickrLogos-32](http://www.multimedia-computing.de/flickrlogos/)
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
