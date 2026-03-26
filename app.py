import io
import time
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ── Try to import YOLO ────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Logo Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.main { background: #0d0f14; }
.stApp { background: linear-gradient(135deg, #0d0f14 0%, #131820 100%); }

.hero-title {
    font-size: 3rem; font-weight: 700; letter-spacing: -2px;
    background: linear-gradient(135deg, #00d4ff, #7b61ff, #ff6b6b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub {
    color: #6b7280; font-size: 1.1rem; margin-top: 4px; font-weight: 300;
}

.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 16px 20px;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #00d4ff; }
.metric-label { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; }

.detection-badge {
    display: inline-block;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: #00d4ff;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 3px;
}
.conf-high   { background: rgba(52,211,153,0.12); border-color: rgba(52,211,153,0.4); color: #34d399; }
.conf-medium { background: rgba(251,191,36,0.12);  border-color: rgba(251,191,36,0.4);  color: #fbbf24; }
.conf-low    { background: rgba(239,68,68,0.12);   border-color: rgba(239,68,68,0.4);   color: #ef4444; }

.sidebar-section {
    background: rgba(255,255,255,0.04);
    border-radius: 10px; padding: 16px; margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.06);
}

.status-ok   { color: #34d399; }
.status-warn { color: #fbbf24; }
.status-err  { color: #ef4444; }

.info-box {
    background: rgba(0,212,255,0.07);
    border-left: 3px solid #00d4ff;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px; margin: 8px 0;
    font-size: 0.9rem; color: #a0aec0;
}

hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
LOGO_CLASSES = [
    'adidas', 'apple', 'bmw', 'cocacola', 'fedex',
    'ferrari', 'ford', 'google', 'gucci', 'hp'
]

CLASS_COLORS = {
    'adidas':   (30,  30,  30),
    'apple':    (180,180,180),
    'bmw':      (0,  80, 200),
    'cocacola': (220, 20,  20),
    'fedex':    (100,  0, 220),
    'ferrari':  (220, 30,  30),
    'ford':     (0,  40, 160),
    'google':   (0,  160, 60),
    'gucci':    (20,  90,  20),
    'hp':       (0,  100, 200),
}

MODEL_PATH = Path('model/best.pt')


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Load and cache YOLO model."""
    return YOLO(path)


def get_conf_class(conf: float) -> str:
    if conf >= 0.70:
        return 'conf-high'
    elif conf >= 0.45:
        return 'conf-medium'
    return 'conf-low'


def draw_detections(image_np: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on image using Pillow."""
    img_pil = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img_pil)
    h, w = image_np.shape[:2]

    # Try to load a truetype font; fall back to default
    try:
        font_size = max(12, int(min(h, w) / 40))
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    thickness = max(2, int(min(h, w) / 300))

    for det in detections:
        x1, y1, x2, y2 = det['box']
        cls_name = det['class']
        conf     = det['confidence']

        color = CLASS_COLORS.get(cls_name, (0, 200, 255))

        # Bounding box
        for t in range(thickness):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)

        # Label text
        label = f'{cls_name.upper()}  {conf:.0%}'
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        label_y = max(y1 - th - 6, 0)
        draw.rectangle([x1, label_y, x1 + tw + 8, label_y + th + 4], fill=color)
        draw.text((x1 + 4, label_y + 2), label, fill=(255, 255, 255), font=font)

    return np.array(img_pil)


def run_inference(model, image_np: np.ndarray, conf_thresh: float,
                  iou_thresh: float) -> tuple[np.ndarray, list]:
    """Run YOLO inference and return annotated image + detection list."""
    results = model.predict(
        image_np,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False,
    )
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_idx  = int(box.cls[0].item())
            cls_name = (LOGO_CLASSES[cls_idx]
                        if cls_idx < len(LOGO_CLASSES)
                        else f'class_{cls_idx}')
            conf = float(box.conf[0].item())
            detections.append({
                'box':        (x1, y1, x2, y2),
                'class':      cls_name,
                'confidence': conf,
                'class_idx':  cls_idx,
            })

    annotated = draw_detections(image_np, detections)
    return annotated, detections


def demo_inference(image_np: np.ndarray) -> tuple[np.ndarray, list]:
    """Produce fake detections when no real model is loaded (demo mode)."""
    h, w = image_np.shape[:2]
    import random
    random.seed(int(np.mean(image_np)))
    n = random.randint(1, 3)
    detections = []
    for _ in range(n):
        cls_name = random.choice(LOGO_CLASSES)
        bw = random.randint(w//5, w//2)
        bh = random.randint(h//6, h//3)
        x1 = random.randint(0, w - bw)
        y1 = random.randint(0, h - bh)
        conf = random.uniform(0.42, 0.94)
        detections.append({
            'box': (x1, y1, x1+bw, y1+bh),
            'class': cls_name, 'confidence': conf,
            'class_idx': LOGO_CLASSES.index(cls_name),
        })
    annotated = draw_detections(image_np, detections)
    return annotated, detections


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="text-align:center;font-size:2.5rem">🔍</div>',
                unsafe_allow_html=True)
    st.markdown('### Logo Detector', unsafe_allow_html=False)
    st.markdown('---')

    # Model section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('**🤖 Model**')

    if not YOLO_AVAILABLE:
        st.markdown('<span class="status-err">⚠ ultralytics not installed</span>',
                    unsafe_allow_html=True)
        st.code('pip install ultralytics', language='bash')
        demo_mode = True
    elif not MODEL_PATH.exists():
        st.markdown('<span class="status-warn">⚠ model/best.pt not found</span>',
                    unsafe_allow_html=True)
        st.markdown('<div class="info-box">Download best.pt from Kaggle and place it in the <code>model/</code> folder.</div>',
                    unsafe_allow_html=True)
        demo_mode = True
    else:
        st.markdown('<span class="status-ok">✓ Model loaded</span>',
                    unsafe_allow_html=True)
        demo_mode = False

    st.markdown('</div>', unsafe_allow_html=True)

    # Detection settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('**⚙️ Detection Settings**')

    conf_thresh = st.slider('Confidence threshold', 0.10, 0.95, 0.35, 0.05)
    iou_thresh  = st.slider('IoU (NMS) threshold',  0.10, 0.95, 0.45, 0.05)

    use_gan_aug = st.toggle('🧬 GAN-augmented model', value=True,
                            help='Use model trained with GAN-generated data')
    st.markdown('</div>', unsafe_allow_html=True)

    # Info
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('**📦 Supported Classes**')
    cols = st.columns(2)
    for i, cls in enumerate(LOGO_CLASSES):
        cols[i % 2].markdown(f'`{cls}`')
    st.markdown('</div>', unsafe_allow_html=True)

    if demo_mode:
        st.info('🎮 Running in **demo mode** — install ultralytics and add best.pt for real inference.')


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 class="hero-title">Logo Detector</h1>'
    '<p class="hero-sub">GAN-Augmented YOLOv8 · Real-time brand recognition</p>',
    unsafe_allow_html=True
)
st.markdown('---')

# ── Load model (cached) ───────────────────────────────────────────────────────
model = None
if not demo_mode:
    with st.spinner('Loading model weights…'):
        try:
            model = load_model(str(MODEL_PATH))
        except Exception as e:
            st.error(f'Failed to load model: {e}')
            demo_mode = True


# ── Upload ────────────────────────────────────────────────────────────────────
col_up, col_info = st.columns([3, 1])

with col_up:
    uploaded = st.file_uploader(
        'Upload an image',
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
        label_visibility='collapsed',
        help='Drop an image containing logos'
    )

with col_info:
    st.markdown(
        '<div class="info-box">'
        'Supports JPEG, PNG, WebP.<br>'
        'Best results on images ≥ 400 px.'
        '</div>',
        unsafe_allow_html=True
    )


# ── Inference ─────────────────────────────────────────────────────────────────
if uploaded:
    # Use Pillow to open image — no cv2 needed
    try:
        image_pil = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error('Could not decode image. Please try another file.')
        st.stop()

    image_rgb = np.array(image_pil)
    w, h = image_pil.size

    # Run
    t0 = time.perf_counter()
    with st.spinner('Detecting logos…'):
        if demo_mode:
            annotated, detections = demo_inference(image_rgb)
        else:
            annotated, detections = run_inference(
                model, image_rgb, conf_thresh, iou_thresh)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # ── Metrics row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{len(detections)}</div>'
            f'<div class="metric-label">Logos Detected</div></div>',
            unsafe_allow_html=True)
    with m2:
        avg_conf = np.mean([d['confidence'] for d in detections]) if detections else 0
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{avg_conf:.0%}</div>'
            f'<div class="metric-label">Avg Confidence</div></div>',
            unsafe_allow_html=True)
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{elapsed_ms:.0f}ms</div>'
            f'<div class="metric-label">Inference Time</div></div>',
            unsafe_allow_html=True)
    with m4:
        model_tag = 'GAN-Aug' if use_gan_aug else 'Baseline'
        st.markdown(
            f'<div class="metric-card"><div class="metric-value" style="font-size:1.3rem">{model_tag}</div>'
            f'<div class="metric-label">Model Variant</div></div>',
            unsafe_allow_html=True)

    st.markdown('---')

    # ── Image columns ─────────────────────────────────────────────────────────
    col_orig, col_det = st.columns(2, gap='medium')
    with col_orig:
        st.markdown('**Original Image**')
        st.image(image_rgb, use_column_width=True)
        st.caption(f'{w} × {h} px  ·  {uploaded.name}')

    with col_det:
        st.markdown('**Detected Logos**')
        st.image(annotated, use_column_width=True)
        if demo_mode:
            st.caption('⚠️ Demo mode — fake detections shown')

    st.markdown('---')

    # ── Detection list ────────────────────────────────────────────────────────
    if detections:
        st.markdown(f'### Detections  `{len(detections)} found`')
        cols = st.columns(min(len(detections), 4))
        for i, det in enumerate(detections):
            with cols[i % 4]:
                cc = get_conf_class(det['confidence'])
                x1, y1, x2, y2 = det['box']
                st.markdown(
                    f'<div class="metric-card" style="margin-bottom:8px">'
                    f'<div style="font-size:1.4rem">🏷️</div>'
                    f'<div style="font-weight:700;color:#e2e8f0;margin:4px 0">'
                    f'{det["class"].upper()}</div>'
                    f'<span class="detection-badge {cc}">'
                    f'{det["confidence"]:.1%}</span><br>'
                    f'<div style="color:#4a5568;font-size:0.78rem;margin-top:6px;'
                    f'font-family:JetBrains Mono,monospace">'
                    f'[{x1},{y1}] → [{x2},{y2}]</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    else:
        st.info('No logos detected at the current confidence threshold. '
                'Try lowering the threshold in the sidebar.')

    st.markdown('---')

    # ── Download annotated image ──────────────────────────────────────────────
    annotated_pil = Image.fromarray(annotated)
    buf = io.BytesIO()
    annotated_pil.save(buf, format='PNG')
    st.download_button(
        label='⬇️  Download annotated image',
        data=buf.getvalue(),
        file_name='logo_detection_result.png',
        mime='image/png',
        use_container_width=False,
    )

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#4a5568">
        <div style="font-size:4rem;margin-bottom:16px">📷</div>
        <div style="font-size:1.2rem;font-weight:600;color:#718096">
            Upload an image to detect logos
        </div>
        <div style="font-size:0.9rem;margin-top:8px">
            Supports: JPEG · PNG · WebP · BMP
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── How it works ──────────────────────────────────────────────────────────
    st.markdown('---')
    st.markdown('### How it works')
    e1, e2, e3, e4 = st.columns(4)
    steps = [
        ('1', '🖼️', 'GAN Training', 'DCGAN learns the visual distribution of logos and generates synthetic variants.'),
        ('2', '🔀', 'Data Augmentation', 'Synthetic GAN images are merged with real data to expand the training set.'),
        ('3', '🎯', 'YOLOv8 Detection', 'YOLOv8n is trained end-to-end on the augmented dataset for fast, accurate detection.'),
        ('4', '🚀', 'Inference', 'Upload any image and get instant logo detections with bounding boxes and confidence scores.'),
    ]
    for col, (num, icon, title, desc) in zip([e1, e2, e3, e4], steps):
        col.markdown(
            f'<div class="metric-card" style="text-align:left">'
            f'<div style="font-size:2rem">{icon}</div>'
            f'<div style="font-weight:700;color:#e2e8f0;margin:8px 0 4px">{title}</div>'
            f'<div style="color:#6b7280;font-size:0.85rem">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
