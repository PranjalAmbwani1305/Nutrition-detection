"""
Smart Nutrition Detection System
=================================
Upload a food image → YOLOv8 detects food items → estimates weight → shows nutrition.

Dataset: https://www.kaggle.com/datasets/gokulprasantht/nutrition-dataset
  Place the downloaded CSV as  data/nutrition.csv  next to this file.
  The app works without it too — it falls back to a built-in nutrition table.

Install:
    pip install streamlit ultralytics pillow numpy pandas

Run:
    streamlit run app.py
"""

import io
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ── optional YOLO ─────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Nutrition Detection",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# CSS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f1eb; color: #1c1409; }

/* hero */
.hero {
    background: linear-gradient(140deg, #1c1409 0%, #3b2110 55%, #6b3d1e 100%);
    border-radius: 22px; padding: 52px 56px 44px;
    margin-bottom: 32px; position: relative; overflow: hidden;
}
.hero::after {
    content:''; position:absolute; top:-60px; right:-60px;
    width:340px; height:340px; border-radius:50%;
    background: radial-gradient(circle, rgba(212,163,89,0.22) 0%, transparent 70%);
}
.hero-eyebrow {
    font-family:'DM Mono',monospace; font-size:0.7rem; letter-spacing:4px;
    color:#d4a359; text-transform:uppercase; margin-bottom:14px;
}
.hero-title {
    font-family:'Playfair Display',serif; font-size:3rem; font-weight:900;
    color:#fdf6ec; line-height:1.08; margin:0 0 16px;
}
.hero-sub { font-size:1rem; color:rgba(253,246,236,0.6); font-weight:300; margin:0; }

/* section header */
.sec-head {
    font-family:'Playfair Display',serif; font-size:1.35rem; font-weight:700;
    color:#1c1409; margin:0 0 14px; display:flex; align-items:center; gap:9px;
}

/* calorie hero */
.cal-hero {
    background: linear-gradient(130deg,#d4a359,#b8722a);
    border-radius:16px; padding:26px 32px;
    display:flex; align-items:flex-end; justify-content:space-between;
    margin-bottom:20px;
}
.cal-num {
    font-family:'Playfair Display',serif; font-size:3.6rem;
    font-weight:900; color:#fff; line-height:1;
}
.cal-unit { font-size:1.1rem; color:rgba(255,255,255,0.7); margin-left:6px; }
.cal-label {
    font-family:'DM Mono',monospace; font-size:0.68rem;
    letter-spacing:3px; text-transform:uppercase; color:rgba(255,255,255,0.65);
    margin-bottom:6px;
}
.cal-note { font-size:0.82rem; color:rgba(255,255,255,0.55); font-style:italic; }

/* macro chips */
.macro-strip { display:flex; gap:10px; flex-wrap:wrap; margin-top:8px; }
.macro-chip { border-radius:30px; padding:5px 14px; font-size:0.8rem; font-weight:500; }
.mc-carb { background:#fce8b2; color:#7a4a00; }
.mc-prot { background:#b3dcfd; color:#003d6b; }
.mc-fat  { background:#fdd5b3; color:#7a2a00; }
.mc-fib  { background:#c9f7d4; color:#0a5c1e; }
.mc-cal  { background:#f5e6c8; color:#5c3a00; font-weight:600; }

/* confidence badge */
.conf { border-radius:20px; padding:2px 10px; font-size:0.75rem;
        font-family:'DM Mono',monospace; font-weight:500; }
.conf-h { background:#d4edda; color:#155724; }
.conf-m { background:#fff3cd; color:#856404; }
.conf-l { background:#f8d7da; color:#721c24; }

/* tip box */
.tip {
    background:#fdf6ec; border-left:4px solid #d4a359;
    border-radius:0 10px 10px 0; padding:12px 16px;
    font-size:0.88rem; color:#6b4226; margin:12px 0;
}

/* sidebar */
section[data-testid="stSidebar"] > div:first-child { background: #1c1409 !important; }
section[data-testid="stSidebar"] * { color: #fdf6ec !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(212,163,89,0.25) !important; }

/* hide chrome */
#MainMenu, footer { visibility:hidden; }
div[data-testid="stDecoration"] { display:none; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# BUILT-IN NUTRITION TABLE (per 100 g)
# Covers all COCO food-related classes + common extras.
# Used when Kaggle CSV is absent OR item not found in it.
# ═════════════════════════════════════════════════════════════════════════════
BUILTIN_NUTRITION = {
    "banana":     {"calories":89,  "proteins":1.1, "carbohydrates":23.0,"fat":0.3, "fiber":2.6},
    "apple":      {"calories":52,  "proteins":0.3, "carbohydrates":14.0,"fat":0.2, "fiber":2.4},
    "orange":     {"calories":47,  "proteins":0.9, "carbohydrates":12.0,"fat":0.1, "fiber":2.4},
    "broccoli":   {"calories":34,  "proteins":2.8, "carbohydrates":6.6, "fat":0.4, "fiber":2.6},
    "carrot":     {"calories":41,  "proteins":0.9, "carbohydrates":10.0,"fat":0.2, "fiber":2.8},
    "pizza":      {"calories":266, "proteins":11.0,"carbohydrates":33.0,"fat":10.0,"fiber":2.3},
    "sandwich":   {"calories":250, "proteins":11.0,"carbohydrates":33.0,"fat":9.0, "fiber":2.0},
    "hot dog":    {"calories":290, "proteins":10.5,"carbohydrates":24.0,"fat":17.0,"fiber":0.9},
    "cake":       {"calories":347, "proteins":5.0, "carbohydrates":53.0,"fat":14.0,"fiber":0.9},
    "donut":      {"calories":452, "proteins":4.9, "carbohydrates":51.0,"fat":25.0,"fiber":1.7},
    "bottle":     {"calories":0,   "proteins":0.0, "carbohydrates":0.0, "fat":0.0, "fiber":0.0},
    "wine glass": {"calories":83,  "proteins":0.1, "carbohydrates":2.6, "fat":0.0, "fiber":0.0},
    "cup":        {"calories":2,   "proteins":0.0, "carbohydrates":0.4, "fat":0.0, "fiber":0.0},
    "bowl":       {"calories":120, "proteins":5.0, "carbohydrates":18.0,"fat":3.5, "fiber":2.0},
    "_default":   {"calories":150, "proteins":5.0, "carbohydrates":20.0,"fat":5.0, "fiber":2.0},
}

FOOD_CLASSES = {
    "banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake",
    "bowl","bottle","cup","wine glass",
}

BOX_COLORS = {
    "banana":(255,200,30), "apple":(220,50,50),   "orange":(255,140,0),
    "broccoli":(50,180,50),"carrot":(255,110,30),  "pizza":(200,80,20),
    "sandwich":(200,160,60),"hot dog":(210,100,40),"cake":(220,120,180),
    "donut":(230,150,80),  "bowl":(140,100,60),    "bottle":(80,160,220),
    "wine glass":(160,80,200),"cup":(100,180,200),
}
DEFAULT_COLOR = (100,160,240)


# ═════════════════════════════════════════════════════════════════════════════
# LOAD KAGGLE DATASET
# Columns expected (case-insensitive):
#   name, calories, proteins, fat, carbohydrates, fiber
# ═════════════════════════════════════════════════════════════════════════════
DATASET_PATH = Path("nutrition.csv")

@st.cache_data(show_spinner=False)
def load_nutrition_csv(path: Path):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        renames = {
            "protein":"proteins","carbs":"carbohydrates",
            "carbohydrate":"carbohydrates","fibre":"fiber",
            "calorie":"calories","kcal":"calories",
        }
        df.rename(columns=renames, inplace=True)
        if not {"name","calories","proteins","fat","carbohydrates"}.issubset(df.columns):
            return None
        if "fiber" not in df.columns:
            df["fiber"] = 0.0
        df["name_lower"] = df["name"].str.lower().str.strip()
        return df
    except Exception:
        return None


def lookup_nutrition(label: str, csv_df) -> dict:
    """CSV (exact → partial) → built-in → default."""
    key = label.lower().strip()
    if csv_df is not None:
        exact = csv_df[csv_df["name_lower"] == key]
        if not exact.empty:
            r = exact.iloc[0]
            return {c: float(r.get(c, 0)) for c in
                    ["calories","proteins","carbohydrates","fat","fiber"]}
        partial = csv_df[csv_df["name_lower"].str.contains(key, na=False)]
        if not partial.empty:
            r = partial.iloc[0]
            return {c: float(r.get(c, 0)) for c in
                    ["calories","proteins","carbohydrates","fat","fiber"]}
    return dict(BUILTIN_NUTRITION.get(key, BUILTIN_NUTRITION["_default"]))


# ═════════════════════════════════════════════════════════════════════════════
# WEIGHT ESTIMATION
# Heuristic: a bounding box covering ~15% of image area ≈ hint_weight grams.
# Scales with sqrt of area fraction (approximates volume scaling).
# ═════════════════════════════════════════════════════════════════════════════
WEIGHT_HINTS = {
    "banana":150,"apple":180,"orange":160,"broccoli":200,
    "carrot":100,"pizza":200,"sandwich":200,"hot dog":120,
    "cake":120,"donut":80,"bowl":300,"bottle":500,
    "wine glass":150,"cup":240,
}

def estimate_weight(label: str, box_area: int, img_area: int) -> float:
    hint = WEIGHT_HINTS.get(label.lower(), 150)
    ref  = 0.15
    frac = max(0.01, min(box_area / max(img_area, 1), 0.9))
    return round(max(10.0, min(hint * math.sqrt(frac / ref), 1000.0)), 1)


# ═════════════════════════════════════════════════════════════════════════════
# DRAWING  (Pillow — zero cv2)
# ═════════════════════════════════════════════════════════════════════════════
def draw_boxes(image_np: np.ndarray, detections: list) -> np.ndarray:
    img  = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)
    h, w = image_np.shape[:2]
    thick = max(2, w // 280)
    fsz   = max(13, w // 45)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fsz)
    except Exception:
        font = ImageFont.load_default()

    for d in detections:
        x1,y1,x2,y2 = d["box"]
        color = BOX_COLORS.get(d["class"].lower(), DEFAULT_COLOR)
        for t in range(thick):
            draw.rectangle([x1-t,y1-t,x2+t,y2+t], outline=color)
        text = f'{d["class"].upper()}  {d["confidence"]:.0%}  ~{d["weight_g"]}g'
        bb   = draw.textbbox((0,0), text, font=font)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        ly = max(y1 - th - 8, 0)
        draw.rectangle([x1, ly, x1+tw+10, ly+th+6], fill=color)
        draw.text((x1+5, ly+3), text, fill=(255,255,255), font=font)

    return np.array(img)


# ═════════════════════════════════════════════════════════════════════════════
# YOLO INFERENCE
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_yolo():
    if not YOLO_AVAILABLE:
        return None
    try:
        return YOLO("yolov8n.pt")   # auto-downloads ~6 MB
    except Exception:
        return None


def run_yolo(model, image_np, conf_thresh, iou_thresh, csv_df, food_only):
    h, w  = image_np.shape[:2]
    img_a = h * w
    results    = model.predict(image_np, conf=conf_thresh, iou=iou_thresh, verbose=False)
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            cls_idx  = int(box.cls[0].item())
            cls_name = model.names.get(cls_idx, f"class_{cls_idx}")
            conf     = float(box.conf[0].item())
            if food_only and cls_name.lower() not in FOOD_CLASSES:
                continue
            weight = estimate_weight(cls_name, (x2-x1)*(y2-y1), img_a)
            n100   = lookup_nutrition(cls_name, csv_df)
            factor = weight / 100.0
            nutr   = {k: round(v*factor, 1) for k,v in n100.items()}
            detections.append({"box":(x1,y1,x2,y2),"class":cls_name,
                "confidence":conf,"weight_g":weight,
                "nutrition_per100g":n100,"nutrition":nutr})
    return draw_boxes(image_np, detections), detections


def demo_run(image_np, csv_df, food_only):
    import random
    h, w  = image_np.shape[:2]
    img_a = h * w
    rng   = random.Random(int(np.mean(image_np)))
    pool  = list(FOOD_CLASSES)
    dets  = []
    for _ in range(rng.randint(2,4)):
        cls  = rng.choice(pool)
        bw   = rng.randint(w//5, w//2)
        bh   = rng.randint(h//6, h//3)
        x1   = rng.randint(0, max(1,w-bw))
        y1   = rng.randint(0, max(1,h-bh))
        conf = rng.uniform(0.44,0.93)
        wt   = estimate_weight(cls, bw*bh, img_a)
        n100 = lookup_nutrition(cls, csv_df)
        nutr = {k: round(v*wt/100, 1) for k,v in n100.items()}
        dets.append({"box":(x1,y1,x1+bw,y1+bh),"class":cls,
            "confidence":conf,"weight_g":wt,
            "nutrition_per100g":n100,"nutrition":nutr})
    return draw_boxes(image_np, dets), dets


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    conf_thresh = st.slider("Confidence threshold", 0.10, 0.95, 0.30, 0.05)
    iou_thresh  = st.slider("IoU (NMS) threshold",  0.10, 0.95, 0.45, 0.05)
    food_only   = st.toggle("Food items only", value=True,
                            help="Hide non-food COCO detections")
    st.markdown("---")
    st.markdown("### 📂 Nutrition Dataset")
    csv_df = load_nutrition_csv(DATASET_PATH)
    if csv_df is not None:
        st.success(f"✓ nutrition.csv loaded ({len(csv_df):,} items)")
    else:
        st.warning("nutrition.csv not found")
        st.caption("Download from Kaggle → save as `data/nutrition.csv` next to app.py")
    st.markdown("---")
    st.markdown("### 🤖 Model")
    if not YOLO_AVAILABLE:
        st.error("ultralytics not installed")
        st.code("pip install ultralytics")
    else:
        st.success("YOLOv8n ready  (auto-downloads ~6 MB)")
    st.markdown("---")
    st.markdown("### 📦 Detectable Food")
    for fc in sorted(FOOD_CLASSES):
        st.caption(f"• {fc}")


# ═════════════════════════════════════════════════════════════════════════════
# HERO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">YOLOv8 · Computer Vision · Nutrition AI</div>
  <h1 class="hero-title">Smart Nutrition<br>Detection System</h1>
  <p class="hero-sub">
    Upload a food photo — YOLO detects every item, estimates portion weight,
    and maps it to detailed nutritional data in real time.
  </p>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-head"><span>📷</span> Upload Food Image</div>',
            unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drag & drop or click to browse",
    type=["jpg","jpeg","png","webp","bmp"],
    label_visibility="collapsed",
)

if not uploaded:
    st.markdown("""
    <div style="text-align:center;padding:64px 20px;color:#9e8060;">
      <div style="font-size:4.5rem;margin-bottom:18px;">🍽️</div>
      <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                  font-weight:700;color:#3b2110;margin-bottom:8px;">
          No image uploaded yet
      </div>
      <div style="font-size:0.92rem;">Supports JPEG · PNG · WebP · BMP</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# PROCESS IMAGE
# ═════════════════════════════════════════════════════════════════════════════
try:
    pil_img = Image.open(uploaded).convert("RGB")
except Exception:
    st.error("Could not open image. Please try another file.")
    st.stop()

image_np = np.array(pil_img)
img_w, img_h = pil_img.size

model     = load_yolo()
demo_mode = (model is None)

with st.spinner("🔍 Detecting food items…"):
    t0 = time.perf_counter()
    if demo_mode:
        annotated, detections = demo_run(image_np, csv_df, food_only)
    else:
        annotated, detections = run_yolo(
            model, image_np, conf_thresh, iou_thresh, csv_df, food_only)
    elapsed = (time.perf_counter() - t0) * 1000

if demo_mode:
    st.info("⚠️ **Demo mode** — install `ultralytics` for real YOLOv8 inference.")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMAGES
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="sec-head"><span>🖼️</span> Detection Output</div>',
            unsafe_allow_html=True)

col_a, col_b = st.columns(2, gap="large")
with col_a:
    st.markdown("**Original Image**")
    st.image(image_np, use_column_width=True)
    st.caption(f"{img_w} × {img_h} px · {uploaded.name}")
with col_b:
    st.markdown("**Annotated Image**")
    st.image(annotated, use_column_width=True)
    st.caption(f"Inference: {elapsed:.0f} ms · {len(detections)} detection(s)")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DETECTED ITEMS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="sec-head"><span>🔎</span> Detected Food Items</div>',
            unsafe_allow_html=True)

if not detections:
    st.info("No food items detected. Try lowering the confidence threshold in the sidebar.")
    st.stop()

for i, d in enumerate(detections, 1):
    conf      = d["confidence"]
    conf_cls  = "conf-h" if conf>=0.70 else ("conf-m" if conf>=0.45 else "conf-l")
    n         = d["nutrition"]
    x1,y1,x2,y2 = d["box"]

    # lookup source label
    src_label = "CSV" if (csv_df is not None and
                          not csv_df[csv_df["name_lower"].str.contains(
                              d["class"].lower(), na=False)].empty) else "Built-in"

    st.markdown(f"""
    <div style="background:#faf6f0;border:1px solid #ead9be;
         border-radius:14px;padding:16px 20px;margin-bottom:10px;">
      <div style="display:flex;align-items:center;
                  justify-content:space-between;flex-wrap:wrap;gap:8px;">
        <div>
          <div style="font-family:'Playfair Display',serif;font-size:1.1rem;
               font-weight:700;color:#1c1409;text-transform:capitalize;">
            {i}. {d['class']}
          </div>
          <div style="font-family:'DM Mono',monospace;font-size:0.78rem;
               color:#8c6d4f;margin-top:3px;">
            Box [{x1},{y1}]→[{x2},{y2}] &nbsp;·&nbsp;
            ~{d['weight_g']} g estimated &nbsp;·&nbsp; data: {src_label}
          </div>
        </div>
        <span class="conf {conf_cls}">{conf:.1%}</span>
      </div>
      <div class="macro-strip">
        <span class="macro-chip mc-cal">🔥 {n['calories']} kcal</span>
        <span class="macro-chip mc-carb">🍞 {n['carbohydrates']} g carbs</span>
        <span class="macro-chip mc-prot">💪 {n['proteins']} g protein</span>
        <span class="macro-chip mc-fat">🧈 {n['fat']} g fat</span>
        <span class="macro-chip mc-fib">🌿 {n['fiber']} g fiber</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — NUTRITION SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="sec-head"><span>📊</span> Nutrition Summary</div>',
            unsafe_allow_html=True)

total_cal  = round(sum(d["nutrition"]["calories"]      for d in detections), 1)
total_prot = round(sum(d["nutrition"]["proteins"]       for d in detections), 1)
total_carb = round(sum(d["nutrition"]["carbohydrates"]  for d in detections), 1)
total_fat  = round(sum(d["nutrition"]["fat"]            for d in detections), 1)
total_fib  = round(sum(d["nutrition"]["fiber"]          for d in detections), 1)
total_wt   = round(sum(d["weight_g"]                    for d in detections), 1)
daily_pct  = round(total_cal / 2000 * 100, 1)

# big calorie card
st.markdown(f"""
<div class="cal-hero">
  <div>
    <div class="cal-label">Total Calories</div>
    <div><span class="cal-num">{total_cal:.0f}</span>
         <span class="cal-unit">kcal</span></div>
    <div class="cal-note">≈ {daily_pct}% of 2,000 kcal daily reference</div>
  </div>
  <div style="text-align:right;">
    <div class="cal-label">Est. Total Weight</div>
    <div style="font-family:'Playfair Display',serif;font-size:2rem;
         font-weight:700;color:#fff;line-height:1;">{total_wt:.0f} g</div>
    <div class="cal-note">{len(detections)} item(s)</div>
  </div>
</div>
""", unsafe_allow_html=True)

# table
rows = []
for d in detections:
    n = d["nutrition"]
    rows.append({
        "Food Item":       d["class"].title(),
        "Conf.":           f"{d['confidence']:.1%}",
        "Weight (g)":      d["weight_g"],
        "Calories (kcal)": round(n["calories"],1),
        "Protein (g)":     round(n["proteins"],1),
        "Carbs (g)":       round(n["carbohydrates"],1),
        "Fat (g)":         round(n["fat"],1),
        "Fiber (g)":       round(n["fiber"],1),
    })
rows.append({
    "Food Item":"🔢 TOTAL","Conf.":"—",
    "Weight (g)":total_wt,"Calories (kcal)":total_cal,
    "Protein (g)":total_prot,"Carbs (g)":total_carb,
    "Fat (g)":total_fat,"Fiber (g)":total_fib,
})

df_out = pd.DataFrame(rows)
st.dataframe(
    df_out, use_container_width=True, hide_index=True,
    column_config={
        "Calories (kcal)": st.column_config.NumberColumn(format="%.1f 🔥"),
        "Weight (g)":      st.column_config.NumberColumn(format="%.1f g"),
    },
)

# macro bar chart
st.markdown("#### Macronutrient breakdown")
macro_df = pd.DataFrame({
    "Macro": ["Carbohydrates","Protein","Fat","Fiber"],
    "Grams": [total_carb, total_prot, total_fat, total_fib],
})
st.bar_chart(macro_df.set_index("Macro"), color="#d4a359", use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# DOWNLOADS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    st.download_button("⬇️ Download annotated image", buf.getvalue(),
                       "nutrition_detection.png", "image/png",
                       use_container_width=True)
with c2:
    st.download_button("⬇️ Download nutrition CSV",
                       df_out.to_csv(index=False).encode(),
                       "nutrition_summary.csv", "text/csv",
                       use_container_width=True)

st.markdown("""
<div class="tip">
  💡 <b>Tip:</b> Download the nutrition dataset from
  <a href="https://www.kaggle.com/datasets/gokulprasantht/nutrition-dataset"
     target="_blank" style="color:#6b4226;">Kaggle</a>
  and save it as <code>data/nutrition.csv</code> beside <code>app.py</code>
  for richer lookups. The built-in table is used as fallback.
</div>
""", unsafe_allow_html=True)
