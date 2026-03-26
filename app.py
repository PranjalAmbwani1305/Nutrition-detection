import io
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Smart Nutrition Detection", layout="wide")

# ─────────────────────────────────────────────
# FOOD CLASSES (FILTER NOISE)
# ─────────────────────────────────────────────
FOOD_CLASSES = {
    "banana", "apple", "orange",
    "broccoli", "carrot",
    "pizza", "sandwich",
    "cake", "donut"
}

# ─────────────────────────────────────────────
# FALLBACK NUTRITION (PER 100g)
# ─────────────────────────────────────────────
FALLBACK = {
    "banana": {"calories":89,"proteins":1.1,"carbs":23,"fat":0.3},
    "apple": {"calories":52,"proteins":0.3,"carbs":14,"fat":0.2},
    "orange": {"calories":47,"proteins":0.9,"carbs":12,"fat":0.1},
    "broccoli": {"calories":34,"proteins":2.8,"carbs":6.6,"fat":0.4},
    "carrot": {"calories":41,"proteins":0.9,"carbs":10,"fat":0.2},
    "pizza": {"calories":266,"proteins":11,"carbs":33,"fat":10},
    "sandwich": {"calories":250,"proteins":11,"carbs":33,"fat":9},
    "cake": {"calories":347,"proteins":5,"carbs":53,"fat":14},
    "donut": {"calories":452,"proteins":4.9,"carbs":51,"fat":25},
}

# Weight estimation
WEIGHT_HINT = {
    "banana":150,"apple":180,"orange":160,
    "broccoli":200,"carrot":100,
    "pizza":200,"sandwich":200,
    "cake":120,"donut":80
}

# ─────────────────────────────────────────────
# LOAD EXCEL (PRIMARY SOURCE)
# ─────────────────────────────────────────────
@st.cache_data
def load_excel():
    try:
        df = pd.read_excel("nutrition.xlsx")
        df.columns = df.columns.str.lower().str.strip()

        # Normalize column names
        rename_map = {
            "protein": "proteins",
            "carbohydrate": "carbs",
            "carbohydrates": "carbs"
        }
        df.rename(columns=rename_map, inplace=True)

        df["name"] = df["name"].str.lower().str.strip()
        return df
    except:
        return None

# ─────────────────────────────────────────────
# NUTRITION LOOKUP (SMART)
# ─────────────────────────────────────────────
def get_nutrition(label, df):
    label = label.lower()

    if df is not None:
        row = df[df["name"] == label]
        if not row.empty:
            r = row.iloc[0]
            return {
                "calories": float(r.get("calories", 0)),
                "proteins": float(r.get("proteins", 0)),
                "carbs": float(r.get("carbs", 0)),
                "fat": float(r.get("fat", 0)),
            }

    return FALLBACK.get(label, {"calories":100,"proteins":2,"carbs":15,"fat":3})

# ─────────────────────────────────────────────
# LOAD YOLO
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not YOLO_AVAILABLE:
        return None
    return YOLO("yolov8n.pt")  # replace with best.pt later

# ─────────────────────────────────────────────
# WEIGHT ESTIMATION
# ─────────────────────────────────────────────
def estimate_weight(label, area, img_area):
    hint = WEIGHT_HINT.get(label, 150)
    frac = max(0.05, area / img_area)
    return round(hint * math.sqrt(frac), 1)

# ─────────────────────────────────────────────
# DRAW BOXES
# ─────────────────────────────────────────────
def draw_boxes(img, detections):
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    for d in detections:
        x1,y1,x2,y2 = d["box"]
        draw.rectangle([x1,y1,x2,y2], outline="green", width=3)

        text = f"{d['class']} {d['confidence']:.2f}"
        draw.text((x1, y1-10), text, fill="green")

    return np.array(pil)

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.title("🥗 Smart Nutrition Detection")

uploaded = st.file_uploader("Upload Food Image", type=["jpg","png","jpeg"])

if not uploaded:
    st.stop()

image = Image.open(uploaded).convert("RGB")
img_np = np.array(image)

model = load_model()
df = load_excel()

start = time.time()

detections = []

if model:
    results = model.predict(img_np, conf=0.3, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id].lower()

            # FILTER NON-FOOD
            if cls_name not in FOOD_CLASSES:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            area = (x2-x1)*(y2-y1)
            weight = estimate_weight(cls_name, area, img_np.shape[0]*img_np.shape[1])

            nutr100 = get_nutrition(cls_name, df)
            factor = weight / 100

            nutr = {k: round(v*factor,1) for k,v in nutr100.items()}

            detections.append({
                "class": cls_name,
                "confidence": conf,
                "box": (x1,y1,x2,y2),
                "weight": weight,
                "nutrition": nutr
            })

end = time.time()

annotated = draw_boxes(img_np, detections)

# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.image(img_np, caption="Original")

with col2:
    st.image(annotated, caption=f"Detected ({len(detections)}) | {round((end-start)*1000)} ms")

st.markdown("---")

total_cal = 0

for d in detections:
    n = d["nutrition"]
    total_cal += n["calories"]

    st.write(f"""
**{d['class'].title()}**
- Confidence: {round(d['confidence'],2)}
- Weight: {d['weight']} g  
- Calories: {n['calories']} kcal  
- Protein: {n['proteins']} g  
- Carbs: {n['carbs']} g  
- Fat: {n['fat']} g  
""")

st.markdown("---")
st.subheader(f"🔥 Total Calories: {round(total_cal,1)} kcal")

st.warning("⚠️ Nutrition is estimated based on visual detection")
