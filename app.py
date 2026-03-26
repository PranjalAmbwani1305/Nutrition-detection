import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Smart Nutrition AI", layout="wide")

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    yolo = YOLO("yolov8n.pt")
    llm = pipeline("text2text-generation", model="google/flan-t5-small")
    return yolo, llm

yolo, llm = load_models()

# ─────────────────────────────────────────────
# LOAD DATASETS (CSV + EXCEL)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df_excel, df_csv = None, None

    try:
        df_excel = pd.read_excel("nutrition.xlsx")
        df_excel.columns = df_excel.columns.str.lower()
        df_excel["name"] = df_excel["name"].str.lower()
    except:
        pass

    try:
        df_csv = pd.read_csv("Indian_Food_Nutrition_Processed.csv")
        df_csv.columns = df_csv.columns.str.lower()

        df_csv.rename(columns={
            "food": "name",
            "energy(kcal)": "calories",
            "protein(g)": "proteins",
            "carbohydrate(g)": "carbs",
            "fat(g)": "fat"
        }, inplace=True)

        df_csv["name"] = df_csv["name"].astype(str).str.lower()
    except:
        pass

    return df_excel, df_csv

df_excel, df_csv = load_data()

# ─────────────────────────────────────────────
# FALLBACK DATA
# ─────────────────────────────────────────────
FALLBACK = {
    "samosa": {"calories":262,"proteins":4,"carbs":31,"fat":14},
    "burger": {"calories":295,"proteins":17,"carbs":30,"fat":12},
    "pizza": {"calories":266,"proteins":11,"carbs":33,"fat":10},
    "unknown": {"calories":250,"proteins":5,"carbs":30,"fat":10}
}

# ─────────────────────────────────────────────
# LOOKUP FUNCTION
# ─────────────────────────────────────────────
def get_nutrition(label):
    label = label.lower()

    if df_excel is not None:
        row = df_excel[df_excel["name"] == label]
        if not row.empty:
            r = row.iloc[0]
            return r.to_dict()

    if df_csv is not None:
        row = df_csv[df_csv["name"].str.contains(label, na=False)]
        if not row.empty:
            r = row.iloc[0]
            return r.to_dict()

    return FALLBACK.get(label, FALLBACK["unknown"])

# ─────────────────────────────────────────────
# HEALTH SCORE
# ─────────────────────────────────────────────
def health_score(n):
    score = 100
    if n["calories"] > 300:
        score -= 20
    if n["fat"] > 15:
        score -= 20
    if n["proteins"] > 10:
        score += 10
    return max(0, min(100, score))

# ─────────────────────────────────────────────
# LLM EXPLANATION
# ─────────────────────────────────────────────
def explain(food, n):
    prompt = f"""
    Food: {food}
    Calories: {n['calories']}
    Protein: {n['proteins']}
    Carbs: {n['carbs']}
    Fat: {n['fat']}

    Explain if it's healthy and when to eat.
    """

    result = llm(prompt, max_length=100)
    return result[0]["generated_text"]

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🥗 Smart Nutrition AI System")

uploaded = st.file_uploader("Upload Food Image", type=["jpg","png","jpeg"])

if not uploaded:
    st.stop()

image = Image.open(uploaded).convert("RGB")
img_np = np.array(image)

st.image(image, caption="Uploaded Image")

# YOLO DETECTION
results = yolo.predict(img_np, conf=0.3, verbose=False)

label = None

for r in results:
    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = yolo.names[cls_id]
            break

if label is None:
    label = "samosa"  # fallback for demo

st.write(f"🔍 Detected: **{label}**")

# GET NUTRITION
nutrition = get_nutrition(label)

# UI DISPLAY
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.metric("🔥 Calories", nutrition.get("calories", 0))
    st.metric("💪 Protein", nutrition.get("proteins", 0))
    st.metric("🍞 Carbs", nutrition.get("carbs", 0))
    st.metric("🧈 Fat", nutrition.get("fat", 0))

with col2:
    score = health_score(nutrition)
    st.metric("🏥 Health Score", score)

    if score > 70:
        st.success("Healthy ✅")
    elif score > 40:
        st.warning("Moderate ⚠️")
    else:
        st.error("Limit ❌")

# LLM
st.markdown("---")

if st.button("🧠 Get AI Advice"):
    with st.spinner("Thinking..."):
        text = explain(label, nutrition)
    st.info(text)

# GOAL
st.markdown("---")

goal = st.selectbox("🎯 Your Goal", ["Weight Loss","Muscle Gain","Maintain"])

if goal == "Weight Loss" and nutrition["calories"] > 300:
    st.warning("High calories for weight loss")
elif goal == "Muscle Gain":
    st.success("Good protein food")

st.success("✅ Running without Pinecone (Stable Mode)")
