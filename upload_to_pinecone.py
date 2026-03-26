import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# INIT
pc = Pinecone(api_key="pcsk_4TdScE_4Dv867xrC1YoKiU32pcZKRmJ1cZvWPsgAZN9vJ6EsERX2zsjVvNw5rLhTjvjGo2")

index_name = "food-nutrition"

# CREATE INDEX (only once)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

model = SentenceTransformer("all-MiniLM-L6-v2")

# LOAD DATA
excel = pd.read_excel("nutrition.xlsx")
csv = pd.read_csv("Indian_Food_Nutrition_Processed.csv")

# CLEAN
def clean(df):
    df.columns = df.columns.str.lower().str.strip()

    rename_map = {
        "food": "name",
        "food_name": "name",
        "energy(kcal)": "calories",
        "protein(g)": "proteins",
        "carbohydrate(g)": "carbs",
        "fat(g)": "fat"
    }
    df.rename(columns=rename_map, inplace=True)

    df["name"] = df["name"].astype(str).str.lower().str.strip()
    return df

excel = clean(excel)
csv = clean(csv)

# MERGE
df = pd.concat([excel, csv], ignore_index=True).drop_duplicates("name")

# UPSERT
vectors = []

for i, row in df.iterrows():
    text = row["name"]

    embedding = model.encode(text).tolist()

    vectors.append({
        "id": str(i),
        "values": embedding,
        "metadata": {
            "name": row["name"],
            "calories": float(row.get("calories", 0)),
            "proteins": float(row.get("proteins", 0)),
            "carbs": float(row.get("carbs", 0)),
            "fat": float(row.get("fat", 0))
        }
    })

# Upload
index.upsert(vectors)

print("✅ Data uploaded to Pinecone")
