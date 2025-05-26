from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import pandas as pd

app = FastAPI(title="Customer Clustering API")

# Izinkan akses dari browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti sesuai kebutuhan
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model dan scaler
with open("kmeans_model.pkl", "rb") as f:
    kmeans_model = pickle.load(f)

with open("scaler_rfm.pkl", "rb") as f:
    scaler_rfm = pickle.load(f)

# Schema input user
class CustomerInput(BaseModel):
    Recency: float = Field(..., alias="Recency")
    Frequency: float = Field(..., alias="Frequency")
    Monetary: float = Field(..., alias="Monetary")

    class Config:
        validate_by_name = True

@app.get("/")
def root():
    return {"message": "✅ Customer Clustering API is running"}

@app.post("/predict")
def predict(data: CustomerInput):
    df = pd.DataFrame([data.dict(by_alias=True)])

    # Validasi: semua nilai 0
    if (df == 0).all(axis=1).iloc[0]:
        return {
            "error": "❌ Data input tidak valid: semua nilai RFM adalah 0. Clustering tidak dapat dilakukan."
        }

    # Validasi: ada nilai negatif
    if (df < 0).any(axis=1).iloc[0]:
        return {
            "error": "❌ Data input tidak valid: tidak boleh ada nilai negatif dalam Recency, Frequency, atau Monetary."
        }

    # Lanjutkan jika data valid
    scaled_data = scaler_rfm.transform(df)
    cluster = kmeans_model.predict(scaled_data)[0]

    return {
        "predicted_cluster": int(cluster)
    }
