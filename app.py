from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import pandas as pd

app = FastAPI(title="Customer Clustering API")

# Izinkan akses dari browser (JavaScript fetch)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bisa diganti dengan ['http://localhost:3000'] jika dibatasi
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
    scaled_data = scaler_rfm.transform(df)
    cluster = kmeans_model.predict(scaled_data)[0]
    return {
        "predicted_cluster": int(cluster)
    }
