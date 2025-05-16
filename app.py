from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd

app = FastAPI(title="Customer Clustering API")

with open("kmeans_model.pkl", "rb") as f:
    kmeans_model = pickle.load(f)

with open("scaler_rfm.pkl", "rb") as f:
    scaler_rfm = pickle.load(f)

# Schema input dari user
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
    # Ubah input ke DataFrame
    df = pd.DataFrame([data.dict(by_alias=True)])

    # Normalisasi data input
    scaled_data = scaler_rfm.transform(df)

    # Prediksi cluster dari data yang sudah dinormalisasi
    cluster = kmeans_model.predict(scaled_data)[0]

    return {
        "predicted_cluster": int(cluster)
    }
