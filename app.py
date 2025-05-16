from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware 
import pickle
import pandas as pd

app = FastAPI(title="Customer Clustering API")

origins = [
    "http://perpus-digital.xyz",  # <- ganti ini
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load hanya model
with open("kmeans_model.pkl", "rb") as f:
    kmeans_model = pickle.load(f)

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

    # Tidak perlu scaling karena input sudah scaled
    cluster = kmeans_model.predict(df.values)[0]

    return {
        "predicted_cluster": int(cluster)
    }
