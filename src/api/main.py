from fastapi import FastAPI

from src.api.schema import Transaction
from src.api.predict import predict

app = FastAPI(
    title="FraudLens AI",
    description="Real-Time Fraud Detection API",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "FraudLens AI API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
def predict_transaction(transaction: Transaction):
    return predict(transaction)
