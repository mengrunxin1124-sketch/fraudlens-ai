# FraudLens AI API

## Overview

The FraudLens AI API provides real-time fraud prediction using the strict HistGradientBoosting baseline model.

The service is implemented with FastAPI.

---

## Start the API

```bash
uvicorn src.api.main:app --reload
```

The API will be available at

http://127.0.0.1:8000

---

## Interactive Documentation

Swagger UI:

http://127.0.0.1:8000/docs

OpenAPI JSON:

http://127.0.0.1:8000/openapi.json

---

## Health Check

### Request

GET /health

### Response

```json
{
  "status": "ok"
}
```

---

## Fraud Prediction

### Request

POST /predict

### Example Input

```json
{
  "step": 500,
  "type": "TRANSFER",
  "amount": 9000,
  "oldbalanceOrg": 20000,
  "oldbalanceDest": 1000
}
```

### Example Response

```json
{
  "fraud_probability": 0.000085,
  "prediction": 0
}
```

---

## Model

Current production model:

- HistGradientBoostingClassifier
- Strict real-time feature set
- Temporal train/validation/test split

---

## Notes

This API is intended for research and demonstration purposes.
