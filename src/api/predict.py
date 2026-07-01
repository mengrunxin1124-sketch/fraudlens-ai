from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/hist_gradient_boosting_strict_baseline.joblib")

model = joblib.load(MODEL_PATH)


FEATURE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_oldbalanceOrg_ratio",
    "amount_to_oldbalanceDest_ratio",
    "is_zero_oldbalanceOrg",
    "is_zero_oldbalanceDest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]

def build_features(transaction):
    row = {}

    row["step"] = transaction.step
    row["amount"] = transaction.amount
    row["oldbalanceOrg"] = transaction.oldbalanceOrg
    row["oldbalanceDest"] = transaction.oldbalanceDest

    row["amount_to_oldbalanceOrg_ratio"] = (
        transaction.amount / max(transaction.oldbalanceOrg, 1)
    )

    row["amount_to_oldbalanceDest_ratio"] = (
        transaction.amount / max(transaction.oldbalanceDest, 1)
    )

    row["is_zero_oldbalanceOrg"] = int(transaction.oldbalanceOrg == 0)
    row["is_zero_oldbalanceDest"] = int(transaction.oldbalanceDest == 0)

    for t in [
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    ]:
        row[f"type_{t}"] = int(transaction.type == t)

    return pd.DataFrame([row])[FEATURE_COLUMNS]


def predict(transaction):
    X = build_features(transaction)

    probability = model.predict_proba(X)[0][1]

    prediction = int(probability >= 0.5)

    return {
        "fraud_probability": float(probability),
        "prediction": prediction,
    }

