# Strict Real-Time Feature Engineering Report

## Purpose

This report documents the strict feature engineering pipeline for a more realistic real-time fraud detection baseline.

## Difference from Diagnostic Baseline

The earlier diagnostic baseline used post-transaction balance fields and rule-derived variables to validate the full pipeline.

This strict baseline removes potential leakage features, including:

- `isFlaggedFraud`
- `newbalanceOrig`
- `newbalanceDest`
- post-transaction balance change features
- post-transaction balance error features
- zero indicators based on new balances

## Retained Feature Groups

- transaction step
- transaction amount
- transaction type one-hot encoding
- pre-transaction origin balance
- pre-transaction destination balance
- amount-to-pre-transaction-balance ratios
- zero indicators based on pre-transaction balances

## Output Summary

| Dataset | Rows | Columns | Fraud Count | Fraud Rate |
|---|---:|---:|---:|---:|
| train_strict_features | 4,453,834 | 14 | 3,639 | 0.000817 |
| validation_strict_features | 954,393 | 14 | 564 | 0.000591 |
| test_strict_features | 954,393 | 14 | 4,010 | 0.004202 |

## Feature Columns

- step
- amount
- oldbalanceOrg
- oldbalanceDest
- amount_to_oldbalanceOrg_ratio
- amount_to_oldbalanceDest_ratio
- is_zero_oldbalanceOrg
- is_zero_oldbalanceDest
- type_CASH_IN
- type_CASH_OUT
- type_DEBIT
- type_PAYMENT
- type_TRANSFER

## Target Column

- `isFraud`

## Output Files

- `data/gold/train_strict_features.csv`
- `data/gold/validation_strict_features.csv`
- `data/gold/test_strict_features.csv`

Generated strict feature files are stored under `data/gold/` and ignored by Git.
