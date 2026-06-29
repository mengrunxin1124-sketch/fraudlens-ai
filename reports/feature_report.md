# Feature Engineering Report

## Purpose

This report documents the first version of the feature engineering pipeline for the PaySim baseline model.

## Feature Strategy

The first baseline uses simple tabular features:

- original numerical transaction features
- transaction type one-hot encoding
- balance change features
- balance error features
- amount-to-balance ratio features
- zero-balance indicator features

Account IDs are removed in the baseline version to avoid overfitting to synthetic identifiers.

## Output Summary

| Dataset | Rows | Columns | Fraud Count | Fraud Rate |
|---|---:|---:|---:|---:|
| train_features | 4,453,834 | 23 | 3,639 | 0.000817 |
| validation_features | 954,393 | 23 | 564 | 0.000591 |
| test_features | 954,393 | 23 | 4,010 | 0.004202 |

## Feature Columns

- step
- amount
- oldbalanceOrg
- newbalanceOrig
- oldbalanceDest
- newbalanceDest
- isFlaggedFraud
- origin_balance_change
- destination_balance_change
- origin_balance_error
- destination_balance_error
- amount_to_oldbalanceOrg_ratio
- amount_to_oldbalanceDest_ratio
- is_zero_oldbalanceOrg
- is_zero_newbalanceOrig
- is_zero_oldbalanceDest
- is_zero_newbalanceDest
- type_CASH_IN
- type_CASH_OUT
- type_DEBIT
- type_PAYMENT
- type_TRANSFER

## Target Column

- `isFraud`

## Output Files

- `data/gold/train_features.csv`
- `data/gold/validation_features.csv`
- `data/gold/test_features.csv`

## Note

Generated feature CSV files are stored under `data/gold/` and are ignored by Git because they are derived data files.
