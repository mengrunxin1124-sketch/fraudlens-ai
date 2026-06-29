# Temporal Split Report

## Purpose

This report documents the temporal train/validation/test split for the PaySim dataset.

A temporal split is used instead of a random split to better simulate a real fraud detection setting, where models are trained on past transactions and evaluated on future transactions.

## Split Strategy

- Train: earliest 70% of transactions by `step` order
- Validation: next 15% of transactions by `step` order
- Test: latest 15% of transactions by `step` order

## Split Summary

| Split | Rows | Step Range | Fraud Count | Fraud Rate |
|---|---:|---|---:|---:|
| train | 4,453,834 | 1 - 323 | 3,639 | 0.000817 |
| validation | 954,393 | 323 - 378 | 564 | 0.000591 |
| test | 954,393 | 378 - 743 | 4,010 | 0.004202 |

## Output Files

- `data/gold/train.csv`
- `data/gold/validation.csv`
- `data/gold/test.csv`

## Note

The generated CSV files are stored under `data/gold/` and are ignored by Git to avoid uploading large data files.
