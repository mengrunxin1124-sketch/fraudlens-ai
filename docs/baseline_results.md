# Baseline Model Results

## Purpose

This report summarises the first reproducible baseline models for the PaySim fraud detection task.

The baseline uses a temporal split instead of a random split to better simulate a real fraud detection setting.

## Training Setup

- Dataset: PaySim synthetic financial transaction data
- Split strategy: temporal train / validation / test split
- Feature set: numerical transaction features, transaction type one-hot encoding, balance error features and ratio features
- Training sample: all fraud cases from the training set plus sampled normal cases
- Models: Logistic Regression and HistGradientBoostingClassifier

## Model Performance Summary

| Model | Dataset | Rows | Fraud Count | PR-AUC | ROC-AUC | Precision@1000 | Recall@1000 |
|---|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | validation | 954,393 | 564 | 0.830835 | 0.999653 | 0.480000 | 0.851064 |
| logistic_regression | test | 954,393 | 4,010 | 0.911287 | 0.997453 | 1.000000 | 0.249377 |
| hist_gradient_boosting | validation | 954,393 | 564 | 1.000000 | 1.000000 | 0.564000 | 1.000000 |
| hist_gradient_boosting | test | 954,393 | 4,010 | 0.999923 | 1.000000 | 1.000000 | 0.249377 |

## Confusion Matrix at Threshold 0.5

| Model | Dataset | TN | FP | FN | TP | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | validation | 943,782 | 10,047 | 0 | 564 | 0.053152 | 1.000000 | 0.100940 |
| logistic_regression | test | 943,131 | 7,252 | 150 | 3,860 | 0.347372 | 0.962594 | 0.510514 |
| hist_gradient_boosting | validation | 953,829 | 0 | 0 | 564 | 1.000000 | 1.000000 | 1.000000 |
| hist_gradient_boosting | test | 950,383 | 0 | 2 | 4,008 | 1.000000 | 0.999501 | 0.999751 |

## Interpretation

Because fraud cases are extremely rare, PR-AUC and Recall@K are more informative than accuracy.

The temporal split shows that the test set has a higher fraud rate than the training and validation sets, which suggests a useful setting for studying temporal distribution shift.

## Baseline Limitations

The baseline results should be interpreted carefully. PaySim is a synthetic dataset, and the current feature set includes post-transaction balance features such as `newbalanceOrig` and `newbalanceDest`. These features make the first baseline useful for validating the pipeline, but they may lead to overly optimistic performance compared with a real-time fraud detection setting.

Future versions will compare this baseline with a stricter pre-transaction feature setting and robustness experiments under temporal distribution shift.

## Output Files

- `reports/baseline_metrics.json`
- `docs/baseline_results.md`
- `models/logistic_regression_baseline.joblib`
- `models/hist_gradient_boosting_baseline.joblib`

Model files are ignored by Git because they are generated artifacts.
