# Strict Real-Time Baseline Results

## Purpose

This report summarises the strict real-time baseline models for the PaySim fraud detection task.

Unlike the diagnostic baseline, this version removes potential leakage features such as rule-derived flags and post-transaction balance fields.

## Training Setup

- Dataset: PaySim synthetic financial transaction data
- Split strategy: temporal train / validation / test split
- Feature set: stricter real-time features only
- Removed features: `isFlaggedFraud`, new balance fields, post-transaction balance error features
- Training sample: all fraud cases from the training set plus sampled normal cases
- Models: Logistic Regression and HistGradientBoostingClassifier

## Model Performance Summary

| Model | Dataset | Rows | Fraud Count | PR-AUC | ROC-AUC | Precision@1000 | Recall@1000 |
|---|---|---:|---:|---:|---:|---:|---:|
| strict_logistic_regression | validation | 954,393 | 564 | 0.125641 | 0.983209 | 0.135000 | 0.239362 |
| strict_logistic_regression | test | 954,393 | 4,010 | 0.367487 | 0.974368 | 0.796000 | 0.198504 |
| strict_hist_gradient_boosting | validation | 954,393 | 564 | 0.996861 | 0.999998 | 0.564000 | 1.000000 |
| strict_hist_gradient_boosting | test | 954,393 | 4,010 | 0.999410 | 0.999998 | 1.000000 | 0.249377 |

## Confusion Matrix at Threshold 0.5

| Model | Dataset | TN | FP | FN | TP | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| strict_logistic_regression | validation | 888,031 | 65,798 | 52 | 512 | 0.007721 | 0.907801 | 0.015312 |
| strict_logistic_regression | test | 892,251 | 58,132 | 624 | 3,386 | 0.055041 | 0.844389 | 0.103345 |
| strict_hist_gradient_boosting | validation | 953,776 | 53 | 0 | 564 | 0.914100 | 1.000000 | 0.955123 |
| strict_hist_gradient_boosting | test | 950,314 | 69 | 5 | 4,005 | 0.983063 | 0.998753 | 0.990846 |

## Interpretation

This strict baseline is more suitable for real-time fraud detection discussion because it avoids direct use of post-transaction balance fields.

If performance drops compared with the diagnostic baseline, that is expected and useful. It shows how much the earlier result depended on potentially leaky features.

## Output Files

- `reports/strict_baseline_metrics.json`
- `docs/strict_baseline_results.md`
- `models/logistic_regression_strict_baseline.joblib`
- `models/hist_gradient_boosting_strict_baseline.joblib`

Model files are ignored by Git because they are generated artifacts.
