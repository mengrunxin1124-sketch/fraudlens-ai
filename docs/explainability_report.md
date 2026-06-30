# Explainability Report

## Purpose

This report explains the strict real-time fraud detection baseline using SHAP feature importance for the HistGradientBoosting model and coefficient importance for Logistic Regression.

The goal is to understand which features most influence fraud prediction and whether the model explanations appear reasonable.

## Explanation Sample

- Sample size used for explanation: 2,000
- Fraud cases in explanation sample: 1,000
- Non-fraud cases in explanation sample: 1,000

## Top HistGradientBoosting SHAP Features

| Rank | Feature | Mean Absolute SHAP | Mean SHAP |
|---:|---|---:|---:|
| 1 | amount_to_oldbalanceOrg_ratio | 7.184185 | 6.547376 |
| 2 | amount_to_oldbalanceDest_ratio | 0.717954 | 0.606443 |
| 3 | amount | 0.544827 | 0.507519 |
| 4 | oldbalanceOrg | 0.371772 | 0.303630 |
| 5 | type_PAYMENT | 0.219848 | 0.132471 |
| 6 | type_CASH_OUT | 0.154896 | -0.012292 |
| 7 | step | 0.108972 | -0.108398 |
| 8 | type_CASH_IN | 0.090146 | 0.064727 |
| 9 | type_TRANSFER | 0.082721 | 0.056861 |
| 10 | oldbalanceDest | 0.026944 | 0.015585 |

## Top Logistic Regression Coefficient Features

| Rank | Feature | Absolute Coefficient | Coefficient |
|---:|---|---:|---:|
| 1 | type_CASH_IN | 42.894767 | -42.894767 |
| 2 | type_CASH_OUT | 37.835104 | 37.835104 |
| 3 | type_TRANSFER | 22.945191 | 22.945191 |
| 4 | oldbalanceOrg | 18.228145 | 18.228145 |
| 5 | type_PAYMENT | 14.564084 | -14.564084 |
| 6 | type_DEBIT | 2.593432 | -2.593432 |
| 7 | amount | 2.232651 | -2.232651 |
| 8 | amount_to_oldbalanceOrg_ratio | 1.650970 | 1.650970 |
| 9 | is_zero_oldbalanceOrg | 1.649644 | -1.649644 |
| 10 | is_zero_oldbalanceDest | 0.546654 | 0.546654 |

## Explanation Consistency

The two models share the following top-5 important features:

- oldbalanceOrg
- type_PAYMENT

## Interpretation

If features such as `amount`, `oldbalanceOrg`, `oldbalanceDest`, transaction type indicators, or amount-to-balance ratios appear near the top, the explanation is broadly reasonable for fraud detection because transaction size, account balance and transaction type are important risk signals.

However, if the model relies heavily on a very small number of balance-related variables, this may indicate that PaySim contains synthetic patterns that are easier to learn than real-world fraud behaviour.

## Synthetic Dataset Caveat

The strict baseline removes obvious leakage features such as `isFlaggedFraud` and post-transaction balance fields. However, PaySim is still a synthetic dataset, so strong model performance and strong feature importance patterns should be interpreted carefully.

These explanations motivate future ablation studies and temporal drift analysis to test whether the model depends too heavily on simple synthetic transaction patterns.

## Output Files

- `reports/shap_summary.csv`
- `reports/figures/shap_summary.png`
- `docs/explainability_report.md`
