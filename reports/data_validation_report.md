# Data Validation Report

## Dataset

- Raw file: `data/raw/PS_20174392719_1491204439457_log.csv`
- Number of rows: 6,362,620
- Number of columns: 11

## Required Columns

All required columns are present.

## Fraud Label Distribution

- isFraud = 0: 6,354,407
- isFraud = 1: 8,213

## Flagged Fraud Distribution

- isFlaggedFraud = 0: 6,362,604
- isFlaggedFraud = 1: 16

## Transaction Type Distribution

- CASH_OUT: 2,237,500
- PAYMENT: 2,151,495
- CASH_IN: 1,399,284
- TRANSFER: 532,909
- DEBIT: 41,432

## Missing Values

No missing values found.

## Preview

```text
   step      type    amount  ... newbalanceDest  isFraud  isFlaggedFraud
0     1   PAYMENT   9839.64  ...            0.0        0               0
1     1   PAYMENT   1864.28  ...            0.0        0               0
2     1  TRANSFER    181.00  ...            0.0        1               0
3     1  CASH_OUT    181.00  ...            0.0        1               0
4     1   PAYMENT  11668.14  ...            0.0        0               0

[5 rows x 11 columns]
```
