# Data

This project will initially use the PaySim synthetic financial transaction dataset.

PaySim is a synthetic mobile money transaction dataset designed for fraud detection research and experimentation.

## Dataset Source

Dataset name: PaySim Synthetic Financial Datasets for Fraud Detection

Source: Kaggle

The raw dataset should be downloaded manually from Kaggle.

## Important Note

The raw dataset is not stored in this GitHub repository.

Please download the dataset manually and place it under:

```text
data/raw/
```

Large data files should not be committed to GitHub.

## Planned Data Structure

```text
data/
├── README.md
├── raw/       # Original downloaded dataset
├── bronze/    # Schema-validated data
├── silver/    # Cleaned and feature-enriched data
└── gold/      # Model-ready data
```

## Data Layers

### Raw Layer

The raw layer stores the original downloaded dataset without modification.

### Bronze Layer

The bronze layer stores schema-validated transaction data.

Planned checks:

- column name validation
- data type validation
- missing value check
- duplicate row check
- basic range check

### Silver Layer

The silver layer stores cleaned and feature-enriched data.

Planned features:

- transaction amount features
- balance difference features
- account-level behaviour features
- time-window transaction frequency features
- risk indicator features

### Gold Layer

The gold layer stores model-ready data for training, validation and testing.

Planned files:

- training dataset
- validation dataset
- test dataset
- inference sample dataset

## Data Governance

Future versions will include:

- data card
- schema validation report
- data quality checks
- feature documentation
- drift monitoring
- feature attribution drift monitoring

## Dataset Limitation

PaySim is synthetic data. It is useful for building a prototype and running controlled experiments, but it cannot fully represent real financial systems or real customer behaviour.

All results should be interpreted as prototype-level findings.
