from pathlib import Path
import pandas as pd
import numpy as np


GOLD_DIR = Path("data/gold")
REPORT_DIR = Path("reports")
REPORT_PATH = REPORT_DIR / "feature_report.md"

INPUT_FILES = {
    "train": GOLD_DIR / "train.csv",
    "validation": GOLD_DIR / "validation.csv",
    "test": GOLD_DIR / "test.csv",
}

OUTPUT_FILES = {
    "train": GOLD_DIR / "train_features.csv",
    "validation": GOLD_DIR / "validation_features.csv",
    "test": GOLD_DIR / "test_features.csv",
}

TARGET_COLUMN = "isFraud"

ID_COLUMNS = [
    "nameOrig",
    "nameDest",
]

CATEGORICAL_COLUMNS = [
    "type",
]

NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
]


def check_input_files() -> None:
    missing_files = [path for path in INPUT_FILES.values() if not path.exists()]
    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Missing split files. Please run `python src/data/split_data.py` first.\n"
            f"{missing}"
        )


def add_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["origin_balance_change"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["destination_balance_change"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df["origin_balance_error"] = df["origin_balance_change"] - df["amount"]
    df["destination_balance_error"] = df["destination_balance_change"] - df["amount"]

    df["amount_to_oldbalanceOrg_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
    df["amount_to_oldbalanceDest_ratio"] = df["amount"] / (df["oldbalanceDest"] + 1.0)

    df["is_zero_oldbalanceOrg"] = (df["oldbalanceOrg"] == 0).astype(int)
    df["is_zero_newbalanceOrig"] = (df["newbalanceOrig"] == 0).astype(int)
    df["is_zero_oldbalanceDest"] = (df["oldbalanceDest"] == 0).astype(int)
    df["is_zero_newbalanceDest"] = (df["newbalanceDest"] == 0).astype(int)

    return df


def encode_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    type_dummies = pd.get_dummies(df["type"], prefix="type", dtype=int)
    df = pd.concat([df, type_dummies], axis=1)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = add_balance_features(df)
    df = encode_transaction_type(df)

    drop_columns = ID_COLUMNS + CATEGORICAL_COLUMNS
    df = df.drop(columns=drop_columns)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def align_columns(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    train_columns = train_df.columns

    val_df = val_df.reindex(columns=train_columns, fill_value=0)
    test_df = test_df.reindex(columns=train_columns, fill_value=0)

    return train_df, val_df, test_df


def summarize_features(name: str, df: pd.DataFrame) -> dict:
    return {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "fraud_count": int(df[TARGET_COLUMN].sum()),
        "fraud_rate": float(df[TARGET_COLUMN].mean()),
    }


def build_report(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    summaries = [
        summarize_features("train_features", train_df),
        summarize_features("validation_features", val_df),
        summarize_features("test_features", test_df),
    ]

    feature_columns = [col for col in train_df.columns if col != TARGET_COLUMN]

    report = []
    report.append("# Feature Engineering Report")
    report.append("")
    report.append("## Purpose")
    report.append("")
    report.append(
        "This report documents the first version of the feature engineering pipeline for the PaySim baseline model."
    )
    report.append("")
    report.append("## Feature Strategy")
    report.append("")
    report.append("The first baseline uses simple tabular features:")
    report.append("")
    report.append("- original numerical transaction features")
    report.append("- transaction type one-hot encoding")
    report.append("- balance change features")
    report.append("- balance error features")
    report.append("- amount-to-balance ratio features")
    report.append("- zero-balance indicator features")
    report.append("")
    report.append("Account IDs are removed in the baseline version to avoid overfitting to synthetic identifiers.")
    report.append("")
    report.append("## Output Summary")
    report.append("")
    report.append("| Dataset | Rows | Columns | Fraud Count | Fraud Rate |")
    report.append("|---|---:|---:|---:|---:|")

    for item in summaries:
        report.append(
            f"| {item['name']} | {item['rows']:,} | {item['columns']:,} | "
            f"{item['fraud_count']:,} | {item['fraud_rate']:.6f} |"
        )

    report.append("")
    report.append("## Feature Columns")
    report.append("")
    for col in feature_columns:
        report.append(f"- {col}")

    report.append("")
    report.append("## Target Column")
    report.append("")
    report.append(f"- `{TARGET_COLUMN}`")
    report.append("")
    report.append("## Output Files")
    report.append("")
    for path in OUTPUT_FILES.values():
        report.append(f"- `{path}`")
    report.append("")
    report.append("## Note")
    report.append("")
    report.append(
        "Generated feature CSV files are stored under `data/gold/` and are ignored by Git because they are derived data files."
    )
    report.append("")

    return "\n".join(report)


def main() -> None:
    check_input_files()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading split files...")
    train_raw = pd.read_csv(INPUT_FILES["train"])
    val_raw = pd.read_csv(INPUT_FILES["validation"])
    test_raw = pd.read_csv(INPUT_FILES["test"])

    print("Building features...")
    train_features = build_features(train_raw)
    val_features = build_features(val_raw)
    test_features = build_features(test_raw)

    train_features, val_features, test_features = align_columns(
        train_features,
        val_features,
        test_features,
    )

    print("Saving feature files...")
    train_features.to_csv(OUTPUT_FILES["train"], index=False)
    val_features.to_csv(OUTPUT_FILES["validation"], index=False)
    test_features.to_csv(OUTPUT_FILES["test"], index=False)

    report = build_report(train_features, val_features, test_features)
    REPORT_PATH.write_text(report)

    print("\nFeature engineering completed.")
    print(f"Train features:      {train_features.shape}")
    print(f"Validation features: {val_features.shape}")
    print(f"Test features:       {test_features.shape}")
    print(f"Feature report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
