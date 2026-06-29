from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw")
REPORT_DIR = Path("reports")
REPORT_PATH = REPORT_DIR / "data_validation_report.md"


REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def find_raw_csv() -> Path:
    csv_files = list(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {RAW_DIR}. "
            "Please place the PaySim CSV file under data/raw/."
        )

    if len(csv_files) > 1:
        print("Multiple CSV files found. Using the first one:")
        for file in csv_files:
            print(f"- {file}")

    return csv_files[0]


def validate_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def build_report(df: pd.DataFrame, csv_path: Path, missing_columns: list[str]) -> str:
    row_count, column_count = df.shape

    fraud_counts = df["isFraud"].value_counts(dropna=False).to_dict()
    flagged_counts = df["isFlaggedFraud"].value_counts(dropna=False).to_dict()
    transaction_types = df["type"].value_counts().to_dict()

    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0]

    report = []
    report.append("# Data Validation Report")
    report.append("")
    report.append("## Dataset")
    report.append("")
    report.append(f"- Raw file: `{csv_path}`")
    report.append(f"- Number of rows: {row_count:,}")
    report.append(f"- Number of columns: {column_count:,}")
    report.append("")
    report.append("## Required Columns")
    report.append("")

    if missing_columns:
        report.append("Missing required columns:")
        for col in missing_columns:
            report.append(f"- {col}")
    else:
        report.append("All required columns are present.")

    report.append("")
    report.append("## Fraud Label Distribution")
    report.append("")
    for label, count in fraud_counts.items():
        report.append(f"- isFraud = {label}: {count:,}")

    report.append("")
    report.append("## Flagged Fraud Distribution")
    report.append("")
    for label, count in flagged_counts.items():
        report.append(f"- isFlaggedFraud = {label}: {count:,}")

    report.append("")
    report.append("## Transaction Type Distribution")
    report.append("")
    for tx_type, count in transaction_types.items():
        report.append(f"- {tx_type}: {count:,}")

    report.append("")
    report.append("## Missing Values")
    report.append("")
    if len(missing_values) == 0:
        report.append("No missing values found.")
    else:
        for col, count in missing_values.items():
            report.append(f"- {col}: {count:,}")

    report.append("")
    report.append("## Preview")
    report.append("")
    report.append("```text")
    report.append(str(df.head()))
    report.append("```")
    report.append("")

    return "\n".join(report)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = find_raw_csv()
    print(f"Reading raw data from: {csv_path}")

    df = pd.read_csv(csv_path)

    print(f"Dataset shape: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
    print("Columns:")
    print(list(df.columns))

    missing_columns = validate_columns(df)

    if missing_columns:
        print("\nMissing required columns:")
        for col in missing_columns:
            print(f"- {col}")
    else:
        print("\nAll required columns are present.")

    print("\nFraud label distribution:")
    print(df["isFraud"].value_counts(dropna=False))

    print("\nTransaction type distribution:")
    print(df["type"].value_counts())

    report = build_report(df, csv_path, missing_columns)
    REPORT_PATH.write_text(report)

    print(f"\nValidation report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
