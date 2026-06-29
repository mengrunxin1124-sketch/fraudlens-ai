from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw")
GOLD_DIR = Path("data/gold")
REPORT_DIR = Path("reports")
REPORT_PATH = REPORT_DIR / "split_report.md"

TRAIN_PATH = GOLD_DIR / "train.csv"
VALIDATION_PATH = GOLD_DIR / "validation.csv"
TEST_PATH = GOLD_DIR / "test.csv"


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


def summarize_split(name: str, df: pd.DataFrame) -> dict:
    return {
        "name": name,
        "rows": len(df),
        "min_step": int(df["step"].min()),
        "max_step": int(df["step"].max()),
        "fraud_count": int(df["isFraud"].sum()),
        "fraud_rate": float(df["isFraud"].mean()),
    }


def build_report(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    summaries = [
        summarize_split("train", train_df),
        summarize_split("validation", val_df),
        summarize_split("test", test_df),
    ]

    report = []
    report.append("# Temporal Split Report")
    report.append("")
    report.append("## Purpose")
    report.append("")
    report.append(
        "This report documents the temporal train/validation/test split for the PaySim dataset."
    )
    report.append("")
    report.append(
        "A temporal split is used instead of a random split to better simulate a real fraud detection setting, "
        "where models are trained on past transactions and evaluated on future transactions."
    )
    report.append("")
    report.append("## Split Strategy")
    report.append("")
    report.append("- Train: earliest 70% of transactions by `step` order")
    report.append("- Validation: next 15% of transactions by `step` order")
    report.append("- Test: latest 15% of transactions by `step` order")
    report.append("")
    report.append("## Split Summary")
    report.append("")
    report.append("| Split | Rows | Step Range | Fraud Count | Fraud Rate |")
    report.append("|---|---:|---|---:|---:|")

    for item in summaries:
        report.append(
            f"| {item['name']} | {item['rows']:,} | "
            f"{item['min_step']} - {item['max_step']} | "
            f"{item['fraud_count']:,} | {item['fraud_rate']:.6f} |"
        )

    report.append("")
    report.append("## Output Files")
    report.append("")
    report.append(f"- `{TRAIN_PATH}`")
    report.append(f"- `{VALIDATION_PATH}`")
    report.append(f"- `{TEST_PATH}`")
    report.append("")
    report.append("## Note")
    report.append("")
    report.append(
        "The generated CSV files are stored under `data/gold/` and are ignored by Git to avoid uploading large data files."
    )
    report.append("")

    return "\n".join(report)


def main() -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = find_raw_csv()
    print(f"Reading raw data from: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Original shape: {df.shape[0]:,} rows, {df.shape[1]:,} columns")

    if "step" not in df.columns:
        raise ValueError("Column `step` is required for temporal split.")

    if "isFraud" not in df.columns:
        raise ValueError("Column `isFraud` is required for fraud label summary.")

    df = df.sort_values("step").reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VALIDATION_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("\nTemporal split completed.")
    print(f"Train:      {len(train_df):,} rows | step {train_df['step'].min()} - {train_df['step'].max()} | fraud {train_df['isFraud'].sum():,}")
    print(f"Validation: {len(val_df):,} rows | step {val_df['step'].min()} - {val_df['step'].max()} | fraud {val_df['isFraud'].sum():,}")
    print(f"Test:       {len(test_df):,} rows | step {test_df['step'].min()} - {test_df['step'].max()} | fraud {test_df['isFraud'].sum():,}")

    report = build_report(train_df, val_df, test_df)
    REPORT_PATH.write_text(report)

    print(f"\nSplit files saved to: {GOLD_DIR}")
    print(f"Split report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
