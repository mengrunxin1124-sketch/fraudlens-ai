from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


GOLD_DIR = Path("data/gold")
REPORT_DIR = Path("reports")
DOCS_DIR = Path("docs")
MODEL_DIR = Path("models")

TRAIN_PATH = GOLD_DIR / "train_strict_features.csv"
VALIDATION_PATH = GOLD_DIR / "validation_strict_features.csv"
TEST_PATH = GOLD_DIR / "test_strict_features.csv"

METRICS_PATH = REPORT_DIR / "strict_baseline_metrics.json"
RESULTS_PATH = DOCS_DIR / "strict_baseline_results.md"

TARGET_COLUMN = "isFraud"
RANDOM_STATE = 42


def check_input_files() -> None:
    required_files = [TRAIN_PATH, VALIDATION_PATH, TEST_PATH]
    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Missing strict feature files. Please run `python src/features/build_strict_features.py` first.\n"
            f"{missing}"
        )


def split_x_y(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def make_training_sample(train_df: pd.DataFrame, normal_per_fraud: int = 50) -> pd.DataFrame:
    fraud_df = train_df[train_df[TARGET_COLUMN] == 1]
    normal_df = train_df[train_df[TARGET_COLUMN] == 0]

    n_normal = min(len(normal_df), len(fraud_df) * normal_per_fraud)

    sampled_normal_df = normal_df.sample(
        n=n_normal,
        random_state=RANDOM_STATE,
    )

    sampled_df = pd.concat([fraud_df, sampled_normal_df], axis=0)
    sampled_df = sampled_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    return sampled_df


def precision_recall_at_k(y_true: pd.Series, y_score: np.ndarray, k: int) -> dict:
    k = min(k, len(y_true))

    order = np.argsort(y_score)[::-1]
    top_k_idx = order[:k]

    y_top = np.asarray(y_true)[top_k_idx]

    fraud_found = int(y_top.sum())
    total_fraud = int(np.asarray(y_true).sum())

    precision_at_k = fraud_found / k if k > 0 else 0.0
    recall_at_k = fraud_found / total_fraud if total_fraud > 0 else 0.0

    return {
        "k": int(k),
        "fraud_found": fraud_found,
        "precision_at_k": float(precision_at_k),
        "recall_at_k": float(recall_at_k),
    }


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> dict:
    y_score = model.predict_proba(X)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "dataset": dataset_name,
        "rows": int(len(y)),
        "fraud_count": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, y_score)),
        "roc_auc": float(roc_auc_score(y, y_score)),
        "threshold_0_5": {
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "top_k": {
            "100": precision_recall_at_k(y, y_score, 100),
            "500": precision_recall_at_k(y, y_score, 500),
            "1000": precision_recall_at_k(y, y_score, 1000),
        },
    }


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=300,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    return model


def train_hist_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series):
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)
    return model


def build_markdown_report(metrics: dict) -> str:
    lines = []

    lines.append("# Strict Real-Time Baseline Results")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This report summarises the strict real-time baseline models for the PaySim fraud detection task."
    )
    lines.append("")
    lines.append(
        "Unlike the diagnostic baseline, this version removes potential leakage features such as rule-derived flags and post-transaction balance fields."
    )
    lines.append("")
    lines.append("## Training Setup")
    lines.append("")
    lines.append("- Dataset: PaySim synthetic financial transaction data")
    lines.append("- Split strategy: temporal train / validation / test split")
    lines.append("- Feature set: stricter real-time features only")
    lines.append("- Removed features: `isFlaggedFraud`, new balance fields, post-transaction balance error features")
    lines.append("- Training sample: all fraud cases from the training set plus sampled normal cases")
    lines.append("- Models: Logistic Regression and HistGradientBoostingClassifier")
    lines.append("")
    lines.append("## Model Performance Summary")
    lines.append("")
    lines.append("| Model | Dataset | Rows | Fraud Count | PR-AUC | ROC-AUC | Precision@1000 | Recall@1000 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    for model_name, model_metrics in metrics["models"].items():
        for dataset_name, result in model_metrics.items():
            top_1000 = result["top_k"]["1000"]
            lines.append(
                f"| {model_name} | {dataset_name} | "
                f"{result['rows']:,} | "
                f"{result['fraud_count']:,} | "
                f"{result['pr_auc']:.6f} | "
                f"{result['roc_auc']:.6f} | "
                f"{top_1000['precision_at_k']:.6f} | "
                f"{top_1000['recall_at_k']:.6f} |"
            )

    lines.append("")
    lines.append("## Confusion Matrix at Threshold 0.5")
    lines.append("")
    lines.append("| Model | Dataset | TN | FP | FN | TP | Precision | Recall | F1 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for model_name, model_metrics in metrics["models"].items():
        for dataset_name, result in model_metrics.items():
            cm = result["threshold_0_5"]
            lines.append(
                f"| {model_name} | {dataset_name} | "
                f"{cm['tn']:,} | {cm['fp']:,} | {cm['fn']:,} | {cm['tp']:,} | "
                f"{cm['precision']:.6f} | {cm['recall']:.6f} | {cm['f1']:.6f} |"
            )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This strict baseline is more suitable for real-time fraud detection discussion because it avoids direct use of post-transaction balance fields."
    )
    lines.append("")
    lines.append(
        "If performance drops compared with the diagnostic baseline, that is expected and useful. It shows how much the earlier result depended on potentially leaky features."
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `reports/strict_baseline_metrics.json`")
    lines.append("- `docs/strict_baseline_results.md`")
    lines.append("- `models/logistic_regression_strict_baseline.joblib`")
    lines.append("- `models/hist_gradient_boosting_strict_baseline.joblib`")
    lines.append("")
    lines.append("Model files are ignored by Git because they are generated artifacts.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    check_input_files()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading strict feature files...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Train shape:      {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape:       {test_df.shape}")

    sampled_train_df = make_training_sample(train_df, normal_per_fraud=50)
    print(f"\nSampled training shape: {sampled_train_df.shape}")
    print("Sampled training fraud distribution:")
    print(sampled_train_df[TARGET_COLUMN].value_counts())

    X_train, y_train = split_x_y(sampled_train_df)
    X_val, y_val = split_x_y(val_df)
    X_test, y_test = split_x_y(test_df)

    metrics = {
        "models": {},
        "metadata": {
            "target_column": TARGET_COLUMN,
            "random_state": RANDOM_STATE,
            "training_rows": int(len(sampled_train_df)),
            "training_fraud_count": int(y_train.sum()),
        },
    }

    print("\nTraining strict Logistic Regression baseline...")
    start_time = time.time()
    logistic_model = train_logistic_regression(X_train, y_train)
    logistic_time = time.time() - start_time
    print(f"Strict Logistic Regression training completed in {logistic_time:.2f} seconds.")

    metrics["models"]["strict_logistic_regression"] = {
        "validation": evaluate_model(logistic_model, X_val, y_val, "validation"),
        "test": evaluate_model(logistic_model, X_test, y_test, "test"),
    }

    joblib.dump(logistic_model, MODEL_DIR / "logistic_regression_strict_baseline.joblib")

    print("\nTraining strict HistGradientBoosting baseline...")
    start_time = time.time()
    hgb_model = train_hist_gradient_boosting(X_train, y_train)
    hgb_time = time.time() - start_time
    print(f"Strict HistGradientBoosting training completed in {hgb_time:.2f} seconds.")

    metrics["models"]["strict_hist_gradient_boosting"] = {
        "validation": evaluate_model(hgb_model, X_val, y_val, "validation"),
        "test": evaluate_model(hgb_model, X_test, y_test, "test"),
    }

    joblib.dump(hgb_model, MODEL_DIR / "hist_gradient_boosting_strict_baseline.joblib")

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    RESULTS_PATH.write_text(build_markdown_report(metrics))

    print("\nStrict baseline training completed.")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Markdown report saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
