from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap


GOLD_DIR = Path("data/gold")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"
DOCS_DIR = Path("docs")

TEST_PATH = GOLD_DIR / "test_strict_features.csv"
TRAIN_PATH = GOLD_DIR / "train_strict_features.csv"

HGB_MODEL_PATH = MODEL_DIR / "hist_gradient_boosting_strict_baseline.joblib"
LR_MODEL_PATH = MODEL_DIR / "logistic_regression_strict_baseline.joblib"

SHAP_SUMMARY_PATH = REPORT_DIR / "shap_summary.csv"
SHAP_FIGURE_PATH = FIGURE_DIR / "shap_summary.png"
REPORT_PATH = DOCS_DIR / "explainability_report.md"

TARGET_COLUMN = "isFraud"
RANDOM_STATE = 42


def check_files() -> None:
    required = [
        TRAIN_PATH,
        TEST_PATH,
        HGB_MODEL_PATH,
        LR_MODEL_PATH,
    ]

    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing required files. Please run strict feature engineering and strict baseline training first.\n"
            f"{missing_text}"
        )


def split_x_y(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def make_explain_sample(df: pd.DataFrame, max_fraud: int = 1000, max_normal: int = 1000) -> pd.DataFrame:
    fraud_df = df[df[TARGET_COLUMN] == 1]
    normal_df = df[df[TARGET_COLUMN] == 0]

    fraud_sample = fraud_df.sample(
        n=min(max_fraud, len(fraud_df)),
        random_state=RANDOM_STATE,
    )

    normal_sample = normal_df.sample(
        n=min(max_normal, len(normal_df)),
        random_state=RANDOM_STATE,
    )

    sample_df = pd.concat([fraud_sample, normal_sample], axis=0)
    sample_df = sample_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    return sample_df


def get_lr_importance(lr_pipeline, feature_names: list[str]) -> pd.DataFrame:
    classifier = lr_pipeline.named_steps["classifier"]
    coef = classifier.coef_[0]

    lr_df = pd.DataFrame(
        {
            "feature": feature_names,
            "lr_coefficient": coef,
            "lr_abs_coefficient": np.abs(coef),
        }
    )

    lr_df["lr_rank"] = lr_df["lr_abs_coefficient"].rank(
        ascending=False,
        method="dense",
    ).astype(int)

    return lr_df


def get_hgb_shap_importance(hgb_model, X_explain: pd.DataFrame) -> pd.DataFrame:
    print("Building SHAP TreeExplainer for HistGradientBoosting model...")

    explainer = shap.TreeExplainer(hgb_model)
    shap_values = explainer.shap_values(X_explain)

    if isinstance(shap_values, list):
        shap_array = shap_values[1]
    else:
        shap_array = shap_values

    if len(shap_array.shape) == 3:
        shap_array = shap_array[:, :, 1]

    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    mean_shap = shap_array.mean(axis=0)

    shap_df = pd.DataFrame(
        {
            "feature": X_explain.columns,
            "hgb_mean_abs_shap": mean_abs_shap,
            "hgb_mean_shap": mean_shap,
        }
    )

    shap_df["hgb_rank"] = shap_df["hgb_mean_abs_shap"].rank(
        ascending=False,
        method="dense",
    ).astype(int)

    return shap_df.sort_values("hgb_mean_abs_shap", ascending=False)


def plot_shap_summary(summary_df: pd.DataFrame, top_n: int = 15) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    top_df = summary_df.sort_values("hgb_mean_abs_shap", ascending=True).tail(top_n)

    plt.figure(figsize=(10, 7))
    plt.barh(top_df["feature"], top_df["hgb_mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title("Strict Baseline SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(SHAP_FIGURE_PATH, dpi=200)
    plt.close()


def build_report(summary_df: pd.DataFrame, sample_size: int, fraud_count: int) -> str:
    top_hgb = summary_df.sort_values("hgb_mean_abs_shap", ascending=False).head(10)
    top_lr = summary_df.sort_values("lr_abs_coefficient", ascending=False).head(10)

    hgb_top5 = set(top_hgb.head(5)["feature"])
    lr_top5 = set(top_lr.head(5)["feature"])
    overlap = sorted(hgb_top5.intersection(lr_top5))

    lines = []

    lines.append("# Explainability Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This report explains the strict real-time fraud detection baseline using SHAP feature importance for the HistGradientBoosting model and coefficient importance for Logistic Regression."
    )
    lines.append("")
    lines.append("The goal is to understand which features most influence fraud prediction and whether the model explanations appear reasonable.")
    lines.append("")
    lines.append("## Explanation Sample")
    lines.append("")
    lines.append(f"- Sample size used for explanation: {sample_size:,}")
    lines.append(f"- Fraud cases in explanation sample: {fraud_count:,}")
    lines.append(f"- Non-fraud cases in explanation sample: {sample_size - fraud_count:,}")
    lines.append("")
    lines.append("## Top HistGradientBoosting SHAP Features")
    lines.append("")
    lines.append("| Rank | Feature | Mean Absolute SHAP | Mean SHAP |")
    lines.append("|---:|---|---:|---:|")

    for _, row in top_hgb.iterrows():
        lines.append(
            f"| {int(row['hgb_rank'])} | {row['feature']} | "
            f"{row['hgb_mean_abs_shap']:.6f} | {row['hgb_mean_shap']:.6f} |"
        )

    lines.append("")
    lines.append("## Top Logistic Regression Coefficient Features")
    lines.append("")
    lines.append("| Rank | Feature | Absolute Coefficient | Coefficient |")
    lines.append("|---:|---|---:|---:|")

    for _, row in top_lr.iterrows():
        lines.append(
            f"| {int(row['lr_rank'])} | {row['feature']} | "
            f"{row['lr_abs_coefficient']:.6f} | {row['lr_coefficient']:.6f} |"
        )

    lines.append("")
    lines.append("## Explanation Consistency")
    lines.append("")

    if overlap:
        lines.append(
            "The two models share the following top-5 important features:"
        )
        lines.append("")
        for feature in overlap:
            lines.append(f"- {feature}")
    else:
        lines.append(
            "The two models do not share top-5 features, which suggests that the linear and non-linear models may rely on different decision patterns."
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "If features such as `amount`, `oldbalanceOrg`, `oldbalanceDest`, transaction type indicators, or amount-to-balance ratios appear near the top, the explanation is broadly reasonable for fraud detection because transaction size, account balance and transaction type are important risk signals."
    )
    lines.append("")
    lines.append(
        "However, if the model relies heavily on a very small number of balance-related variables, this may indicate that PaySim contains synthetic patterns that are easier to learn than real-world fraud behaviour."
    )
    lines.append("")
    lines.append("## Synthetic Dataset Caveat")
    lines.append("")
    lines.append(
        "The strict baseline removes obvious leakage features such as `isFlaggedFraud` and post-transaction balance fields. However, PaySim is still a synthetic dataset, so strong model performance and strong feature importance patterns should be interpreted carefully."
    )
    lines.append("")
    lines.append(
        "These explanations motivate future ablation studies and temporal drift analysis to test whether the model depends too heavily on simple synthetic transaction patterns."
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `reports/shap_summary.csv`")
    lines.append("- `reports/figures/shap_summary.png`")
    lines.append("- `docs/explainability_report.md`")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    check_files()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading strict feature data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    explain_df = make_explain_sample(test_df)
    X_explain, y_explain = split_x_y(explain_df)

    print(f"Explanation sample shape: {X_explain.shape}")
    print("Explanation sample fraud distribution:")
    print(y_explain.value_counts())

    print("Loading strict baseline models...")
    hgb_model = joblib.load(HGB_MODEL_PATH)
    lr_model = joblib.load(LR_MODEL_PATH)

    feature_names = list(X_explain.columns)

    print("Computing Logistic Regression coefficient importance...")
    lr_df = get_lr_importance(lr_model, feature_names)

    print("Computing SHAP importance for HistGradientBoosting...")
    hgb_df = get_hgb_shap_importance(hgb_model, X_explain)

    summary_df = hgb_df.merge(lr_df, on="feature", how="left")
    summary_df = summary_df.sort_values("hgb_mean_abs_shap", ascending=False)

    summary_df.to_csv(SHAP_SUMMARY_PATH, index=False)
    plot_shap_summary(summary_df)

    report = build_report(
        summary_df=summary_df,
        sample_size=len(explain_df),
        fraud_count=int(y_explain.sum()),
    )
    REPORT_PATH.write_text(report)

    print("\nExplainability analysis completed.")
    print(f"SHAP summary saved to: {SHAP_SUMMARY_PATH}")
    print(f"SHAP figure saved to: {SHAP_FIGURE_PATH}")
    print(f"Explainability report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
