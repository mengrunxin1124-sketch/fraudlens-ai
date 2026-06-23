Research Proposal

Title

FraudLens AI: Studying Explanation Stability and Human-in-the-Loop Robustness in Financial Fraud Detection

1. Background

Financial fraud detection is a high-stakes machine learning application. In real-world financial institutions, machine learning models are often used to support analysts rather than fully replace human decision makers.

Many fraud detection projects focus only on classification performance, such as accuracy, precision, recall or AUC. However, real financial fraud detection systems also need to be explainable, robust and auditable.

There are two important challenges:

1. Fraud patterns may change over time.
2. Fraudsters may intentionally modify transaction behaviour to avoid detection.

These challenges can create distribution shift and adversarial manipulation. As a result, model performance may decrease, and model explanations may become unstable or misleading.

This project studies whether fraud detection explanations remain stable and useful under these challenging conditions.

2. Main Research Question

Are explanations of fraud detection models stable and actionable under temporal distribution shift and adversarial transaction manipulation?

3. Sub-Questions

This project will investigate the following sub-questions:

1. How does fraud detection performance change under temporal distribution shift?
2. How stable are SHAP feature rankings when transaction behaviour changes?
3. Can uncertainty-aware review reduce false positives under a fixed analyst review budget?
4. Do explanation-based summaries help analysts make better decisions?
5. Can graph-based features improve fraud-ring detection?

4. Dataset

The first version of this project will use the PaySim synthetic financial transaction dataset.

PaySim provides labelled synthetic transaction records for fraud detection experimentation. It is suitable for building an initial prototype because it contains transaction-level features and fraud labels.

However, PaySim is synthetic and simplified. Therefore, this project will clearly document its limitations and avoid claiming that the results directly represent real banking systems.

5. Baseline Models

The initial experiments will compare several baseline models:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* Isolation Forest
* graph-feature-enhanced model

The first supervised baseline will likely use XGBoost or LightGBM because these models are strong tabular machine learning methods and commonly used in fraud detection tasks.

6. Experimental Design

Experiment 1: Baseline Fraud Detection

This experiment trains fraud detection models using a temporal train/test split.

The goal is to establish a strong baseline.

Evaluation metrics:

* ROC-AUC
* PR-AUC
* precision
* recall
* F1-score
* Recall@Top-K
* false-positive cost

Experiment 2: Temporal Distribution Shift

This experiment evaluates whether model performance decreases when tested on later transaction periods.

The dataset will be split by time order instead of random split. Earlier transactions will be used for training, and later transactions will be used for testing.

Evaluation metrics:

* performance drop
* recall drop
* precision drop
* calibration error
* false-positive rate
* recall under fixed review budget

Experiment 3: Adversarial Transaction Manipulation

This experiment simulates simple fraudster behaviour changes.

Possible manipulations include:

* reducing transaction amount
* splitting one large transaction into multiple smaller transactions
* changing transaction frequency
* modifying balance-related features
* reducing obvious rule-triggering behaviours

The goal is to test whether the model and explanation remain reliable when suspicious behaviour is slightly modified.

Evaluation metrics:

* risk score change
* prediction flip rate
* performance degradation
* uncertainty change
* top SHAP feature change

Experiment 4: Explanation Stability

This experiment studies whether SHAP explanations remain stable under distribution shift or adversarial manipulation.

Evaluation metrics:

* top-k feature ranking overlap
* Spearman rank correlation
* explanation sign consistency
* explanation drift score
* change in top contributing factors

A stable explanation should not change dramatically when the underlying risk pattern is still similar.

Experiment 5: Human-in-the-Loop Review

This experiment simulates analyst review under limited review capacity.

Different review strategies will be compared:

* model-only decision
* risk-threshold-based review
* uncertainty-aware review
* explanation-assisted review

Evaluation metrics:

* fraud caught under fixed review budget
* false-positive workload
* decision cost
* review efficiency
* number of high-risk cases escalated

7. Expected Contribution

This project aims to contribute:

1. A full-stack fraud investigation platform prototype
2. A reproducible fraud detection experiment pipeline
3. A practical study of explanation stability under shift
4. A human-in-the-loop evaluation design
5. A portfolio-quality AI engineering project
6. A research-ready technical report

8. Practical Value

This project is relevant to several career and research directions:

* machine learning engineering
* financial risk analytics
* fraud detection
* responsible AI
* MLOps
* explainable AI
* human-AI decision making
* AI governance

It is designed to be useful for both job applications and future PhD research preparation.

9. Limitations

The first version uses synthetic data, so the results cannot be directly generalised to real financial institutions.

Other limitations include:

* limited realism of PaySim data
* lack of real customer behaviour data
* simplified fraud labels
* simplified adversarial manipulation
* simulated analyst workflow instead of real human study

Future work may include more realistic datasets, graph-based fraud-ring detection, stronger adversarial simulation and user studies with human analysts.

10. Summary

FraudLens AI studies fraud detection as a responsible AI decision-support problem.

The main focus is not only whether the model can detect fraud, but also whether the model explanation is stable, whether uncertainty can support human review, and whether the system can help analysts investigate suspicious financial behaviour.