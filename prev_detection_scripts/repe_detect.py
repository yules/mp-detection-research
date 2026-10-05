import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import roc_auc_score, classification_report

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lib.evaluation_utils import evaluate_repe_status as evaluate_status
from lib.model_utils import load_latent_vectors, load_model_config


def main():
    config_path = project_root / "config" / "config.yml"
    model_id, _, _ = load_model_config(config_path)

    # 1. Load cached vectors
    X_benign, X_malicious, prompts_benign, prompts_malicious = load_latent_vectors(
        model_id
    )

    # 2. Train/Test Split for Out-of-Distribution Validation
    # We isolate a small calibration set (50 samples each) to learn the direction,
    # keeping 1,950 benign and 350 malicious prompts purely for held-out evaluation.
    np.random.seed(42)
    calib_benign_idx = np.random.choice(len(X_benign), size=50, replace=False)
    calib_malicious_idx = np.random.choice(len(X_malicious), size=50, replace=False)

    test_benign_idx = np.setdiff1d(np.arange(len(X_benign)), calib_benign_idx)
    test_malicious_idx = np.setdiff1d(np.arange(len(X_malicious)), calib_malicious_idx)

    # 3. Extract the Direction of Harm (Difference in Means)
    # Averaging cancels out background syntax and conversational variance,
    # isolating the single direction in activation space representing malicious intent.
    mu_malicious = np.mean(X_malicious[calib_malicious_idx], axis=0)
    mu_benign = np.mean(X_benign[calib_benign_idx], axis=0)

    harm_vector = mu_malicious - mu_benign
    harm_vector = harm_vector / np.linalg.norm(harm_vector)

    # 4. Evaluate Held-Out Prompts via Linear Projection
    X_test = np.vstack([X_benign[test_benign_idx], X_malicious[test_malicious_idx]])
    prompts_test = np.concatenate([prompts_benign[test_benign_idx], prompts_malicious[test_malicious_idx]])
    ground_truth_test = ["Benign"] * len(test_benign_idx) + ["Malicious"] * len(test_malicious_idx)
    y_true = np.array([0] * len(test_benign_idx) + [1] * len(test_malicious_idx))

    # Scalar projection: dot product against the unit direction vector
    scores = np.dot(X_test, harm_vector)

    # 5. Threshold Selection & Evaluation
    # Set threshold to maintain a 1% false positive rate on held-out benign queries
    benign_scores = scores[:len(test_benign_idx)]
    malicious_scores = scores[len(test_benign_idx):]

    threshold = np.percentile(benign_scores, 99)
    y_pred = (scores > threshold).astype(int)

    auc = roc_auc_score(y_true, scores)
    print("\n" + "=" * 40)
    print("REPRESENTATION ENGINEERING (RepE) RESULTS")
    print("=" * 40)
    print(f"ROC-AUC Score: {auc:.4f}")
    print(f"Decision Threshold (99th %ile Benign): {threshold:.4f}")
    print(f"Benign Correctly Passed:  {np.sum(y_pred[:len(test_benign_idx)] == 0)} / {len(test_benign_idx)}")
    print(f"False Alarms (FP):         {np.sum(y_pred[:len(test_benign_idx)] == 1)} / {len(test_benign_idx)}")
    print(f"Attacks Caught (TP):       {np.sum(y_pred[len(test_benign_idx):] == 1)} / {len(test_malicious_idx)}")
    print(f"Attacks Missed (FN):       {np.sum(y_pred[len(test_benign_idx):] == 0)} / {len(test_malicious_idx)}")
    print("=" * 40)

    # 6. Export Held-Out Results to CSV
    df = pd.DataFrame({
        "Ground_Truth": ground_truth_test,
        "RepE_Harm_Score": scores,
        "Prediction": ["Malicious" if p == 1 else "Benign" for p in y_pred],
        "Prompt_Text": prompts_test
    })

    df["Result_Status"] = df.apply(evaluate_status, axis=1)
    df = df.sort_values(by="RepE_Harm_Score", ascending=False)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"./data/repe_results_{ts}.csv"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\")
    print(f"\nSaved evaluation metrics and prompts to {csv_path}")


if __name__ == "__main__":
    main()