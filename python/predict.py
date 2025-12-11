"""
predict.py

Use the trained fraud detection model to score new credit card transactions.

This script:
- Loads the best trained model from ./models/best_fraud_model_*.pkl
- Loads an input CSV file of transactions
- Applies the same feature engineering as training (hour_of_day from time_sec)
- Drops label/time columns as in training
- Outputs predictions, fraud probabilities, and (optionally) evaluation metrics
  if a true label column is present (class / Class / FraudFlag / etc.)

Usage (from the python/ directory):

    # 1) Score a new file WITHOUT labels (pure inference)
    python predict.py --input ../data/new_transactions.csv --output ../data/predictions.csv

    # 2) Score the original dataset WITH labels (gets evaluation metrics)
    python predict.py --input ../data/creditcard_transactions.csv --output ../data/creditcard_scored.csv
"""

import argparse
import os
import pickle
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
)


# ------------------------------------------------------------------
# 1. Config
# ------------------------------------------------------------------

RANDOM_SEED = 42

# Default model location (same as fraud_model_training.py)
MODEL_DIR = os.path.join(".", "models")
# If you ever change the model file name, update this pattern or hard-code it.
DEFAULT_MODEL_PATH = None  # we will resolve dynamically


# ------------------------------------------------------------------
# 2. Utility helpers
# ------------------------------------------------------------------

def find_best_model_path(model_dir: str) -> str:
    """
    Attempts to find a 'best_fraud_model_*.pkl' file in the model_dir.
    If multiple files exist, picks the most recently modified.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    candidates: List[str] = []
    for fname in os.listdir(model_dir):
        if fname.startswith("best_fraud_model_") and fname.endswith(".pkl"):
            candidates.append(os.path.join(model_dir, fname))

    if not candidates:
        raise FileNotFoundError(
            f"No best_fraud_model_*.pkl found in {model_dir}. "
            f"Train a model first with fraud_model_training.py."
        )

    # Pick most recently modified file
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model file at: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"[INFO] Loaded trained model from: {model_path}")
    return model


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same feature engineering as in fraud_model_training.py:
    - Derive 'hour_of_day' from a time column if present.
    """
    df = df.copy()

    time_col_candidates = ["Time", "TimeSec", "Time_Sec", "time_sec", "Time Sec"]
    time_col = None
    for c in time_col_candidates:
        if c in df.columns:
            time_col = c
            break

    if time_col is not None:
        df["hour_of_day"] = (df[time_col] // 3600).astype(int).clip(lower=0, upper=23)
        print(f"[INFO] Derived feature 'hour_of_day' from '{time_col}' column.")
    else:
        print("[INFO] No time column found. Skipping 'hour_of_day' feature.")

    return df


def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    """
    Tries to detect the fraud label column if it exists.
    Returns the column name or None if not found.
    """
    possible_labels = [
        "class", "Class",
        "fraudflag", "FraudFlag", "fraud_flag",
        "fraud", "Fraud",
        "is_fraud", "Is_Fraud",
    ]

    normalized = {col.lower(): col for col in df.columns}

    for lbl in possible_labels:
        if lbl.lower() in normalized:
            return normalized[lbl.lower()]

    return None


def prepare_features_for_inference(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """
    Applies the same transformation as in training:
    - Adds hour_of_day if possible
    - Detects and separates label column if present
    - Drops time columns from X
    Returns:
        X: feature dataframe
        y_true: true labels (or None if no label column)
        feature_names: list of feature names in X
    """
    df = add_derived_features(df)

    label_col = detect_label_column(df)
    y_true: Optional[pd.Series] = None

    if label_col is not None:
        print(f"[INFO] Detected label column '{label_col}' for evaluation.")
        y_true = df[label_col].astype(int)
        X = df.drop(columns=[label_col])
    else:
        print("[INFO] No label column detected. Running in pure prediction mode.")
        X = df

    # Drop time columns (same as training)
    drop_cols = [c for c in ["Time", "time", "TimeSec", "time_sec"] if c in X.columns]
    if drop_cols:
        print(f"[INFO] Dropping time columns from features: {drop_cols}")
        X = X.drop(columns=drop_cols)

    feature_names = list(X.columns)
    print(f"[INFO] Final feature shape for inference: {X.shape}")
    return X, y_true, feature_names


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Print evaluation metrics when ground truth labels are available.
    """
    roc = roc_auc_score(y_true, y_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    print("\n==================== Evaluation on Provided Labels ====================")
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


# ------------------------------------------------------------------
# 3. Main prediction routine
# ------------------------------------------------------------------

def run_inference(
    input_path: str,
    output_path: str,
    model_path: Optional[str] = None,
) -> None:
    # Resolve model path
    if model_path is None:
        resolved_model_path = find_best_model_path(MODEL_DIR)
    else:
        resolved_model_path = model_path

    model = load_model(resolved_model_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    print(f"[INFO] Loading input data from: {input_path}")
    df_raw = pd.read_csv(input_path)
    print(f"[INFO] Input shape: {df_raw.shape}")

    X, y_true, feature_names = prepare_features_for_inference(df_raw)

    # Get probabilities and predictions
    print("[INFO] Generating fraud probabilities...")
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Attach predictions to a copy of original dataframe for output
    df_out = df_raw.copy()
    df_out["fraud_probability"] = y_prob
    df_out["fraud_prediction"] = y_pred

    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"[INFO] Saved scored dataset to: {output_path}")

    # If labels exist, evaluate
    if y_true is not None:
        evaluate_predictions(y_true, y_prob, y_pred)
    else:
        print("[INFO] No labels found in input. Skipping evaluation metrics.")


# ------------------------------------------------------------------
# 4. CLI entry point
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score credit card transactions with the trained fraud model."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file with transaction data.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV file where predictions will be saved.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional path to a specific model .pkl file. "
            "If not provided, the most recent 'best_fraud_model_*.pkl' "
            "in ./models will be used."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_inference(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()
