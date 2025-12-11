"""
fraud_model_training.py

Train machine-learning models to detect credit card fraud on
creditcard_transactions.csv and evaluate their performance.

This script:
- Loads the dataset
- Does minimal feature engineering
- Splits into train/test with stratification
- Trains Logistic Regression and Random Forest (class-imbalance aware)
- Evaluates with precision, recall, F1, ROC-AUC
- Saves the best model as models/best_fraud_model_<Name>.pkl
"""

import os
import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# 1. Config
# -----------------------------

RANDOM_SEED = 42

# Relative path from this script to the data file
DATA_PATH = os.path.join("..", "data", "creditcard_transactions.csv")

# Where to save trained model(s)
MODEL_DIR = os.path.join(".", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# 2. Data loading & features
# -----------------------------

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find data file at: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset with shape: {df.shape}")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional feature engineering:
    - Derive hour_of_day if we have a time column.
    """

    df = df.copy()

    # Try to infer a time column
    time_col_candidates = ["Time", "TimeSec", "Time_Sec", "time_sec", "Time Sec"]
    time_col = None
    for c in time_col_candidates:
        if c in df.columns:
            time_col = c
            break

    if time_col is not None:
        # Convert seconds to hour of day (0-23)
        df["hour_of_day"] = (df[time_col] // 3600).astype(int).clip(lower=0, upper=23)
        print(f"[INFO] Derived feature 'hour_of_day' from '{time_col}' column.")
    else:
        print("[INFO] No time column found. Skipping 'hour_of_day' feature.")

    return df


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Splits the dataframe into features (X) and label (y).
    Automatically detects fraud label column: class, Class, FraudFlag, etc.
    Returns: X, y, feature_names
    """

    possible_labels = [
        "class", "Class",
        "fraudflag", "FraudFlag", "fraud_flag",
        "fraud", "Fraud",
        "is_fraud", "Is_Fraud",
    ]

    # Map lowercase -> original column name
    normalized = {col.lower(): col for col in df.columns}

    label_col = None
    for lbl in possible_labels:
        if lbl.lower() in normalized:
            label_col = normalized[lbl.lower()]
            break

    if label_col is None:
        raise KeyError(
            "No fraud label column found. Expected one of: "
            "class, Class, FraudFlag, fraud, is_fraud."
        )

    print(f"[INFO] Using '{label_col}' as fraud label column.")

    # y = label column
    y = df[label_col].astype(int)

    # X = all other columns
    X = df.drop(columns=[label_col])

    # Drop time-like columns if present (we already derived hour_of_day)
    drop_cols = [c for c in ["Time", "time", "TimeSec", "time_sec", "Time Sec"] if c in X.columns]
    if drop_cols:
        print(f"[INFO] Dropping unused time columns: {drop_cols}")
        X = X.drop(columns=drop_cols)

    # List of feature names for later use (e.g., feature importance)
    feature_names = list(X.columns)

    return X, y, feature_names


# -----------------------------
# 3. Train/test split
# -----------------------------

def make_train_test_split(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,  # keeps class balance similar in train/test
    )
    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"[INFO] Fraud rate in train: {y_train.mean():.4f}")
    print(f"[INFO] Fraud rate in test:  {y_test.mean():.4f}")
    return X_train, X_test, y_train, y_test


# -----------------------------
# 4. Model training
# -----------------------------

def build_log_reg_pipeline():
    """
    Logistic Regression with scaling and class_weight='balanced'
    to handle class imbalance.
    """
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipe


def build_random_forest():
    """
    Random Forest with class_weight='balanced'
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )
    return rf


# -----------------------------
# 5. Evaluation helpers
# -----------------------------

def evaluate_model(name: str, model, X_test, y_test) -> float:
    """
    Evaluate the model and print useful metrics.
    Returns ROC-AUC for model comparison.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc = roc_auc_score(y_test, y_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    print(f"\n==================== {name} ====================")
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return roc


def show_feature_importance_rf(rf: RandomForestClassifier, feature_names: List[str]):
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    print("\n[INFO] Top 15 features by importance (Random Forest):")
    for i in idx[:15]:
        print(f"{feature_names[i]:<20s}  {importances[i]:.4f}")


# -----------------------------
# 6. Main training routine
# -----------------------------

def main():
    print("[INFO] Starting fraud model training...")

    # Load & prep data
    df = load_dataset(DATA_PATH)
    df = add_derived_features(df)
    X, y, feature_names = split_features_labels(df)

    # Train/test split
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    # Model 1: Logistic Regression
    logreg_model = build_log_reg_pipeline()
    logreg_model.fit(X_train, y_train)
    roc_logreg = evaluate_model("Logistic Regression", logreg_model, X_test, y_test)

    # Model 2: Random Forest
    rf_model = build_random_forest()
    rf_model.fit(X_train, y_train)
    roc_rf = evaluate_model("Random Forest", rf_model, X_test, y_test)

    # Feature importance for RF
    show_feature_importance_rf(rf_model, feature_names)

    # Choose best model by ROC-AUC
    if roc_rf >= roc_logreg:
        best_model = rf_model
        best_name = "RandomForest"
        best_roc = roc_rf
    else:
        best_model = logreg_model
        best_name = "LogisticRegression"
        best_roc = roc_logreg

    # Save best model
    model_path = os.path.join(MODEL_DIR, f"best_fraud_model_{best_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"\n[INFO] Best model: {best_name} with ROC-AUC={best_roc:.4f}")
    print(f"[INFO] Saved to: {model_path}")
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
