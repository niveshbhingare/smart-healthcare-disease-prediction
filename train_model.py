"""
train_model.py
----------------
Trains a machine learning model to predict a DISEASE CATEGORY (body system)
from symptoms, using the Community Healthcare Multi-Symptoms Disease
Diagnostic Dataset.

Why categories instead of the raw 46 individual diseases?
The raw symptom data is heavily noisy (verified empirically - see README),
so any model predicting all 46 individual diseases caps out around ~35%
accuracy. Grouping into 10 clinically meaningful body-system categories
raises real, explainable accuracy while staying medically sensible.
Both the baseline ("always guess the largest class") and true model
accuracy are reported side by side, so you can show genuine model skill,
not just class-imbalance luck.

Run:
    python train_model.py

Outputs (saved into ./model/):
    - disease_model.pkl        -> trained classifier
    - label_encoder.pkl        -> maps category name <-> numeric label
    - feature_columns.pkl      -> ordered list of symptom columns the model expects
    - model_metrics.json       -> accuracy / classification report (used by dashboard)
    - category_map.pkl         -> disease -> category lookup (used by dashboard)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "disease_dataset.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Map each of the 46 individual diseases to one of 10 broad body-system
# categories. This is a clinical simplification, not a data trick: each
# category groups diseases that share an organ system and, generally,
# overlapping symptom profiles.
# ---------------------------------------------------------------------------
DISEASE_CATEGORY_MAP = {
    # Respiratory
    "Asthma": "Respiratory", "Pneumonia": "Respiratory", "Flu": "Respiratory",
    "Common_Cold": "Respiratory", "COPD": "Respiratory", "COVID19": "Respiratory",
    "Cystic_Fibrosis": "Respiratory",
    # Infectious disease
    "Tuberculosis": "Infectious_Disease", "Dengue": "Infectious_Disease",
    "Typhoid_Fever": "Infectious_Disease",
    # Endocrine / metabolic
    "Hyperthyroidism": "Endocrine_Metabolic", "Obesity": "Endocrine_Metabolic",
    "Type2_Diabetes": "Endocrine_Metabolic", "Hypothyroidism": "Endocrine_Metabolic",
    "Polycystic_Ovary_Syndrome": "Endocrine_Metabolic", "Metabolic_Syndrome": "Endocrine_Metabolic",
    # Gastrointestinal
    "Irritable_Bowel_Syndrome": "Gastrointestinal", "Peptic_Ulcer_Disease": "Gastrointestinal",
    "Gastritis": "Gastrointestinal", "Cirrhosis": "Gastrointestinal",
    "Inflammatory_Bowel_Disease": "Gastrointestinal", "Ulcerative_Colitis": "Gastrointestinal",
    "Chronic_Liver_Disease": "Gastrointestinal", "Crohn_Disease": "Gastrointestinal",
    "Acute_Gastroenteritis": "Gastrointestinal",
    # Renal / urinary
    "Chronic_Kidney_Disease": "Renal_Urinary", "Urinary_Tract_Infection": "Renal_Urinary",
    # Cardiovascular
    "Hypertension": "Cardiovascular", "Coronary_Artery_Disease": "Cardiovascular",
    "Stroke": "Cardiovascular",
    # Musculoskeletal / connective tissue
    "Osteoarthritis": "Musculoskeletal_Connective", "Osteopetrosis": "Musculoskeletal_Connective",
    "Marfan_Syndrome": "Musculoskeletal_Connective", "Ehlers_Danlos_Syndrome": "Musculoskeletal_Connective",
    # Neurological
    "Migraine": "Neurological", "Epilepsy": "Neurological",
    "Amyotrophic_Lateral_Sclerosis": "Neurological", "Spinal_Muscular_Atrophy": "Neurological",
    "Multiple_Sclerosis": "Neurological",
    # Mental health
    "Depression": "Mental_Health", "Generalized_Anxiety_Disorder": "Mental_Health",
    # Autoimmune / genetic / rare
    "Systemic_Lupus_Erythematosus": "Autoimmune_Genetic_Rare", "Psoriasis": "Autoimmune_Genetic_Rare",
    "Niemann_Pick_Disease": "Autoimmune_Genetic_Rare", "Gaucher_Disease": "Autoimmune_Genetic_Rare",
    "Fanconi_Anemia": "Autoimmune_Genetic_Rare",
}


def load_data(path):
    df = pd.read_csv(path)

    # Sanity checks - catch a corrupted/truncated upload (common on Colab)
    # early, with a clear message, instead of a confusing crash later.
    expected_rows, expected_cols = 100000, 175
    if df.shape != (expected_rows, expected_cols):
        print(
            f"WARNING: expected shape ({expected_rows}, {expected_cols}) but got "
            f"{df.shape}. The CSV may be truncated or corrupted - re-download/re-upload it."
        )

    unmapped = set(df["disease"].dropna().unique()) - set(DISEASE_CATEGORY_MAP.keys())
    if unmapped:
        raise ValueError(f"These diseases have no category mapping: {unmapped}")

    df = df.copy()
    df["category"] = df["disease"].map(DISEASE_CATEGORY_MAP)
    feature_cols = [c for c in df.columns if c not in ("disease", "category")]

    n_missing = df[feature_cols].isna().sum().sum()
    if n_missing > 0:
        print(
            f"WARNING: found {n_missing} missing symptom values - filling with 0 "
            "(treated as 'symptom not present'). This shouldn't happen with the "
            "original dataset, so double check your CSV wasn't corrupted during upload."
        )

    # Coerce safely instead of a hard .astype(int) that crashes on NaN/inf
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    y = df["category"]

    # Drop any rows where disease/category ended up missing entirely
    valid_mask = y.notna()
    if (~valid_mask).sum() > 0:
        print(f"WARNING: dropping {(~valid_mask).sum()} rows with missing disease/category label.")
    return X[valid_mask], y[valid_mask], feature_cols


def main():
    print("Loading dataset and mapping diseases to categories...")
    X, y, feature_cols = load_data(DATA_PATH)
    print(f"Rows: {len(X)}, Symptom features: {len(feature_cols)}, Categories: {y.nunique()}")
    print("\nCategory sizes:")
    print(y.value_counts())

    # Baseline: what accuracy would you get by always guessing the biggest class?
    baseline_acc = y.value_counts(normalize=True).max()
    print(f"\n'Always guess majority class' baseline accuracy: {baseline_acc:.1%}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    # --- Model 1: Random Forest (main model) ---
    print("\nTraining Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=22,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average="macro")
    print(f"Random Forest -> accuracy: {rf_acc:.4f}, macro-F1: {rf_f1:.4f}")

    # --- Model 2: Logistic Regression (baseline for comparison) ---
    print("Training Logistic Regression baseline...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred, average="macro")
    print(f"Logistic Regression -> accuracy: {lr_acc:.4f}, macro-F1: {lr_f1:.4f}")

    if rf_acc >= lr_acc:
        best_model, best_name, best_pred, best_acc, best_f1 = rf, "RandomForestClassifier", rf_pred, rf_acc, rf_f1
    else:
        best_model, best_name, best_pred, best_acc, best_f1 = lr, "LogisticRegression", lr_pred, lr_acc, lr_f1

    print(f"\nSelected best model: {best_name}")
    print(f"Test accuracy: {best_acc:.1%}  (vs {baseline_acc:.1%} majority-class baseline)")

    all_labels = np.arange(len(le.classes_))
    report = classification_report(
        y_test, best_pred, labels=all_labels, target_names=le.classes_,
        output_dict=True, zero_division=0,
    )

    top_features = []
    if best_name == "RandomForestClassifier":
        importances = best_model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        top_features = [(feature_cols[i], float(importances[i])) for i in top_idx]
        print("\nTop 15 most predictive symptoms:")
        for f, imp in top_features:
            print(f"  {f}: {imp:.4f}")

    joblib.dump(best_model, os.path.join(MODEL_DIR, "disease_model.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_columns.pkl"))
    joblib.dump(DISEASE_CATEGORY_MAP, os.path.join(MODEL_DIR, "category_map.pkl"))

    metrics = {
        "model_name": best_name,
        "accuracy": best_acc,
        "macro_f1": best_f1,
        "baseline_accuracy": float(baseline_acc),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": len(le.classes_),
        "classes": le.classes_.tolist(),
        "top_features": top_features,
        "classification_report": report,
        "comparison": {
            "random_forest": {"accuracy": rf_acc, "macro_f1": rf_f1},
            "logistic_regression": {"accuracy": lr_acc, "macro_f1": lr_f1},
        },
    }
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model + metrics to ./{MODEL_DIR}/")


if __name__ == "__main__":
    main()
