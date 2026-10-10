"""
FloodGuard: Real-World INDOFLOODS Machine Learning Pipeline
Trains:
1. Severe Flood Hazard Classifier (Random Forest) predicting 'Flood' vs 'Severe Flood'
2. Peak Flood Level / Discharge Regressor (Gradient Boosting)
3. Computes benchmarks and serializes trained models with metadata.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, r2_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "indofloods_merged_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

INDOFLOODS_FEATURES = [
    "Peak Flood Level (m)",
    "Warning Level",
    "Danger Level",
    "Gauge_Danger_Surge_m",
    "Water_to_Danger_Ratio",
    "Event Duration (days)",
    "Time to Peak (days)",
    "Recession Time (day)",
    "Num Peak FL",
    "Catchment Area"
]

def train_indofloods():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    
    # Filter valid rows
    data = df.dropna(subset=INDOFLOODS_FEATURES + ["Is_Severe_Flood"]).copy()
    
    X = data[INDOFLOODS_FEATURES]
    y_class = data["Is_Severe_Flood"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # 1. Train Severe Flood Classifier
    clf = RandomForestClassifier(
        n_estimators=180,
        max_depth=8,
        min_samples_split=4,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print("\n--- INDOFLOODS Real Data Model Evaluation ---")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    
    # Feature importances
    importances = dict(zip(INDOFLOODS_FEATURES, [round(float(x), 4) for x in clf.feature_importances_]))
    
    # 2. Train Peak Discharge Regressor (on events where streamflow entries exist)
    reg_data = df.dropna(subset=INDOFLOODS_FEATURES + ["Peak Discharge Q (cumec)"]).copy()
    reg_data = reg_data[reg_data["Peak Discharge Q (cumec)"] > 0]
    
    reg_features = ["Peak Flood Level (m)", "Warning Level", "Danger Level", "Gauge_Danger_Surge_m", "Event Duration (days)", "Catchment Area"]
    X_reg = reg_data[reg_features]
    y_reg = np.log1p(reg_data["Peak Discharge Q (cumec)"]) # Log transform for scale stability
    
    reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42)
    reg.fit(reg_X_train, reg_y_train)
    
    reg_pred = reg.predict(reg_X_test)
    r2 = r2_score(reg_y_test, reg_pred)
    mae = mean_absolute_error(np.expm1(reg_y_test), np.expm1(reg_pred))
    print(f"Discharge Regressor Log R2: {r2:.4f} | Median Absolute Error: {mae:.2f} cumec")
    
    # Save artifacts
    clf_path = os.path.join(MODELS_DIR, "indofloods_classifier.joblib")
    reg_path = os.path.join(MODELS_DIR, "indofloods_regressor.joblib")
    meta_path = os.path.join(MODELS_DIR, "indofloods_metadata.json")
    
    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)
    
    metadata = {
        "features": INDOFLOODS_FEATURES,
        "reg_features": reg_features,
        "metrics": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "confusion_matrix": cm,
            "discharge_log_r2": round(float(r2), 4)
        },
        "feature_importances": importances
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"INDOFLOODS models saved to: {MODELS_DIR}")
    return metadata

if __name__ == "__main__":
    train_indofloods()
