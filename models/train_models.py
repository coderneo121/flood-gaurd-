"""
FloodGuard: Machine Learning Model Training Pipeline
Trains:
1. Multi-class Gradient Boosting / Random Forest Classifier for Early Flood Warning Risk Levels.
2. Gradient Boosting Regressor for 24h Peak River Water Level Forecast.
Computes comprehensive metrics and exports serialized models with metadata.
"""

import os
import sys
import json
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix

FEATURE_COLUMNS = [
    "elevation_m",
    "rainfall_24h_mm",
    "rainfall_72h_mm",
    "soil_moisture_pct",
    "river_discharge_m3s",
    "current_water_level_m",
    "reservoir_capacity_pct",
    "topographic_wetness_index",
    "drainage_efficiency_score",
    "urbanization_ratio",
    "runoff_risk_ratio",
    "capacity_stress",
    "hazard_flow_index"
]

RISK_LABELS = {
    0: "Safe (Normal)",
    1: "Advisory (Watch)",
    2: "Warning (High Risk)",
    3: "Emergency (Severe Flood)"
}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds hydrological domain features."""
    df = df.copy()
    # Soil saturation & precipitation interaction
    df["runoff_risk_ratio"] = (df["rainfall_24h_mm"] * (df["soil_moisture_pct"] / 100.0)).round(3)
    # Upstream stress: high reservoir level combined with limited drainage capacity
    df["capacity_stress"] = (df["reservoir_capacity_pct"] * (1.0 - df["drainage_efficiency_score"])).round(3)
    # Hydraulic pressure: current gauge height scaled by flow discharge
    df["hazard_flow_index"] = (df["current_water_level_m"] * (df["river_discharge_m3s"] / 100.0)).round(3)
    return df

def train_and_evaluate(data_path: str, model_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading hydrological dataset from {data_path}...")
    df = pd.read_csv(data_path)
    df = engineer_features(df)
    
    X = df[FEATURE_COLUMNS]
    y_class = df["flood_risk_level"]
    y_reg = df["forecasted_peak_water_level_m"]
    
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # 1. Train Classification Model (Random Forest Classifier)
    print("\n[1/2] Training Flood Risk Classifier (Random Forest)...")
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_cls_train)
    y_cls_pred = clf.predict(X_test)
    
    cls_acc = accuracy_score(y_cls_test, y_cls_pred)
    cls_f1 = f1_score(y_cls_test, y_cls_pred, average="weighted")
    cls_cm = confusion_matrix(y_cls_test, y_cls_pred).tolist()
    
    print(f"Classifier Accuracy: {cls_acc:.4f} | Weighted F1: {cls_f1:.4f}")
    print("Classification Report:\n", classification_report(y_cls_test, y_cls_pred, target_names=[RISK_LABELS[i] for i in range(4)]))
    
    # Feature importances for classification
    clf_importances = dict(zip(FEATURE_COLUMNS, [round(float(x), 4) for x in clf.feature_importances_]))
    
    # 2. Train Regression Model (Gradient Boosting Regressor)
    print("\n[2/2] Training Peak River Water Level Regressor (Gradient Boosting)...")
    reg = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        random_state=42
    )
    reg.fit(X_train, y_reg_train)
    y_reg_pred = reg.predict(X_test)
    
    reg_mae = mean_absolute_error(y_reg_test, y_reg_pred)
    reg_rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    reg_r2 = r2_score(y_reg_test, y_reg_pred)
    
    print(f"Regressor R2 Score: {reg_r2:.4f} | MAE: {reg_mae:.3f}m | RMSE: {reg_rmse:.3f}m")
    
    reg_importances = dict(zip(FEATURE_COLUMNS, [round(float(x), 4) for x in reg.feature_importances_]))
    
    # Save artifacts
    clf_path = os.path.join(model_dir, "flood_classifier.joblib")
    reg_path = os.path.join(model_dir, "flood_regressor.joblib")
    meta_path = os.path.join(model_dir, "model_metadata.json")
    
    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)
    
    metadata = {
        "features": FEATURE_COLUMNS,
        "risk_labels": RISK_LABELS,
        "classifier": {
            "model_type": "RandomForestClassifier",
            "accuracy": round(float(cls_acc), 4),
            "f1_weighted": round(float(cls_f1), 4),
            "confusion_matrix": cls_cm,
            "feature_importances": clf_importances
        },
        "regressor": {
            "model_type": "GradientBoostingRegressor",
            "r2_score": round(float(reg_r2), 4),
            "mae": round(float(reg_mae), 4),
            "rmse": round(float(reg_rmse), 4),
            "feature_importances": reg_importances
        }
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nModels and metadata saved successfully in: {model_dir}")
    return metadata

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "flood_sensor_data.csv")
    model_dir = os.path.join(base_dir, "models")
    
    if not os.path.exists(data_path):
        from data.generate_dataset import generate_flood_dataset
        df = generate_flood_dataset(3000)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        
    train_and_evaluate(data_path, model_dir)
