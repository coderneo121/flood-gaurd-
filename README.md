# 🌊 FloodGuard: AI-Powered Hydrological Early Warning & Flood Prediction System

**FloodGuard** is an end-to-end Machine Learning and decision-support system designed for municipal disaster management, river basin authorities, and civil defense agencies. It continuously analyzes meteorological and river sensor telemetry to forecast river peak heights and categorize flood hazard levels up to 24 hours in advance.

---

## 🌟 Key Features

1. **Dual Machine Learning Engine**:
   - **Flood Risk Classification**: Random Forest Classifier evaluating 4 operational danger tiers (*Safe*, *Advisory Watch*, *Flood Warning*, *Emergency*).
   - **Continuous Peak Gauge Regression**: Gradient Boosting Regressor predicting maximum forecasted water height (in meters) over the next 12-24 hours.
2. **Hydrological Domain Feature Engineering**:
   - *Runoff Risk Ratio*: Interaction between 24h precipitation intensity and antecedent soil saturation percentage.
   - *Capacity Stress*: Upstream reservoir storage pressure against municipal drainage channel efficiency.
   - *Hydraulic Flow Index*: Current river gauge level coupled with flow discharge rate ($m^3/s$).
3. **INDOFLOODS Real-World Indian River Basins Module**:
   - Integrated **885 historical flood events** from the Indian Flood Inventory across the Cauvery, Ganga, Tapi, Godavari, Krishna, Narmada, and Mahanadi river basins.
   - Real-data **Severe Flood Classifier** (Random Forest) & **Peak Discharge Regressor** ($R^2 \approx 0.897$).
   - Geospatial river gauge mapping with official CWC Warning & Danger thresholds.
   - National Flood Mitigation Outlays tracker across 25+ states (analyzing ₹16,051 Crore in central investments).
4. **Interactive Streamlit Web Dashboard**:
   - **Real Catchment AI Explorer**: Live simulations on real Indian gauge stations.
   - **Live Risk Simulator**: Interactive sliders and real-world disaster presets.
   - **Basin GIS Map & Time-Series Trends**: Hydrographs, station maps, and danger mark monitoring.
   - **Batch Telemetry Ingestion**: Upload custom CSV telemetry streams and export automated risk scores.

---

## 📁 Project Architecture

```
flood_guard/
├── data/
│   ├── indofloods_events.csv          # Real Indian flood events (885 historical records)
│   ├── indofloods_stations.csv        # Real river gauge stations with Warning/Danger marks
│   ├── indofloods_merged_dataset.csv  # Enriched merged INDOFLOODS training dataset
│   ├── state_flood_projects.csv       # Government flood mitigation investment dataset
│   ├── generate_dataset.py            # Synthetic hydrological sensor generator
│   └── flood_sensor_data.csv          # Synthetic basin telemetry
├── models/
│   ├── train_indofloods_model.py      # ML pipeline for real INDOFLOODS dataset
│   ├── indofloods_classifier.joblib   # Trained Severe Flood classifier
│   ├── indofloods_regressor.joblib    # Trained Peak Discharge regressor
│   ├── indofloods_metadata.json       # INDOFLOODS evaluation metrics
│   ├── train_models.py                # Synthetic basin ML training pipeline
│   ├── flood_classifier.joblib        # Serialized Random Forest classifier
│   └── flood_regressor.joblib         # Serialized Gradient Boosting regressor
├── utils/
│   ├── __init__.py
│   └── risk_engine.py                 # Civil protection protocols and action guidelines
├── app.py                             # Streamlit interactive application
├── requirements.txt                   # Project dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Generate Dataset & Train ML Models
```bash
python models/train_models.py
```

### 3. Launch the Web Application
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Flood Alert Classification Tiers

| Level | Severity | River Water Level | Civil Defense Protocol |
| :--- | :--- | :--- | :--- |
| **0: Safe** | Baseline / Normal | $< 5.0\text{ m}$ | Standard 6-hour polling; inspect drainage channels |
| **1: Advisory** | Watch | $5.0 - 7.5\text{ m}$ | 1-hour polling; pre-stage barrier pumps; issue advisory |
| **2: Warning** | High Risk | $7.5 - 9.5\text{ m}$ | Controlled dam releases; mobilize rescue teams; voluntary evacuation |
| **3: Emergency** | Severe Flood | $> 9.5\text{ m}$ | Sound flood sirens; mandatory evacuation; emergency relief shelters |

---

## 🔬 Technologies Used
- **Machine Learning**: Scikit-Learn (Random Forest, Gradient Boosting)
- **Data Engineering**: Pandas, NumPy
- **Interactive UI**: Streamlit, Plotly Express & Graph Objects
- **Serialization**: Joblib, JSON
