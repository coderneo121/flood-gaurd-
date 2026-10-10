"""
Ingestion & Parsing Script for Real-World INDOFLOODS Datasets
Saves the raw Indian Hydrological telemetry datasets and creates an enriched merged dataset.
"""

import os
import re
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def save_and_merge():
    # File paths
    events_file = os.path.join(DATA_DIR, "indofloods_events.csv")
    stations_file = os.path.join(DATA_DIR, "indofloods_stations.csv")
    projects_file = os.path.join(DATA_DIR, "state_flood_projects.csv")
    merged_file = os.path.join(DATA_DIR, "indofloods_merged_dataset.csv")

    if not os.path.exists(events_file) or not os.path.exists(stations_file):
        print("Raw files missing, please ensure raw CSVs are saved.")
        return

    df_events = pd.read_csv(events_file)
    df_stations = pd.read_csv(stations_file)

    # Extract GaugeID from EventID (e.g. INDOFLOODS-gauge-1010-1 -> INDOFLOODS-gauge-1010)
    df_events["GaugeID"] = df_events["EventID"].apply(lambda x: "-".join(str(x).split("-")[:-1]))

    # Merge on GaugeID
    merged = pd.merge(df_events, df_stations, on="GaugeID", how="inner")
    
    # Feature engineering for hydrological dynamics
    # Height breach above warning and danger levels
    merged["Warning Level"] = pd.to_numeric(merged["Warning Level"], errors="coerce")
    merged["Danger Level"] = pd.to_numeric(merged["Danger Level"], errors="coerce")
    merged["Peak Flood Level (m)"] = pd.to_numeric(merged["Peak Flood Level (m)"], errors="coerce")
    
    # Water level normalized to danger level
    merged["Gauge_Danger_Surge_m"] = (merged["Peak Flood Level (m)"] - merged["Danger Level"]).round(2)
    merged["Water_to_Danger_Ratio"] = (merged["Peak Flood Level (m)"] / merged["Danger Level"]).round(3)
    
    # Fill numeric columns
    merged["Peak Discharge Q (cumec)"] = pd.to_numeric(merged["Peak Discharge Q (cumec)"], errors="coerce")
    merged["Flood Volume (cumec)"] = pd.to_numeric(merged["Flood Volume (cumec)"], errors="coerce")
    merged["Catchment Area"] = pd.to_numeric(merged["Catchment Area"], errors="coerce")
    
    # Binary target for classification: Severe Flood (1) vs Flood (0)
    merged["Is_Severe_Flood"] = (merged["Flood Type"].str.strip() == "Severe Flood").astype(int)
    
    merged.to_csv(merged_file, index=False)
    print(f"Merged INDOFLOODS dataset saved to {merged_file} with {len(merged)} records.")
    return merged

if __name__ == "__main__":
    save_and_merge()
