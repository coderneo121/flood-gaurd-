"""
FloodGuard: Realistic Synthetic Hydrological Dataset Generator
Generates multi-station time-series sensor data modeling storm events,
antecedent soil saturation, river basin discharge, and flood risk classes.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_flood_dataset(n_samples=2500, random_state=42):
    np.random.seed(random_state)
    
    # Station locations in river catchment basin
    stations = [
        {"name": "Station A - Upper Highland", "lat": 18.5204, "lon": 73.8567, "elevation_m": 580},
        {"name": "Station B - Mid-River Gorge", "lat": 18.5314, "lon": 73.8446, "elevation_m": 420},
        {"name": "Station C - Valley Confluence", "lat": 18.5089, "lon": 73.8258, "elevation_m": 310},
        {"name": "Station D - Lowland Metro Bridge", "lat": 18.4981, "lon": 73.8610, "elevation_m": 190},
        {"name": "Station E - Coastal Estuary", "lat": 18.4750, "lon": 73.8902, "elevation_m": 45},
    ]
    
    start_time = datetime(2025, 6, 1, 0, 0)
    records = []
    
    for i in range(n_samples):
        timestamp = start_time + timedelta(hours=i % 720) # Cycles across monsoon months
        station = stations[i % len(stations)]
        
        # Seasonality & precipitation simulation (occasional extreme monsoon surge)
        is_surge_event = np.random.rand() < 0.15
        
        if is_surge_event:
            rainfall_24h = np.random.exponential(scale=65) + 40
            rainfall_72h = rainfall_24h * np.random.uniform(1.8, 2.9)
            soil_moisture = np.clip(np.random.normal(82, 8), 60, 100)
            reservoir_capacity = np.clip(np.random.normal(88, 7), 65, 100)
        else:
            rainfall_24h = np.random.exponential(scale=12)
            rainfall_72h = rainfall_24h * np.random.uniform(1.1, 1.8) + np.random.uniform(0, 15)
            soil_moisture = np.clip(np.random.normal(45, 15), 10, 85)
            reservoir_capacity = np.clip(np.random.normal(55, 15), 15, 85)
            
        # Physical hydrological correlation:
        # River discharge (m3/s) depends on rainfall, antecedent saturation, and station elevation
        base_discharge = (600 - station["elevation_m"]) * 0.4
        runoff_coeff = 0.35 + (soil_moisture / 100.0) * 0.45
        river_discharge = max(15.0, base_discharge + (rainfall_24h * 5.2 * runoff_coeff) + np.random.normal(0, 8))
        
        # River water gauge height (meters)
        # Normal baseline: 2.0 to 4.0 meters. Danger threshold starts ~ 7.5 meters.
        current_water_level = 2.2 + (river_discharge / 110.0) + (rainfall_24h * 0.035) + np.random.normal(0, 0.25)
        current_water_level = max(1.0, current_water_level)
        
        # Topographic and environmental features
        topographic_wetness_index = np.round(np.random.uniform(6.5, 12.5), 2)
        drainage_efficiency = np.clip(np.random.normal(0.68, 0.12), 0.2, 0.98)
        urbanization_ratio = np.clip(np.random.normal(0.55, 0.20), 0.1, 0.95)
        
        # Future predicted peak water level in next 18 hours (Physical Target for Regression)
        lag_surge = (rainfall_72h * 0.02) + (reservoir_capacity > 85) * 1.5 + (1.0 - drainage_efficiency) * 1.2
        peak_water_level_next_24h = current_water_level + lag_surge + np.random.normal(0, 0.3)
        peak_water_level_next_24h = round(max(current_water_level, peak_water_level_next_24h), 2)
        
        # Multi-class Risk Assessment Target (Classification)
        # 0: Normal (Safe)
        # 1: Advisory (Watch)
        # 2: Warning (High Risk)
        # 3: Severe Emergency (Extreme Risk)
        if peak_water_level_next_24h < 5.0 and rainfall_24h < 40:
            risk_class = 0 # Safe
        elif peak_water_level_next_24h < 7.2 and rainfall_24h < 85:
            risk_class = 1 # Advisory
        elif peak_water_level_next_24h < 9.5:
            risk_class = 2 # Warning
        else:
            risk_class = 3 # Emergency
            
        records.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "station_name": station["name"],
            "station_latitude": station["lat"],
            "station_longitude": station["lon"],
            "elevation_m": station["elevation_m"],
            "rainfall_24h_mm": round(rainfall_24h, 2),
            "rainfall_72h_mm": round(rainfall_72h, 2),
            "soil_moisture_pct": round(soil_moisture, 2),
            "river_discharge_m3s": round(river_discharge, 2),
            "current_water_level_m": round(current_water_level, 2),
            "reservoir_capacity_pct": round(reservoir_capacity, 2),
            "topographic_wetness_index": topographic_wetness_index,
            "drainage_efficiency_score": round(drainage_efficiency, 2),
            "urbanization_ratio": round(urbanization_ratio, 2),
            "forecasted_peak_water_level_m": peak_water_level_next_24h,
            "flood_risk_level": risk_class
        })
        
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    df = generate_flood_dataset(n_samples=3000)
    output_path = os.path.join(os.path.dirname(__file__), "flood_sensor_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} flood monitoring records saved to: {output_path}")
    print(df["flood_risk_level"].value_counts(normalize=True).rename({
        0: "0 - Normal (Safe)",
        1: "1 - Advisory (Watch)",
        2: "2 - Warning (High)",
        3: "3 - Emergency (Extreme)"
    }))
