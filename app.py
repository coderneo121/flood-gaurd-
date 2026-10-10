"""
FloodGuard AI - Early Warning & Hydrological Flood Prediction System
Built with Streamlit, Scikit-Learn, and Plotly
"""

import os
import sys

# Ensure local modules are found
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.risk_engine import get_risk_meta
from models.train_models import FEATURE_COLUMNS, RISK_LABELS, engineer_features

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "flood_sensor_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "flood_classifier.joblib")
REGRESSOR_PATH = os.path.join(MODELS_DIR, "flood_regressor.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# INDOFLOODS Real-World Dataset & Model Paths
INDO_DATA_PATH = os.path.join(BASE_DIR, "data", "indofloods_merged_dataset.csv")
PROJECTS_DATA_PATH = os.path.join(BASE_DIR, "data", "state_flood_projects.csv")
INDO_CLF_PATH = os.path.join(MODELS_DIR, "indofloods_classifier.joblib")
INDO_REG_PATH = os.path.join(MODELS_DIR, "indofloods_regressor.joblib")
INDO_META_PATH = os.path.join(MODELS_DIR, "indofloods_metadata.json")

# Streamlit Page Config
st.set_page_config(
    page_title="FloodGuard AI | Hydrological Early Warning",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_data():
    """Loads trained ML models, metadata, and datasets."""
    classifier = None
    regressor = None
    metadata = None
    df_data = None
    
    indo_clf = None
    indo_reg = None
    indo_meta = None
    df_indo = None
    df_projects = None
    
    if os.path.exists(CLASSIFIER_PATH) and os.path.exists(REGRESSOR_PATH):
        classifier = joblib.load(CLASSIFIER_PATH)
        regressor = joblib.load(REGRESSOR_PATH)
        
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
            
    if os.path.exists(DATA_PATH):
        df_data = pd.read_csv(DATA_PATH)
        
    # Load INDOFLOODS Real Data Artifacts
    if os.path.exists(INDO_CLF_PATH) and os.path.exists(INDO_REG_PATH):
        indo_clf = joblib.load(INDO_CLF_PATH)
        indo_reg = joblib.load(INDO_REG_PATH)
        
    if os.path.exists(INDO_META_PATH):
        with open(INDO_META_PATH, "r") as f:
            indo_meta = json.load(f)
            
    if os.path.exists(INDO_DATA_PATH):
        df_indo = pd.read_csv(INDO_DATA_PATH)
        
    if os.path.exists(PROJECTS_DATA_PATH):
        df_projects = pd.read_csv(PROJECTS_DATA_PATH)
        
    return classifier, regressor, metadata, df_data, indo_clf, indo_reg, indo_meta, df_indo, df_projects

classifier, regressor, metadata, df_data, indo_clf, indo_reg, indo_meta, df_indo, df_projects = load_models_and_data()

# Sidebar: Controls & System Info
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1547683905-f686c993aae5?auto=format&fit=crop&w=600&q=80", use_container_width=True)
    st.title("🛡️ FloodGuard AI")
    st.markdown("**Real-Time Hydrological Defense & Early Warning System**")
    st.divider()
    
    st.subheader("System Status")
    if classifier and regressor:
        st.success("🟢 AI Models Loaded & Operational")
    else:
        st.error("🔴 Models not found. Run training script first.")
        
    st.metric(
        label="Classifier Accuracy",
        value=f"{metadata['classifier']['accuracy']*100:.1f}%" if metadata else "N/A"
    )
    st.metric(
        label="Peak Level Regressor R²",
        value=f"{metadata['regressor']['r2_score']:.3f}" if metadata else "N/A"
    )
    
    st.divider()
    zip_export_path = os.path.join(os.path.dirname(BASE_DIR), "flood_guard.zip")
    if os.path.exists(zip_export_path):
        with open(zip_export_path, "rb") as zf:
            st.download_button(
                label="📦 Download Full Project (ZIP)",
                data=zf.read(),
                file_name="flood_guard.zip",
                mime="application/zip",
                use_container_width=True
            )
            
    st.markdown("### Emergency Hotlines")
    st.markdown("- **Disaster Management:** 108 / 112\n- **Flood Control Room:** 1800-180-1551\n- **Weather Advisory:** IMD Telemetry")

# Main Header
st.markdown('<div class="main-header">🌊 FloodGuard: AI Early Flood Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-variable sensor telemetry forecasting river gauge levels, inundation risk, and automated civil defense actions.</div>', unsafe_allow_html=True)

# Tabs Navigation
tab_indo, tab_sim, tab_map, tab_analytics, tab_perf, tab_batch = st.tabs([
    "🇮🇳 INDOFLOODS Real Catchment AI",
    "🚨 Live Risk Simulator",
    "🗺️ Basin Telemetry Map",
    "📈 Hydrological Trends",
    "🤖 Model Performance",
    "📁 Batch Data Assessment"
])

# ----------------- TAB INDOFLOODS: REAL CATCHMENT AI -----------------
with tab_indo:
    st.markdown("### 🇮🇳 INDOFLOODS Real Catchment Intelligence & AI Severity Engine")
    st.markdown("Trained on **885 historical flood events** across major Indian river basins (Cauvery, Ganga, Tapi, Godavari, Krishna, Narmada, Mahanadi, etc.).")
    
    if df_indo is not None:
        # High level summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Historical Flood Events", f"{len(df_indo):,}")
        m2.metric("Severe Flood Records", f"{(df_indo['Flood Type'] == 'Severe Flood').sum():,}")
        m3.metric("River Gauges Cataloged", f"{df_indo['GaugeID'].nunique():,}")
        m4.metric("National Mitigation Funds", "₹16,051 Cr" if df_projects is not None else "N/A")
        
        st.divider()
        
        # Sub-sections
        sec_col1, sec_col2 = st.columns([1.1, 0.9])
        
        with sec_col1:
            st.markdown("#### 🗺️ Interactive National Gauging Map")
            basins = ["All Basins"] + sorted([str(b) for b in df_indo["Basin"].dropna().unique()])
            sel_basin = st.selectbox("Filter by River Basin:", basins)
            
            map_data = df_indo if sel_basin == "All Basins" else df_indo[df_indo["Basin"] == sel_basin]
            station_map = map_data.groupby("Station").first().reset_index()
            
            fig_indo_map = px.scatter_mapbox(
                station_map,
                lat="Latitude",
                lon="Longitude",
                color="Reliability",
                size="Catchment Area",
                size_max=20,
                hover_name="Station",
                hover_data={
                    "River Name/ Tributory/ SubTributory": True,
                    "State": True,
                    "Warning Level": ":.2f m",
                    "Danger Level": ":.2f m",
                    "Latitude": False,
                    "Longitude": False
                },
                color_discrete_map={"Safe": "#10B981", "Caution": "#F59E0B"},
                zoom=4,
                center={"lat": 20.5937, "lon": 78.9629},
                mapbox_style="carto-positron",
                title=f"Indian River Gauge Stations ({len(station_map)} nodes)"
            )
            fig_indo_map.update_layout(height=420, margin=dict(l=0, r=0, t=35, b=0))
            st.plotly_chart(fig_indo_map, use_container_width=True)
            
        with sec_col2:
            st.markdown("#### 🤖 Real-Data Flood Severity Forecaster")
            st.markdown("Select an official monitoring station to simulate flood progression and predict severity class.")
            
            stations_list = sorted(df_indo["Station"].dropna().unique())
            default_idx = stations_list.index("Biligundulu") if "Biligundulu" in stations_list else 0
            sel_station = st.selectbox("Select Gauge Station:", stations_list, index=default_idx)
            
            stn_row = df_indo[df_indo["Station"] == sel_station].iloc[0]
            
            w_lvl = float(stn_row["Warning Level"]) if pd.notnull(stn_row["Warning Level"]) else 10.0
            d_lvl = float(stn_row["Danger Level"]) if pd.notnull(stn_row["Danger Level"]) else 12.0
            c_area = float(stn_row["Catchment Area"]) if pd.notnull(stn_row["Catchment Area"]) else 5000.0
            
            st.caption(f"**River:** {stn_row['River Name/ Tributory/ SubTributory']} | **State:** {stn_row['State']} | **Warning:** {w_lvl}m | **Danger:** {d_lvl}m")
            
            sim_peak_lvl = st.number_input("Simulated Water Level (m)", min_value=1.0, max_value=2000.0, value=float(d_lvl + 0.8), step=0.1)
            sim_duration = st.slider("Flood Event Duration (days)", 1, 30, 4)
            sim_time_to_peak = st.slider("Time to Peak (days)", 1, sim_duration, max(1, sim_duration // 2))
            sim_recession = sim_duration - sim_time_to_peak + 1
            
            surge_above_danger = sim_peak_lvl - d_lvl
            ratio_to_danger = sim_peak_lvl / max(0.1, d_lvl)
            
            if indo_clf and indo_reg:
                indo_input = pd.DataFrame([{
                    "Peak Flood Level (m)": sim_peak_lvl,
                    "Warning Level": w_lvl,
                    "Danger Level": d_lvl,
                    "Gauge_Danger_Surge_m": surge_above_danger,
                    "Water_to_Danger_Ratio": ratio_to_danger,
                    "Event Duration (days)": sim_duration,
                    "Time to Peak (days)": sim_time_to_peak,
                    "Recession Time (day)": sim_recession,
                    "Num Peak FL": 1,
                    "Catchment Area": c_area
                }])
                
                pred_severe = indo_clf.predict(indo_input)[0]
                pred_severe_prob = indo_clf.predict_proba(indo_input)[0][1]
                
                reg_input = pd.DataFrame([{
                    "Peak Flood Level (m)": sim_peak_lvl,
                    "Warning Level": w_lvl,
                    "Danger Level": d_lvl,
                    "Gauge_Danger_Surge_m": surge_above_danger,
                    "Event Duration (days)": sim_duration,
                    "Catchment Area": c_area
                }])
                est_discharge = np.expm1(indo_reg.predict(reg_input)[0])
                
                if pred_severe == 1:
                    st.error(f"🚨 **SEVERE FLOOD ALERT** (AI Confidence: {pred_severe_prob*100:.1f}%)")
                    st.markdown(f"> **Surge:** {surge_above_danger:+.2f}m relative to danger mark. Peak river discharge forecasted at **{est_discharge:,.1f} cumec**.")
                else:
                    st.warning(f"⚠️ **STANDARD FLOOD WATCH** (Severe Risk: {pred_severe_prob*100:.1f}%)")
                    st.markdown(f"> **Surge:** {surge_above_danger:+.2f}m relative to danger mark. Peak river discharge forecasted at **{est_discharge:,.1f} cumec**.")
            else:
                st.info("INDOFLOODS models are training or loading...")
                
        # Government mitigation projects overview
        if df_projects is not None:
            st.divider()
            st.markdown("#### 🏛️ National Flood Management & Mitigation Outlay")
            proj_chart_df = df_projects[df_projects["State/UT"] != "Total"].copy()
            proj_chart_df["Funding_Cr"] = pd.to_numeric(proj_chart_df["Total CA Released Since Inception upto March 2025 (Rs. in Crore)"], errors="coerce").fillna(0)
            
            top_funded = proj_chart_df.sort_values(by="Funding_Cr", ascending=False).head(10)
            fig_proj = px.bar(
                top_funded,
                x="State/UT",
                y="Funding_Cr",
                color="Funding_Cr",
                color_continuous_scale="Tealgrn",
                labels={"Funding_Cr": "Funds Released (₹ Crore)", "State/UT": "State"},
                title="Top 10 States by Central Assistance Flood Management Funds (₹ Crore)"
            )
            fig_proj.update_layout(height=350, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_proj, use_container_width=True)
    else:
        st.info("INDOFLOODS dataset not loaded.")

# ----------------- TAB 1: LIVE SIMULATOR -----------------
with tab_sim:
    st.markdown("### 🎛️ Real-Time Telemetry & Risk Simulator")
    st.markdown("Adjust meteorological sensors below or choose a pre-configured scenario to simulate AI predictions.")
    
    # Presets
    scenario = st.selectbox(
        "⚡ Choose a Quick Scenario Preset:",
        [
            "Custom (Use Sliders Below)",
            "Dry Season Baseline (Normal)",
            "Moderate Monsoon Rain (Advisory Watch)",
            "Heavy Downpour & Saturated Catchment (Warning)",
            "Extreme Cloudburst & Reservoir Spillway Surge (Emergency)"
        ]
    )
    
    # Set default slider values based on preset
    if scenario == "Dry Season Baseline (Normal)":
        p_rain24, p_rain72, p_soil, p_discharge, p_level, p_reservoir = 4.0, 10.0, 32.0, 45.0, 2.5, 40.0
    elif scenario == "Moderate Monsoon Rain (Advisory Watch)":
        p_rain24, p_rain72, p_soil, p_discharge, p_level, p_reservoir = 65.0, 110.0, 75.0, 130.0, 4.8, 68.0
    elif scenario == "Heavy Downpour & Saturated Catchment (Warning)":
        p_rain24, p_rain72, p_soil, p_discharge, p_level, p_reservoir = 120.0, 240.0, 88.0, 220.0, 7.3, 86.0
    elif scenario == "Extreme Cloudburst & Reservoir Spillway Surge (Emergency)":
        p_rain24, p_rain72, p_soil, p_discharge, p_level, p_reservoir = 210.0, 390.0, 96.0, 340.0, 8.8, 97.0
    else: # Custom
        p_rain24, p_rain72, p_soil, p_discharge, p_level, p_reservoir = 45.0, 80.0, 58.0, 95.0, 3.8, 60.0
        
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 🌧️ Precipitation & Moisture")
        rain_24h = st.slider("24h Rainfall (mm)", 0.0, 300.0, float(p_rain24), 1.0)
        rain_72h = st.slider("72h Antecedent Rainfall (mm)", 0.0, 500.0, float(p_rain72), 2.0)
        soil_moisture = st.slider("Soil Moisture Saturation (%)", 5.0, 100.0, float(p_soil), 1.0)
        
    with col2:
        st.markdown("##### 🌊 River & Hydraulic Dynamics")
        current_water_level = st.slider("Current Water Gauge Level (m)", 1.0, 14.0, float(p_level), 0.1)
        river_discharge = st.slider("River Discharge (m³/s)", 10.0, 450.0, float(p_discharge), 5.0)
        reservoir_capacity = st.slider("Upstream Reservoir Capacity (%)", 10.0, 100.0, float(p_reservoir), 1.0)
        
    with col3:
        st.markdown("##### 🏔️ Basin Topography & Infrastructure")
        elevation = st.slider("Monitoring Station Elevation (m)", 40, 650, 310, 10)
        twi = st.slider("Topographic Wetness Index (TWI)", 5.0, 15.0, 9.2, 0.1)
        drainage = st.slider("Drainage Channel Capacity Score (0-1)", 0.1, 1.0, 0.70, 0.05)
        urbanization = st.slider("Urbanization Impervious Ratio (0-1)", 0.1, 1.0, 0.55, 0.05)
        
    st.divider()
    
    if classifier and regressor:
        # Construct feature vector
        sample_dict = {
            "elevation_m": [elevation],
            "rainfall_24h_mm": [rain_24h],
            "rainfall_72h_mm": [rain_72h],
            "soil_moisture_pct": [soil_moisture],
            "river_discharge_m3s": [river_discharge],
            "current_water_level_m": [current_water_level],
            "reservoir_capacity_pct": [reservoir_capacity],
            "topographic_wetness_index": [twi],
            "drainage_efficiency_score": [drainage],
            "urbanization_ratio": [urbanization]
        }
        sample_df = pd.DataFrame(sample_dict)
        sample_features = engineer_features(sample_df)[FEATURE_COLUMNS]
        
        # Inference
        pred_class = int(classifier.predict(sample_features)[0])
        pred_probs = classifier.predict_proba(sample_features)[0]
        pred_peak_level = float(regressor.predict(sample_features)[0])
        
        meta = get_risk_meta(pred_class)
        
        # Display Prediction Cards
        res_col1, res_col2 = st.columns([1.1, 0.9])
        
        with res_col1:
            st.markdown(f"""
            <div style="background-color: {meta['bg_color']}; border-left: 8px solid {meta['color']}; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.4rem; font-weight: 700; color: {meta['color']};">{meta['title']}</span>
                    <span style="background-color: {meta['color']}; color: white; padding: 4px 14px; border-radius: 20px; font-weight: 700;">{meta['badge']}</span>
                </div>
                <p style="margin-top: 10px; font-size: 1.05rem; color: #1E293B;">{meta['desc']}</p>
                <div style="margin-top: 12px; font-weight: 600; color: #334155;">Civil Protection Protocol:</div>
                <ul style="margin-top: 6px; color: #334155;">
                    {''.join(f'<li>{action}</li>' for action in meta['actions'])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability Distribution Chart
            prob_df = pd.DataFrame({
                "Risk Level": [RISK_LABELS[i] for i in range(4)],
                "Probability": [p * 100 for p in pred_probs]
            })
            fig_prob = px.bar(
                prob_df,
                x="Probability",
                y="Risk Level",
                orientation="h",
                text="Probability",
                color="Risk Level",
                color_discrete_map={
                    RISK_LABELS[0]: "#10B981",
                    RISK_LABELS[1]: "#F59E0B",
                    RISK_LABELS[2]: "#F97316",
                    RISK_LABELS[3]: "#EF4444"
                },
                title="AI Risk Confidence Distribution (%)"
            )
            fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_prob.update_layout(height=240, margin=dict(l=10, r=10, t=35, b=10), showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)
            
        with res_col2:
            # Peak Level Gauge / Comparison
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred_peak_level,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Peak Gauge (Next 24h)", 'font': {'size': 18}},
                delta={'reference': current_water_level, 'increasing': {'color': "#EF4444"}, 'suffix': "m surge"},
                gauge={
                    'axis': {'range': [None, 12], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': meta['color']},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 5], 'color': '#D1FAE5'},
                        {'range': [5, 7.5], 'color': '#FEF3C7'},
                        {'range': [7.5, 9.5], 'color': '#FFEDD5'},
                        {'range': [9.5, 12], 'color': '#FEE2E2'},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 9.5
                    }
                }
            ))
            gauge_fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Hydrological Derived Metrics
            st.markdown("##### 🔍 Hydrological Stress Indicators")
            m1, m2, m3 = st.columns(3)
            m1.metric("Runoff Risk Index", f"{sample_features['runoff_risk_ratio'].values[0]:.1f}")
            m2.metric("Dam Stress Score", f"{sample_features['capacity_stress'].values[0]:.1f}")
            m3.metric("Hydraulic Flow Index", f"{sample_features['hazard_flow_index'].values[0]:.1f}")
            
    else:
        st.warning("Please train the model first to see live predictions.")

# ----------------- TAB 2: BASIN TELEMETRY MAP -----------------
with tab_map:
    st.markdown("### 🗺️ River Catchment Basin Monitoring Stations")
    st.markdown("Live spatial overview of river stations, gauge heights, and risk categories.")
    
    if df_data is not None:
        # Get latest reading for each station
        latest_stations = df_data.groupby("station_name").last().reset_index()
        
        # Assign color based on risk level
        color_map = {0: "#10B981", 1: "#F59E0B", 2: "#F97316", 3: "#EF4444"}
        latest_stations["Risk Label"] = latest_stations["flood_risk_level"].map(RISK_LABELS)
        
        map_fig = px.scatter_mapbox(
            latest_stations,
            lat="station_latitude",
            lon="station_longitude",
            color="Risk Label",
            size="current_water_level_m",
            size_max=22,
            hover_name="station_name",
            hover_data={
                "elevation_m": True,
                "current_water_level_m": ":.2f m",
                "rainfall_24h_mm": ":.1f mm",
                "river_discharge_m3s": ":.1f m³/s",
                "station_latitude": False,
                "station_longitude": False
            },
            color_discrete_map={
                RISK_LABELS[0]: "#10B981",
                RISK_LABELS[1]: "#F59E0B",
                RISK_LABELS[2]: "#F97316",
                RISK_LABELS[3]: "#EF4444"
            },
            zoom=10,
            center={"lat": 18.5089, "lon": 73.8567},
            mapbox_style="carto-positron",
            title="River Catchment Sensor Nodes (Dot Size = Water Gauge Height)"
        )
        map_fig.update_layout(height=480, margin=dict(l=0, r=0, t=35, b=0))
        st.plotly_chart(map_fig, use_container_width=True)
        
        st.markdown("#### 📋 Current Station Telemetry Table")
        st.dataframe(
            latest_stations[[
                "station_name", "elevation_m", "current_water_level_m",
                "forecasted_peak_water_level_m", "rainfall_24h_mm", "soil_moisture_pct",
                "river_discharge_m3s", "Risk Label"
            ]],
            use_container_width=True
        )
    else:
        st.info("Dataset not loaded.")

# ----------------- TAB 3: HYDROLOGICAL TRENDS -----------------
with tab_analytics:
    st.markdown("### 📈 Time-Series & Catchment Hydrology Trends")
    
    if df_data is not None:
        station_select = st.selectbox("Select Station for Time-Series Analysis:", df_data["station_name"].unique())
        station_df = df_data[df_data["station_name"] == station_select].head(120).copy()
        station_df["timestamp"] = pd.to_datetime(station_df["timestamp"])
        
        # Time series chart: Rainfall vs Water Level
        ts_fig = go.Figure()
        ts_fig.add_trace(go.Bar(
            x=station_df["timestamp"],
            y=station_df["rainfall_24h_mm"],
            name="24h Rainfall (mm)",
            marker_color="rgba(59, 130, 246, 0.6)",
            yaxis="y2"
        ))
        ts_fig.add_trace(go.Scatter(
            x=station_df["timestamp"],
            y=station_df["current_water_level_m"],
            name="Current Water Level (m)",
            line=dict(color="#10B981", width=2.5)
        ))
        ts_fig.add_trace(go.Scatter(
            x=station_df["timestamp"],
            y=station_df["forecasted_peak_water_level_m"],
            name="Peak Forecast (m)",
            line=dict(color="#EF4444", width=2, dash="dash")
        ))
        
        # Danger Threshold line
        ts_fig.add_hline(y=7.5, line_dash="dot", line_color="orange", annotation_text="Warning Threshold (7.5m)")
        ts_fig.add_hline(y=9.5, line_dash="dot", line_color="red", annotation_text="Emergency Breach (9.5m)")
        
        ts_fig.update_layout(
            title=f"Telemetry Stream: {station_select}",
            xaxis_title="Timeline",
            yaxis=dict(title="Water Level (meters)"),
            yaxis2=dict(title="Precipitation (mm)", overlaying="y", side="right", showgrid=False),
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(ts_fig, use_container_width=True)
        
        # Scatter correlation
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_corr1 = px.scatter(
                df_data,
                x="rainfall_24h_mm",
                y="forecasted_peak_water_level_m",
                color="flood_risk_level",
                color_continuous_scale="Viridis",
                labels={"rainfall_24h_mm": "24h Rainfall (mm)", "forecasted_peak_water_level_m": "Peak Water Level (m)"},
                title="Rainfall Intensity vs. Peak Water Level"
            )
            st.plotly_chart(fig_corr1, use_container_width=True)
            
        with col_c2:
            fig_corr2 = px.scatter(
                df_data,
                x="soil_moisture_pct",
                y="river_discharge_m3s",
                color="current_water_level_m",
                color_continuous_scale="Plasma",
                labels={"soil_moisture_pct": "Soil Moisture (%)", "river_discharge_m3s": "River Discharge (m³/s)"},
                title="Soil Saturation vs. River Discharge Rate"
            )
            st.plotly_chart(fig_corr2, use_container_width=True)
    else:
        st.info("Dataset not loaded.")

# ----------------- TAB 4: MODEL PERFORMANCE -----------------
with tab_perf:
    st.markdown("### 🤖 Model Architecture, Benchmarking & Explainability")
    
    if metadata is not None:
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric("Classification Accuracy", f"{metadata['classifier']['accuracy']*100:.1f}%")
        col_p2.metric("Classifier Weighted F1", f"{metadata['classifier']['f1_weighted']:.3f}")
        col_p3.metric("Regressor R² Score", f"{metadata['regressor']['r2_score']:.3f}")
        col_p4.metric("Regressor Mean Absolute Error", f"{metadata['regressor']['mae']:.3f} m")
        
        st.divider()
        
        col_imp, col_cm = st.columns([1.1, 0.9])
        
        with col_imp:
            st.markdown("#### 🎯 Random Forest Feature Importance")
            feat_df = pd.DataFrame(list(metadata["classifier"]["feature_importances"].items()), columns=["Feature", "Importance"])
            feat_df = feat_df.sort_values(by="Importance", ascending=True)
            
            fig_feat = px.bar(
                feat_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Teal",
                title="Feature Influence on Flood Alert Prediction"
            )
            fig_feat.update_layout(height=420, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_feat, use_container_width=True)
            
        with col_cm:
            st.markdown("#### 🧮 Confusion Matrix")
            cm_matrix = np.array(metadata["classifier"]["confusion_matrix"])
            labels = [RISK_LABELS[i] for i in range(4)]
            
            fig_cm = px.imshow(
                cm_matrix,
                labels=dict(x="Predicted Risk Level", y="Actual Ground Truth", color="Count"),
                x=labels,
                y=labels,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Risk Classification Confusion Matrix"
            )
            fig_cm.update_layout(height=420, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)
            
    else:
        st.info("Train the model pipeline to view evaluation metrics.")

# ----------------- TAB 5: BATCH ASSESSMENT -----------------
with tab_batch:
    st.markdown("### 📁 Batch Telemetry Ingestion & Risk Scoring")
    st.markdown("Upload telemetry CSV files to calculate bulk risk classifications and forecasted peak water levels.")
    
    sample_download = df_data.head(10)[[
        "elevation_m", "rainfall_24h_mm", "rainfall_72h_mm",
        "soil_moisture_pct", "river_discharge_m3s", "current_water_level_m",
        "reservoir_capacity_pct", "topographic_wetness_index",
        "drainage_efficiency_score", "urbanization_ratio"
    ]] if df_data is not None else None
    
    if sample_download is not None:
        st.download_button(
            label="📥 Download Sample Batch Template (CSV)",
            data=sample_download.to_csv(index=False).encode('utf-8'),
            file_name="sample_telemetry_template.csv",
            mime="text/csv"
        )
        
    uploaded_file = st.file_uploader("Upload CSV Telemetry Data", type=["csv"])
    
    if uploaded_file is not None and classifier and regressor:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", batch_df.head())
        
        # Verify required columns
        req_cols = [
            "elevation_m", "rainfall_24h_mm", "rainfall_72h_mm",
            "soil_moisture_pct", "river_discharge_m3s", "current_water_level_m",
            "reservoir_capacity_pct", "topographic_wetness_index",
            "drainage_efficiency_score", "urbanization_ratio"
        ]
        
        missing = [c for c in req_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Missing required columns in CSV: {missing}")
        else:
            with st.spinner("Processing AI risk inference..."):
                proc_df = engineer_features(batch_df)
                batch_features = proc_df[FEATURE_COLUMNS]
                
                batch_df["Predicted_Risk_Level"] = classifier.predict(batch_features)
                batch_df["Risk_Category"] = batch_df["Predicted_Risk_Level"].map(RISK_LABELS)
                batch_df["Forecasted_Peak_Level_m"] = regressor.predict(batch_features).round(2)
                
            st.success(f"Batch scoring completed for {len(batch_df)} records!")
            
            # Summary metric distribution
            st.write("### Assessment Summary:")
            st.bar_chart(batch_df["Risk_Category"].value_counts())
            
            st.dataframe(batch_df, use_container_width=True)
            
            csv_result = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Scored Telemetry CSV",
                data=csv_result,
                file_name="floodguard_scored_telemetry.csv",
                mime="text/csv"
            )
