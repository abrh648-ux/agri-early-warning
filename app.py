# =====================================================
# AGRICULTURAL EARLY WARNING SYSTEM
# Ethiopia - Food Security Risk Prediction
# Inputs: Year, Region, Crop Type only
# All other features auto-filled from historical data
# =====================================================

import subprocess
import sys

# Auto-install all required packages if missing
required = [
    "joblib", "shap==0.51.0", "matplotlib", "numpy",
    "pandas", "plotly", "scikit-learn", "seaborn", "xgboost"
]
for pkg in required:
    try:
        __import__(pkg.split("==")[0].split(">=")[0])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import json
import seaborn as sns

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Agricultural Early Warning System",
    page_icon="🌾",
    layout="wide"
)

# -------------------------------------------------------
# LOAD DATA & MODELS (cached)
# -------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("labeled_agri_risk_data.csv")

@st.cache_data
def load_xtest():
    return pd.read_csv("X_test.csv")

@st.cache_data
def load_geojson():
    with open("ethiopia_regions.geojson") as f:
        return json.load(f)

@st.cache_resource
def load_models():
    xgb = joblib.load("xgboost_model.pkl")
    rf  = joblib.load("random_forest_model.pkl")
    return xgb, rf

df            = load_data()
X_test        = load_xtest()
geojson       = load_geojson()
xgb_model, rf_model = load_models()

# -------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------
MODEL_FEATURES = [
    'Region_Code', 'Crop_Code', 'Year',
    'Area cultivated(Ha)', 'Production(kg)',
    'Yield_Growth_Rate', 'Production_Growth_Rate',
    'Area_Efficiency', 'Regional_Avg_Yield', 'Crop_Avg_Yield',
    'Yield_Anomaly', 'Rolling_Yield_Trend', 'Yield_Stability',
    'Production_Area_Ratio', 'Early_Warning_Score'
]

RISK_NAMES  = {0: "Low Risk",  1: "Medium Risk", 2: "High Risk"}
RISK_COLORS = {0: "#28a745",   1: "#fd7e14",     2: "#dc3545"}
RISK_EMOJI  = {0: "🟢",        1: "🟡",           2: "🔴"}

# -------------------------------------------------------
# ENCODERS
# -------------------------------------------------------
region_enc = LabelEncoder()
crop_enc   = LabelEncoder()
region_enc.fit(sorted(df["Region"].dropna().unique()))
crop_enc.fit(sorted(df["crop type"].dropna().unique()))

# -------------------------------------------------------
# HELPER: auto-fill features from historical data
# User provides only: year, region, crop
# -------------------------------------------------------
def build_input(year, region, crop):
    r_code = int(region_enc.transform([region])[0])
    c_code = int(crop_enc.transform([crop])[0])

    # Find the closest historical row for this region+crop
    sub = df[(df["Region"] == region) & (df["crop type"] == crop)].copy()
    if sub.empty:
        sub = df[df["Region"] == region].copy()
    if sub.empty:
        sub = df.copy()

    sub = sub.copy()
    sub["_diff"] = (sub["Year"] - year).abs()
    ref = sub.sort_values("_diff").iloc[0]

    row = {}
    for feat in MODEL_FEATURES:
        if feat == "Region_Code":
            row[feat] = r_code
        elif feat == "Crop_Code":
            row[feat] = c_code
        elif feat == "Year":
            row[feat] = year
        else:
            val = ref.get(feat, np.nan)
            row[feat] = val if pd.notna(val) and not np.isinf(val) \
                        else float(df[feat].replace([np.inf, -np.inf], np.nan).median())

    inp = pd.DataFrame([row], columns=MODEL_FEATURES)
    inp.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in MODEL_FEATURES:
        if inp[c].isnull().any():
            inp[c] = float(df[c].replace([np.inf, -np.inf], np.nan).median())
    return inp


# -------------------------------------------------------
# HELPER: predict for every region × crop for a given year
# -------------------------------------------------------
@st.cache_data
def predict_all_for_year(year):
    rows = []
    for reg in sorted(df["Region"].dropna().unique()):
        for crop in sorted(df["crop type"].dropna().unique()):
            inp      = build_input(year, reg, crop)
            xgb_pred = int(xgb_model.predict(inp)[0])
            rf_pred  = int(rf_model.predict(inp)[0])
            rows.append({
                "Region":        reg,
                "Crop":          crop,
                "Year":          year,
                "XGB_Risk":      xgb_pred,
                "RF_Risk":       rf_pred,
                "Risk_Level":    xgb_pred,
                "Risk_Category": RISK_NAMES[xgb_pred],
            })
    return pd.DataFrame(rows)


# -------------------------------------------------------
# SIDEBAR — 3 inputs + predict button
# -------------------------------------------------------
st.sidebar.title("🌾 Agricultural Early Warning System")
st.sidebar.markdown("---")
st.sidebar.header("📥 Enter Prediction Inputs")

sel_year   = st.sidebar.selectbox(
    "📅 Year",
    sorted(df["Year"].dropna().unique(), reverse=True)
)
sel_region = st.sidebar.selectbox(
    "📍 Region",
    sorted(df["Region"].dropna().unique())
)
sel_crop   = st.sidebar.selectbox(
    "🌱 Crop Type",
    sorted(df["crop type"].dropna().unique())
)

predict_btn = st.sidebar.button("▶ Predict", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "**How to use:**\n"
    "1. Select Year, Region and Crop\n"
    "2. Click ▶ Predict\n"
    "3. All pages update automatically"
)

# Run prediction when button clicked OR on first load
if predict_btn or "predicted" not in st.session_state:
    with st.spinner("Running predictions..."):
        inp       = build_input(int(sel_year), sel_region, sel_crop)
        xgb_pred  = int(xgb_model.predict(inp)[0])
        rf_pred   = int(rf_model.predict(inp)[0])
        all_preds = predict_all_for_year(int(sel_year))

        st.session_state["predicted"]  = True
        st.session_state["inp"]        = inp
        st.session_state["xgb_pred"]   = xgb_pred
        st.session_state["rf_pred"]    = rf_pred
        st.session_state["all_preds"]  = all_preds
        st.session_state["sel_year"]   = int(sel_year)
        st.session_state["sel_region"] = sel_region
        st.session_state["sel_crop"]   = sel_crop

# Pull from session state
inp        = st.session_state["inp"]
xgb_pred   = st.session_state["xgb_pred"]
rf_pred    = st.session_state["rf_pred"]
all_preds  = st.session_state["all_preds"]
cur_year   = st.session_state["sel_year"]
cur_region = st.session_state["sel_region"]
cur_crop   = st.session_state["sel_crop"]

# -------------------------------------------------------
# NAVIGATION
# -------------------------------------------------------
st.sidebar.markdown("---")
page = st.sidebar.radio("🗂 Navigate", [
    "🏠 Home",
    "📂 Dataset",
    "🤖 Prediction",
    "🔍 Explainable AI",
    "📊 Risk Analysis",
    "🗺️ Ethiopia Risk Map",
    "ℹ️ About"
])

# =====================================================
# HOME PAGE
# =====================================================
if page == "🏠 Home":
    st.title("🌾 Agricultural Early Warning System")
    st.subheader("Explainable AI-Based Food Security Risk Mapping — Ethiopia")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.info(f"📅 Year: **{cur_year}**")
    c2.info(f"📍 Region: **{cur_region}**")
    c3.info(f"🌱 Crop: **{cur_crop}**")

    st.markdown("---")
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div style='background:{RISK_COLORS[xgb_pred]};padding:24px;"
            f"border-radius:10px;color:white;text-align:center;'>"
            f"<h3>🚀 XGBoost</h3>"
            f"<h2>{RISK_EMOJI[xgb_pred]} {RISK_NAMES[xgb_pred]}</h2></div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"<div style='background:{RISK_COLORS[rf_pred]};padding:24px;"
            f"border-radius:10px;color:white;text-align:center;'>"
            f"<h3>🌳 Random Forest</h3>"
            f"<h2>{RISK_EMOJI[rf_pred]} {RISK_NAMES[rf_pred]}</h2></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records",  len(df))
    m2.metric("Regions",        df["Region"].nunique())
    m3.metric("Crop Types",     df["crop type"].nunique())
    m4.metric("Years Covered",  df["Year"].nunique())

# =====================================================
# DATASET PAGE
# =====================================================
elif page == "📂 Dataset":
    st.title("📂 Agricultural Dataset")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(30), use_container_width=True)
    st.subheader("Statistical Summary")
    st.dataframe(df.describe().T, use_container_width=True)

# =====================================================
# PREDICTION PAGE
# =====================================================
elif page == "🤖 Prediction":
    st.title("🤖 Early Warning Prediction")
    st.write(
        f"Results for **{cur_region}** | **{cur_crop}** | **{cur_year}**"
    )
    st.caption("Change inputs in the sidebar and click ▶ Predict to update.")
    st.markdown("---")

    # Risk cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div style='background:{RISK_COLORS[xgb_pred]};padding:24px;"
            f"border-radius:10px;color:white;text-align:center;'>"
            f"<h3>🚀 XGBoost Prediction</h3>"
            f"<h2>{RISK_EMOJI[xgb_pred]} {RISK_NAMES[xgb_pred]}</h2></div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"<div style='background:{RISK_COLORS[rf_pred]};padding:24px;"
            f"border-radius:10px;color:white;text-align:center;'>"
            f"<h3>🌳 Random Forest Prediction</h3>"
            f"<h2>{RISK_EMOJI[rf_pred]} {RISK_NAMES[rf_pred]}</h2></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("📋 Auto-Filled Features Used")
    st.caption("These were automatically filled from historical data based on your inputs.")
    styled = inp.T.rename(columns={0: "Value"})
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")
    st.subheader(f"🌍 Risk for All Crops in {cur_region} ({cur_year})")
    reg_data = all_preds[all_preds["Region"] == cur_region]
    fig = px.bar(
        reg_data, x="Crop", y="Risk_Level", color="Risk_Category",
        color_discrete_map={
            "Low Risk": "#28a745",
            "Medium Risk": "#fd7e14",
            "High Risk": "#dc3545"
        },
        labels={"Risk_Level": "Risk Level (0=Low, 1=Med, 2=High)"},
        title=f"Predicted Risk by Crop — {cur_region} ({cur_year})"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# EXPLAINABLE AI PAGE
# Based on the prediction for selected Year/Region/Crop
# =====================================================
elif page == "🔍 Explainable AI":
    st.title("🔍 Explainable AI (SHAP)")
    st.write(
        f"Explaining the prediction for **{cur_region}** | "
        f"**{cur_crop}** | **{cur_year}**"
    )
    st.markdown("---")

    explainer = shap.TreeExplainer(xgb_model)

    # Waterfall plot for this specific prediction
    st.subheader(f"Why is {cur_region} — {cur_crop} predicted as {RISK_NAMES[xgb_pred]}?")
    shap_vals = explainer.shap_values(inp)

    if isinstance(shap_vals, list):
        sv = shap_vals[xgb_pred][0]
        ev = explainer.expected_value[xgb_pred]
    else:
        sv = shap_vals[0] if shap_vals.ndim > 1 else shap_vals
        ev = float(explainer.expected_value)

    explanation = shap.Explanation(
        values=sv,
        base_values=ev,
        data=inp.values[0],
        feature_names=list(inp.columns)
    )

    fig1, _ = plt.subplots()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig1, bbox_inches="tight")
    plt.close(fig1)

    st.markdown("---")

    # Global feature importance from test set
    st.subheader("Global Feature Importance (across all test predictions)")
    sample = X_test.sample(min(300, len(X_test)), random_state=42)
    shap_global = explainer.shap_values(sample)

    fig2, _ = plt.subplots()
    shap.summary_plot(shap_global, sample, plot_type="bar", show=False)
    st.pyplot(fig2)
    plt.close(fig2)

# =====================================================
# RISK ANALYSIS PAGE
# Based on all_preds for selected year
# =====================================================
elif page == "📊 Risk Analysis":
    st.title("📊 Risk Analysis")
    st.write(f"Analysis for year **{cur_year}** — all regions and crops")
    st.markdown("---")

    # Regional average risk
    reg_summary = (
        all_preds.groupby("Region")["Risk_Level"]
        .mean().reset_index()
        .rename(columns={"Risk_Level": "Avg_Risk"})
    )
    reg_summary["Risk_Category"] = reg_summary["Avg_Risk"].apply(
        lambda x: "High Risk" if x >= 1.5 else ("Medium Risk" if x >= 0.5 else "Low Risk")
    )

    st.subheader(f"Regional Risk Summary — {cur_year}")
    st.dataframe(reg_summary.sort_values("Avg_Risk", ascending=False),
                 use_container_width=True)

    fig = px.bar(
        reg_summary, x="Region", y="Avg_Risk", color="Risk_Category",
        color_discrete_map={
            "Low Risk": "#28a745",
            "Medium Risk": "#fd7e14",
            "High Risk": "#dc3545"
        },
        title=f"Average Risk Score by Region — {cur_year}",
        labels={"Avg_Risk": "Average Risk (0=Low, 2=High)"}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Heatmap: region × crop
    st.subheader(f"Risk Heatmap: Region × Crop — {cur_year}")
    pivot = all_preds.pivot_table(
        values="Risk_Level", index="Region", columns="Crop", aggfunc="mean"
    )
    fig2, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, vmin=0, vmax=2,
        cbar_kws={"label": "Risk Level"}
    )
    ax.set_title(f"Risk Level — {cur_year}  (0=Low  1=Medium  2=High)")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown("---")

    # High risk table
    st.subheader("🔴 High Risk Combinations")
    high = all_preds[all_preds["Risk_Category"] == "High Risk"][
        ["Region", "Crop", "Year", "Risk_Category"]
    ]
    if not high.empty:
        st.dataframe(high, use_container_width=True)
    else:
        st.success(f"No High Risk combinations found for {cur_year}.")

# =====================================================
# ETHIOPIA RISK MAP
# Based on all_preds for selected year
# =====================================================
elif page == "🗺️ Ethiopia Risk Map":
    st.title("🗺️ Ethiopia Food Security Risk Map")
    st.write(f"Average predicted risk across all crops — **{cur_year}**")
    st.markdown("---")

    reg_summary = (
        all_preds.groupby("Region")["Risk_Level"]
        .mean().reset_index()
        .rename(columns={"Risk_Level": "Avg_Risk"})
    )
    reg_summary["Risk_Category"] = reg_summary["Avg_Risk"].apply(
        lambda x: "High Risk" if x >= 1.5 else ("Medium Risk" if x >= 0.5 else "Low Risk")
    )

    fig = px.choropleth_mapbox(
        reg_summary,
        geojson=geojson,
        featureidkey="properties.ADM1_EN",
        locations="Region",
        color="Risk_Category",
        color_discrete_map={
            "Low Risk":    "#28a745",
            "Medium Risk": "#fd7e14",
            "High Risk":   "#dc3545"
        },
        hover_data={"Avg_Risk": ":.2f"},
        mapbox_style="carto-positron",
        zoom=5,
        center={"lat": 9.145, "lon": 40.4897},
        opacity=0.7,
        title=f"Food Security Risk Map — {cur_year}"
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Regional Risk Table")
    st.dataframe(
        reg_summary.sort_values("Avg_Risk", ascending=False),
        use_container_width=True
    )

# =====================================================
# ABOUT PAGE
# =====================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About")
    st.markdown(f"""
### Agricultural Early Warning System

**Purpose:** Predict food security risk across Ethiopian regions using machine learning.

**How it works:**
- You enter only 3 inputs: **Year**, **Region**, **Crop Type**
- The system automatically fills all other features from historical data
- Two models predict the risk: **XGBoost** (primary) and **Random Forest**
- All pages — Explainable AI, Risk Analysis, Ethiopia Risk Map — are based on these predictions

**Risk Levels:**
| Level | Label | Meaning |
|-------|-------|---------|
| 🟢 0 | Low Risk | Yield near or above historical average |
| 🟡 1 | Medium Risk | Yield moderately below average |
| 🔴 2 | High Risk | Yield significantly below average |

**Data:** Ethiopian crop production statistics 1996–2022  
**Regions covered:** {df['Region'].nunique()}  
**Crops covered:** {df['crop type'].nunique()}  
**Total records:** {len(df):,}
""")
