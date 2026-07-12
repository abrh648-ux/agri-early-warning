# =====================================================
# AGRICULTURAL EARLY WARNING SYSTEM
# Explainable AI-Based Food Security Risk Mapping
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import io
from contextlib import redirect_stdout
import json
import seaborn as sns

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Agricultural Early Warning System",
    page_icon="🌾",
    layout="wide",
)

# =====================================================
# LOAD DATA & MODELS
# =====================================================
@st.cache_data
def load_dataset():
    return pd.read_csv("labeled_agri_risk_data.csv")

@st.cache_data
def load_X_test():
    return pd.read_csv("X_test.csv")

@st.cache_data
def load_geojson():
    with open('ethiopia_regions.geojson') as f:
        return json.load(f)

@st.cache_resource
def load_xgboost():
    return joblib.load("xgboost_model.pkl")

@st.cache_resource
def load_random_forest():
    return joblib.load("random_forest_model.pkl")

df         = load_dataset()
X_test     = load_X_test()
geojson    = load_geojson()
xgb_model  = load_xgboost()
rf_model   = load_random_forest()

# =====================================================
# CONSTANTS
# =====================================================
MODEL_FEATURES = [
    'Region_Code', 'Crop_Code', 'Year', 'Area cultivated(Ha)', 'Production(kg)',
    'Yield_Growth_Rate', 'Production_Growth_Rate', 'Area_Efficiency',
    'Regional_Avg_Yield', 'Crop_Avg_Yield', 'Yield_Anomaly',
    'Rolling_Yield_Trend', 'Yield_Stability', 'Production_Area_Ratio',
    'Early_Warning_Score'
]
RISK_NAMES  = {0: "Low Risk",    1: "Medium Risk", 2: "High Risk"}
RISK_COLORS = {0: "#28a745",     1: "#fd7e14",     2: "#dc3545"}
RISK_EMOJI  = {0: "🟢",          1: "🟡",           2: "🔴"}

# Fit encoders once on full dataset
region_encoder = LabelEncoder()
crop_encoder   = LabelEncoder()
region_encoder.fit(sorted(df['Region'].dropna().unique()))
crop_encoder.fit(sorted(df['crop type'].dropna().unique()))

# =====================================================
# HELPER: build one input row from Year/Region/Crop
# =====================================================
def build_input_row(year: int, region: str, crop: str) -> pd.DataFrame:
    """
    Look up the closest historical row for this Region+Crop,
    then override Year-dependent fields.  All engineered features
    are taken from that historical row so the user never has to
    enter them manually.
    """
    region_code = int(region_encoder.transform([region])[0])
    crop_code   = int(crop_encoder.transform([crop])[0])

    # Find closest year match for this region+crop combination
    subset = df[(df['Region'] == region) & (df['crop type'] == crop)].copy()
    if subset.empty:
        # Fall back to region-only medians
        subset = df[df['Region'] == region].copy()
    if subset.empty:
        subset = df.copy()

    subset['year_diff'] = (subset['Year'] - year).abs()
    closest = subset.sort_values('year_diff').iloc[0]

    row = {}
    for feat in MODEL_FEATURES:
        if feat == 'Region_Code':
            row[feat] = region_code
        elif feat == 'Crop_Code':
            row[feat] = crop_code
        elif feat == 'Year':
            row[feat] = year
        else:
            val = closest.get(feat, np.nan)
            row[feat] = val if pd.notna(val) else float(df[feat].median())

    input_df = pd.DataFrame([row], columns=MODEL_FEATURES)
    input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in MODEL_FEATURES:
        if input_df[col].isnull().any():
            input_df[col] = input_df[col].fillna(float(df[col].median()))
    return input_df

# =====================================================
# HELPER: run prediction for every region × crop
# =====================================================
@st.cache_data
def predict_all(year: int) -> pd.DataFrame:
    """Predict risk for all Region×Crop combos for a given year."""
    rows = []
    for region in sorted(df['Region'].dropna().unique()):
        for crop in sorted(df['crop type'].dropna().unique()):
            input_row = build_input_row(year, region, crop)
            pred_xgb = int(xgb_model.predict(input_row)[0])
            pred_rf  = int(rf_model.predict(input_row)[0])
            rows.append({
                'Region':       region,
                'Crop':         crop,
                'Year':         year,
                'XGB_Risk':     pred_xgb,
                'RF_Risk':      pred_rf,
                'Risk_Level':   pred_xgb,          # primary = XGBoost
                'Risk_Category': RISK_NAMES[pred_xgb],
            })
    return pd.DataFrame(rows)

# =====================================================
# SIDEBAR — shared inputs drive the whole app
# =====================================================
st.sidebar.title("🌾 Navigation")

page = st.sidebar.radio(
    "Choose a Page",
    (
        "🏠 Home",
        "📂 Dataset",
        "🤖 Early Warning Prediction",
        "🔍 Explainable AI",
        "📊 Risk Analysis",
        "🗺️ Ethiopia Risk Map",
        "ℹ️ About"
    )
)

st.sidebar.markdown("---")
st.sidebar.header("🔎 Prediction Inputs")

sel_year   = st.sidebar.selectbox(
    "Year",
    sorted(df['Year'].dropna().unique(), reverse=True),
    index=0
)
sel_region = st.sidebar.selectbox(
    "Region",
    sorted(df['Region'].dropna().unique())
)
sel_crop   = st.sidebar.selectbox(
    "Crop Type",
    sorted(df['crop type'].dropna().unique())
)

run_btn = st.sidebar.button("▶ Run Prediction", type="primary")

# Store prediction in session state so all pages can use it
if run_btn or 'prediction_done' not in st.session_state:
    input_row  = build_input_row(int(sel_year), sel_region, sel_crop)
    pred_xgb   = int(xgb_model.predict(input_row)[0])
    pred_rf    = int(rf_model.predict(input_row)[0])
    all_preds  = predict_all(int(sel_year))

    st.session_state['prediction_done'] = True
    st.session_state['input_row']       = input_row
    st.session_state['pred_xgb']        = pred_xgb
    st.session_state['pred_rf']         = pred_rf
    st.session_state['all_preds']       = all_preds
    st.session_state['sel_year']        = sel_year
    st.session_state['sel_region']      = sel_region
    st.session_state['sel_crop']        = sel_crop

# Convenience aliases
input_row = st.session_state['input_row']
pred_xgb  = st.session_state['pred_xgb']
pred_rf   = st.session_state['pred_rf']
all_preds = st.session_state['all_preds']

st.sidebar.markdown("---")
st.sidebar.info(
    "This app predicts food security risk using:\n"
    "- 🌳 Random Forest\n"
    "- 🚀 XGBoost\n"
    "- 🔍 SHAP Explainability\n\n"
    "Country: Ethiopia"
)

# =====================================================
# HOME PAGE
# =====================================================
if page == "🏠 Home":
    st.title("🌾 Agricultural Early Warning System")
    st.subheader("Explainable AI-Based Food Security Risk Mapping for Ethiopia")
    st.markdown("---")

    st.write("""
Select a **Year**, **Region**, and **Crop Type** in the sidebar and click
**▶ Run Prediction**. The result will automatically update all pages.
    """)

    st.markdown("### Project Objectives")
    st.markdown("""
- Predict agricultural food security risk from minimal inputs
- Compare XGBoost and Random Forest models
- Explain predictions using SHAP
- Visualize regional food security risk across Ethiopia
""")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records",    len(df))
    col2.metric("Regions",    df["Region"].nunique())
    col3.metric("Crop Types", df["crop type"].nunique())
    col4.metric("Years",      df["Year"].nunique())

    st.markdown("---")
    st.markdown("### Current Selection")
    c1, c2, c3 = st.columns(3)
    c1.info(f"📅 Year: **{st.session_state['sel_year']}**")
    c2.info(f"📍 Region: **{st.session_state['sel_region']}**")
    c3.info(f"🌱 Crop: **{st.session_state['sel_crop']}**")

    risk_color = RISK_COLORS[pred_xgb]
    risk_name  = RISK_NAMES[pred_xgb]
    st.markdown(
        f"<div style='background:{risk_color};padding:16px;border-radius:8px;color:white;text-align:center;'>"
        f"<h2>{RISK_EMOJI[pred_xgb]} XGBoost Prediction: {risk_name}</h2>"
        f"<h3>{RISK_EMOJI[pred_rf]} Random Forest Prediction: {RISK_NAMES[pred_rf]}</h3>"
        f"</div>",
        unsafe_allow_html=True
    )

# =====================================================
# DATASET PAGE
# =====================================================
elif page == "📂 Dataset":
    st.title("📂 Agricultural Dataset")
    st.write("Explore the dataset used for training.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe().T, use_container_width=True)

# =====================================================
# EARLY WARNING PREDICTION PAGE
# =====================================================
elif page == "🤖 Early Warning Prediction":
    st.title("🤖 Early Warning Prediction")
    st.write(
        f"Results for **{st.session_state['sel_region']}** | "
        f"**{st.session_state['sel_crop']}** | "
        f"**{st.session_state['sel_year']}**"
    )
    st.caption("Change the inputs in the sidebar and click ▶ Run Prediction to update.")

    st.markdown("---")

    # Risk result cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div style='background:{RISK_COLORS[pred_xgb]};padding:20px;"
            f"border-radius:10px;color:white;text-align:center;'>"
            f"<h3>🚀 XGBoost</h3>"
            f"<h2>{RISK_EMOJI[pred_xgb]} {RISK_NAMES[pred_xgb]}</h2>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"<div style='background:{RISK_COLORS[pred_rf]};padding:20px;"
            f"border-radius:10px;color:white;text-align:center;'>"
            f"<h3>🌳 Random Forest</h3>"
            f"<h2>{RISK_EMOJI[pred_rf]} {RISK_NAMES[pred_rf]}</h2>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("📋 Features Used for This Prediction")
    st.dataframe(input_row.T.rename(columns={0: "Value"}), use_container_width=True)

    st.markdown("---")
    st.subheader(f"🌍 All Crops Risk for {st.session_state['sel_region']} in {st.session_state['sel_year']}")
    region_preds = all_preds[all_preds['Region'] == st.session_state['sel_region']]
    fig = px.bar(
        region_preds, x='Crop', y='Risk_Level', color='Risk_Category',
        color_discrete_map={'Low Risk': '#28a745', 'Medium Risk': '#fd7e14', 'High Risk': '#dc3545'},
        title=f"Risk by Crop — {st.session_state['sel_region']} ({st.session_state['sel_year']})",
        labels={'Risk_Level': 'Risk Score (0=Low, 2=High)'}
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# EXPLAINABLE AI PAGE
# =====================================================
elif page == "🔍 Explainable AI":
    st.title("🔍 Explainable AI (SHAP)")
    st.write(
        f"Explaining the XGBoost prediction for **{st.session_state['sel_region']}** | "
        f"**{st.session_state['sel_crop']}** | **{st.session_state['sel_year']}**"
    )

    explainer = shap.TreeExplainer(xgb_model)

    # --- Waterfall: why this specific prediction ---
    st.subheader("Why this prediction? (Waterfall Plot)")
    shap_single = explainer.shap_values(input_row)

    if isinstance(shap_single, list):
        sv = shap_single[pred_xgb][0]
        ev = explainer.expected_value[pred_xgb]
    else:
        sv = shap_single[0]
        ev = float(explainer.expected_value)

    explanation = shap.Explanation(
        values=sv,
        base_values=ev,
        data=input_row.values[0],
        feature_names=list(input_row.columns)
    )
    fig1, ax1 = plt.subplots()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig1, bbox_inches='tight')
    plt.close(fig1)

    st.markdown("---")

    # --- Global importance on test set ---
    st.subheader("Global Feature Importance (Test Set)")
    sample = X_test.sample(min(300, len(X_test)), random_state=42)
    shap_global = explainer.shap_values(sample)

    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_global, sample, plot_type="bar", show=False)
    st.pyplot(fig2)
    plt.close(fig2)

# =====================================================
# RISK ANALYSIS PAGE
# =====================================================
elif page == "📊 Risk Analysis":
    st.title("📊 Risk Analysis")
    st.write(f"Analysis based on predictions for year **{st.session_state['sel_year']}**.")
    st.caption("Change the year in the sidebar to explore different periods.")

    # Regional summary for selected year
    regional_summary = (
        all_preds.groupby('Region')['Risk_Level']
        .mean()
        .reset_index()
        .rename(columns={'Risk_Level': 'Avg_Risk'})
    )
    regional_summary['Risk_Category'] = regional_summary['Avg_Risk'].apply(
        lambda x: 'High Risk' if x >= 1.5 else ('Medium Risk' if x >= 0.5 else 'Low Risk')
    )

    st.subheader(f"Regional Risk Summary — {st.session_state['sel_year']}")
    st.dataframe(regional_summary, use_container_width=True)

    st.subheader("Average Risk Score by Region")
    fig = px.bar(
        regional_summary, x='Region', y='Avg_Risk', color='Risk_Category',
        color_discrete_map={
            'Low Risk': '#28a745', 'Medium Risk': '#fd7e14', 'High Risk': '#dc3545'
        },
        title=f"Average Food Security Risk by Region ({st.session_state['sel_year']})"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Risk heatmap: region × crop for selected year
    st.subheader(f"Risk Heatmap: Region × Crop ({st.session_state['sel_year']})")
    pivot = all_preds.pivot_table(
        values='Risk_Level', index='Region', columns='Crop', aggfunc='mean'
    )
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd',
                linewidths=0.5, ax=ax3, vmin=0, vmax=2)
    ax3.set_title(f"Risk Level by Region and Crop — {st.session_state['sel_year']}")
    st.pyplot(fig3)
    plt.close(fig3)

    # High risk table
    st.subheader("🔴 High Risk Combinations")
    high_risk = all_preds[all_preds['Risk_Category'] == 'High Risk'][
        ['Region', 'Crop', 'Year', 'Risk_Category']
    ]
    if not high_risk.empty:
        st.dataframe(high_risk, use_container_width=True)
    else:
        st.success("No high-risk combinations found for this year.")

# =====================================================
# ETHIOPIA RISK MAP PAGE
# =====================================================
elif page == "🗺️ Ethiopia Risk Map":
    st.title("🗺️ Ethiopia Food Security Risk Map")
    st.write(f"Showing average predicted risk across all crops for year **{st.session_state['sel_year']}**.")

    regional_summary = (
        all_preds.groupby('Region')['Risk_Level']
        .mean()
        .reset_index()
        .rename(columns={'Risk_Level': 'Avg_Risk'})
    )
    regional_summary['Risk_Category'] = regional_summary['Avg_Risk'].apply(
        lambda x: 'High Risk' if x >= 1.5 else ('Medium Risk' if x >= 0.5 else 'Low Risk')
    )

    fig = px.choropleth_mapbox(
        regional_summary,
        geojson=geojson,
        featureidkey="properties.ADM1_EN",
        locations='Region',
        color='Risk_Category',
        color_discrete_map={
            'Low Risk': '#28a745',
            'Medium Risk': '#fd7e14',
            'High Risk': '#dc3545'
        },
        hover_data={'Avg_Risk': ':.2f'},
        mapbox_style="carto-positron",
        zoom=5,
        center={"lat": 9.145, "lon": 40.4897},
        opacity=0.7,
        title=f"Food Security Risk Map — {st.session_state['sel_year']}"
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Regional Risk Table")
    st.dataframe(
        regional_summary.sort_values('Avg_Risk', ascending=False),
        use_container_width=True
    )

# =====================================================
# ABOUT PAGE
# =====================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This System")
    st.markdown("""
This Agricultural Early Warning System supports food security decision-making in Ethiopia.

**How to use:**
1. Select a **Year**, **Region**, and **Crop Type** in the sidebar
2. Click **▶ Run Prediction**
3. Navigate between pages — all pages update based on your selection

**Models:**
- 🌳 Random Forest Classifier
- 🚀 XGBoost Classifier (primary)

**Explainability:**
- 🔍 SHAP Waterfall + Summary plots

**Data:** Ethiopian regional crop production data (1996–2022)

**Risk Levels:**
- 🟢 Low Risk (0)
- 🟡 Medium Risk (1)
- 🔴 High Risk (2)
""")
