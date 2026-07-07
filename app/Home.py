import os
import sqlite3
import subprocess
import sys

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Enterprise AI Segmentation",
    page_icon="🏢",
    layout="centered",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ETL_DIR  = os.path.join(BASE_DIR, "..", "etl")
DB_PATH  = os.path.join(BASE_DIR, "enterprise_crm.db")


def _pkl(name):
    """Return full path to a pkl file in the app directory."""
    return os.path.join(BASE_DIR, name)


# ── HOME PAGE ─────────────────────────────────────────────────
st.title("🏢 Welcome to the Enterprise Segmentation Portal")
st.markdown("""
This platform provides end-to-end customer intelligence using Machine Learning.

**💡 Core Business Impact:**
* **Increases Retention:** Identifies high-risk customers before they churn using predictive modeling.
* **Reduces Marketing Waste:** Allocates discounts only to segments that need them.
* **Automates Effort:** The Smart Marketing Assistant drastically reduces manual CRM workload.

**👈 Please select a module from the sidebar to begin:**
* **📊 Dashboard:** View historical segmentation analytics.
* **🔮 Predict:** Input real-time metrics to classify a new customer & calculate churn risk.
* **📁 Training Data:** Explore the complete database archive and view customer profiles.
* **🤖 Marketing Assistant:** Look up users and automatically dispatch targeted emails.
* **📈 Model Metrics:** Understand the AI architecture powering this platform.
* **🚀 ETL:** Upload custom transaction logs for dynamic clustering.
""")

# ── PRE-BOOT CHECK ────────────────────────────────────────────
_required = ["persona_model.pkl", "churn_model.pkl", "scaler.pkl", "persona_label_map.pkl"]
_missing  = [f for f in _required if not os.path.exists(_pkl(f))]

if _missing:
    st.error(
        f"⚠️ Missing model files: **{', '.join(_missing)}**\n\n"
        "**Fix:** Run these scripts in order on your local machine, then push to GitHub:\n"
        "1. `etl/1_database_setup.py`\n"
        "2. `etl/2_feature_engineering.py`\n"
        "3. `models/3_model_training.py`"
    )
    st.stop()


# ── LOAD CORE ASSETS ──────────────────────────────────────────
@st.cache_data
def load_historical_data():
    conn      = sqlite3.connect(DB_PATH)
    df        = pd.read_sql_query("SELECT * FROM customer_features", conn)
    conn.close()

    scaler    = joblib.load(_pkl("scaler.pkl"))
    kmeans    = joblib.load(_pkl("persona_model.pkl"))
    label_map = joblib.load(_pkl("persona_label_map.pkl"))

    features  = ["Recency", "Frequency", "Monetary"]
    X_scaled  = scaler.transform(df[features])
    df["Cluster"] = kmeans.predict(X_scaled)
    df["Persona"] = df["Cluster"].map(label_map)
    return df


@st.cache_resource
def load_historical_model():
    return joblib.load(_pkl("persona_model.pkl"))


@st.cache_resource
def load_churn_model():
    return joblib.load(_pkl("churn_model.pkl"))


@st.cache_resource
def load_scaler():
    return joblib.load(_pkl("scaler.pkl"))


@st.cache_resource
def load_label_map():
    return joblib.load(_pkl("persona_label_map.pkl"))


# ── BOOT SEQUENCE ─────────────────────────────────────────────
try:
    if "historical_df" not in st.session_state:
        st.session_state["historical_df"] = load_historical_data()
    if "historical_model" not in st.session_state:
        st.session_state["historical_model"] = load_historical_model()
    if "churn_model" not in st.session_state:
        st.session_state["churn_model"] = load_churn_model()
    if "scaler" not in st.session_state:
        st.session_state["scaler"] = load_scaler()
    if "label_map" not in st.session_state:
        st.session_state["label_map"] = load_label_map()

    st.success("✅ Enterprise Database & AI Models successfully loaded into memory!")

except Exception as e:
    st.error(f"⚠️ System Boot Error: **{e}**")