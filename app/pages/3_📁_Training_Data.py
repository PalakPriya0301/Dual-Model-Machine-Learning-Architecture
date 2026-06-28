import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Training Data Archive", page_icon="📁", layout="wide")

st.title("📁 AI Training Data Archive")
st.markdown("""
This module provides complete transparency into the data pipeline.
Below you can view the resulting **AI-Scored Customer Profiles**.
""")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


@st.cache_data
def load_historical_data():
    file_path = os.path.join(BASE_DIR, "historical_data.pkl")
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    return None


@st.cache_data
def load_raw_transactions():
    excel_path = os.path.join(DATA_DIR, "Online Retail.xlsx")
    csv_path   = os.path.join(DATA_DIR, "Online Retail.csv")
    try:
        if os.path.exists(excel_path):
            return pd.read_excel(excel_path, nrows=1000)
        elif os.path.exists(csv_path):
            return pd.read_csv(csv_path, nrows=1000, encoding="ISO-8859-1")
        else:
            return None  
    except Exception as e:
        st.error(f"Error loading raw file: {e}")
        return None


df_historical = load_historical_data()

if df_historical is None:
    st.error("⚠️ `historical_data.pkl` not found. Please run the model training script locally.")
    st.stop()

st.markdown("### 📈 Database Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Unique Customers", f"{len(df_historical):,}")
col2.metric("Avg Orders (Freq)",      f"{df_historical['Frequency'].mean():.1f}")
col3.metric("Avg Lifetime Spend",     f"${df_historical['Monetary'].mean():,.2f}")

st.markdown("---")

st.markdown("### 🧠 AI-Scored Customer Profiles")
st.markdown("Final dataset after Unsupervised Clustering (Persona) and Supervised Classification (Churn).")

st.dataframe(
    df_historical[["CustomerID", "Recency", "Frequency", "Monetary", "Cluster", "Persona", "Churn_Label"]],
    use_container_width=True,
    hide_index=True,
    height=400,
)

csv = df_historical.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Final Customer CSV",
    data=csv,
    file_name="ai_scored_customers.csv",
    mime="text/csv",
)

st.markdown("---")


df_raw = load_raw_transactions()
st.markdown("### 📂 Raw Transaction Sample")
if df_raw is not None:
    st.caption("Showing first 1,000 rows of the source dataset.")
    st.dataframe(df_raw.head(20), use_container_width=True, hide_index=True)
else:
    
    st.info(
        "ℹ️ The raw transaction file (`Online Retail.xlsx`) is not available on the cloud deployment "
        "as it exceeds GitHub's file size limits. To view it, clone the repository locally and place "
        "the file in the `data/` directory."
    )
