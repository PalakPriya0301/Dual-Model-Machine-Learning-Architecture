import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Training Data Archive", page_icon="📁", layout="wide")

st.title("📁 AI Training Data Archive")
st.markdown("""
This module provides complete transparency into the data pipeline. 
Below you can view the **Raw Transactional Data** (Before) and the resulting **AI-Scored Customer Profiles** (After).
""")

# CLOUD-SAFE PATHING
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
    csv_path = os.path.join(DATA_DIR, "Online Retail.csv")
    
    try:
        if os.path.exists(excel_path):
             return pd.read_excel(excel_path, nrows=1000)
        elif os.path.exists(csv_path):
             return pd.read_csv(csv_path, nrows=1000, encoding="ISO-8859-1")
        else:
             st.info(f"⚠️ Note: Raw dataset not pushed to cloud storage to save space. AI profiles below are fully loaded.")
             return None
    except Exception as e:
        st.error(f"Error loading raw file: {e}")
        return None

df_historical = load_historical_data()
df_raw = load_raw_transactions()

if df_historical is None:
    st.error("⚠️ `historical_data.pkl` not found. Please run the model training script locally.")
    st.stop()

st.markdown("### 📈 Database Overview")
col1, col2, col3 = st.columns(3)

col1.metric("Total Unique Customers", f"{len(df_historical):,}")
col2.metric("Avg Orders (Freq)", f"{df_historical['Frequency'].mean():.1f}")
col3.metric("Avg Lifetime Spend", f"${df_historical['Monetary'].mean():,.2f}")

st.markdown("---")

st.markdown("### 🛒 1. Raw Transactional File (The 'Before')")
st.markdown("This is a sample of the raw `Online Retail` file before the ETL pipeline cleans it and groups it by CustomerID.")

if df_raw is not None:
    st.dataframe(df_raw, use_container_width=True, height=300)

st.markdown("<br><div style='text-align: center;'>⬇️ <i>ETL Pipeline groups these transactions by Customer into RFM features...</i> ⬇️</div><br>", unsafe_allow_html=True)

st.markdown("### 🧠 2. AI-Scored Customer Profiles (The 'After')")
st.markdown("This is the final dataset after Unsupervised Clustering (Persona) and Supervised Classification (Churn).")

st.dataframe(
    df_historical[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Cluster', 'Persona', 'Churn_Label']],
    use_container_width=True,
    hide_index=True,
    height=400
)

csv = df_historical.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Final Customer CSV",
    data=csv,
    file_name='ai_scored_customers.csv',
    mime='text/csv',
)