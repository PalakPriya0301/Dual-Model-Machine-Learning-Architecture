import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans

st.set_page_config(page_title="Dynamic ETL Pipeline", layout="wide")

st.title("🚀 Automated RFM & Clustering Engine")

def load_data(file):
    if file.name.endswith(".csv"):
        try:
            return pd.read_csv(file)
        except:
            return pd.read_csv(file, encoding="latin1")
    return pd.read_excel(file)

uploaded_file = st.file_uploader("Upload Raw Transaction Data", type=["csv", "xlsx"])

if uploaded_file:
    raw_data = load_data(uploaded_file)
    st.write("### 1. Data Preview")
    st.dataframe(raw_data.head(), use_container_width=True)

    # Use a form to prevent the "snap out" behavior
    with st.form("etl_form"):
        st.write("### 2. Map Your Data Columns")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            id_col = st.selectbox("Customer ID", raw_data.columns)
        with col2:
            date_col = st.selectbox("Transaction Date", raw_data.columns)
        with col3:
            spend_col = st.selectbox("Spend (Total Amount)", raw_data.columns)
        with col4:
            # We derive frequency from the ID column automatically
            st.write("**Frequency is auto-calculated from Customer ID**")
            
        submitted = st.form_submit_button("🚀 Execute Live Clustering Pipeline", type="primary")

    if submitted:
        with st.spinner("Processing..."):
            try:
                # Clean
                raw_data[date_col] = pd.to_datetime(raw_data[date_col], errors="coerce")
                raw_data = raw_data.dropna(subset=[id_col])
                recent_date = raw_data[date_col].max()

                # Aggregate (Recency: Date, Frequency: Count of rows, Monetary: Sum of Spend)
                rfm = raw_data.groupby(id_col).agg({
                    date_col: lambda x: (recent_date - x.max()).days,
                    id_col: "count",
                    spend_col: "sum"
                }).rename(columns={date_col: "Recency", id_col: "Frequency", spend_col: "Monetary"})

                # K-Means
                qt = QuantileTransformer(output_distribution="normal", random_state=42)
                scaled = qt.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                rfm["Cluster"] = kmeans.fit_predict(scaled)

                # Assign Personas (Composite Score logic)
                cluster_means = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
                # Simple logic: High Monetary/Freq, Low Recency = Top Tier
                rfm["Assigned_Persona"] = rfm["Cluster"].map({
                    0: "At-Risk Sleepers",
                    1: "Promising Newcomers",
                    2: "Top-Tier Customers"
                })

                st.success("✅ Pipeline Complete!")
                st.dataframe(rfm.head(), use_container_width=True)
                
                # Visual
                fig = px.scatter_3d(rfm, x="Recency", y="Frequency", z="Monetary", color="Assigned_Persona")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {e}")