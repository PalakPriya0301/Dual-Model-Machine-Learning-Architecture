import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans

st.set_page_config(page_title="Dynamic ETL Pipeline", layout="wide")

st.title("🚀 Automated RFM & Clustering Engine")
st.markdown("Upload raw transaction data, map columns, and compute customer personas.")

def load_data(file):
    if file.name.endswith(".csv"):
        try: return pd.read_csv(file)
        except: return pd.read_csv(file, encoding="latin1")
    return pd.read_excel(file)

uploaded_file = st.file_uploader("Upload Raw Transaction Data", type=["csv", "xlsx"])

if uploaded_file:
    raw_data = load_data(uploaded_file)
    st.dataframe(raw_data.head(), use_container_width=True)

    st.write("### 2. Map Your Data Columns")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1: id_col    = st.selectbox("Customer ID", raw_data.columns)
    with col2: date_col  = st.selectbox("Transaction Date", raw_data.columns)
    with col3: spend_col = st.selectbox("Spend / Unit Price", raw_data.columns)
    with col4: 
        qty_options = ["-- None (Already a Total) --"] + list(raw_data.columns)
        qty_col     = st.selectbox("Quantity (Optional)", qty_options)

    if st.button("🚀 Execute Live Clustering Pipeline", type="primary", use_container_width=True):
        try:
            raw_data[date_col] = pd.to_datetime(raw_data[date_col], errors="coerce")
            raw_data = raw_data.dropna(subset=[id_col, date_col])
            
            # Logic: If Qty column selected, multiply. Else, use Spend column.
            if qty_col != "-- None (Already a Total) --":
                raw_data["Calculated_Spend"] = raw_data[spend_col] * raw_data[qty_col]
            else:
                raw_data["Calculated_Spend"] = raw_data[spend_col]
            
            raw_data = raw_data[raw_data["Calculated_Spend"] > 0]
            recent_date = raw_data[date_col].max()

            # RFM Aggregation
            rfm = raw_data.groupby(id_col).agg(
                Recency=(date_col, lambda x: (recent_date - x.max()).days),
                Frequency=(id_col, "count"), 
                Monetary=("Calculated_Spend", "sum"),
            ).reset_index()

            # Scaling & Clustering
            qt = QuantileTransformer(output_distribution="normal", random_state=42)
            scaled = qt.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
            
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            rfm["Cluster"] = kmeans.fit_predict(scaled)

            # Persona Assignment
            cluster_means = rfm.groupby("Cluster")[["Monetary"]].mean().sort_values("Monetary")
            labels = {
                cluster_means.index[0]: "At-Risk Sleepers",
                cluster_means.index[1]: "Promising Newcomers",
                cluster_means.index[2]: "Top-Tier Customers",
            }
            rfm["Assigned_Persona"] = rfm["Cluster"].map(labels)

            st.success("✅ Clustering Complete!")
            st.dataframe(rfm, use_container_width=True)

            # Viz
            fig = px.scatter_3d(rfm, x="Recency", y="Frequency", z="Monetary", color="Assigned_Persona")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"ETL Error: {e}")