import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans

st.set_page_config(page_title="SaaS Dynamic ETL", layout="wide")

# --- UI Header ---
st.title("🚀 Enterprise SaaS: Universal Transaction Segmenter")
st.markdown(
    "Upload any raw transaction file, map your columns, and the pipeline will "
    "compute RFM features and assign customer personas — using the same algorithm "
    "as the core training pipeline."
)

# ── FILE LOADING ──────────────────────────────────────────────
# No cache here: ETL uploads are expected to be different each time.
def load_data(file):
    if file.name.endswith('.csv'):
        try:
            return pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, encoding='latin1')
    else:
        return pd.read_excel(file)

uploaded_file = st.file_uploader("Upload Raw Transaction Data", type=["csv", "xlsx"])

if uploaded_file:
    raw_data = load_data(uploaded_file)

    st.write("### 1. Data Preview")
    st.dataframe(raw_data.head(), use_container_width=True)

    st.write("### 2. Map Your Data Columns")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        id_col    = st.selectbox("Customer ID Column", raw_data.columns, index=len(raw_data.columns) - 2)
    with col2:
        date_col  = st.selectbox("Transaction Date Column", raw_data.columns, index=len(raw_data.columns) - 4)
    with col3:
        spend_col = st.selectbox("Transaction Amount/Spend Column", raw_data.columns, index=len(raw_data.columns) - 3)

    if st.button("🚀 Execute Live Clustering Pipeline", type="primary", use_container_width=True):
        with st.spinner("Processing ETL Pipeline & Running Live K-Means..."):
            try:
                # Format the dates
                raw_data[date_col] = pd.to_datetime(raw_data[date_col], errors='coerce')
                
                # ─── 🧹 BASIC DATA CLEANING ──────────────────────────────────
                initial_rows = len(raw_data)
                
                # 1. Drop rows where we don't know who the customer is
                raw_data = raw_data.dropna(subset=[id_col])
                
                # 2. Drop negative/zero spend (This automatically removes refunds & cancellations)
                raw_data = raw_data[raw_data[spend_col] > 0]
                
                final_rows = len(raw_data)
                dropped_rows = initial_rows - final_rows
                
                if dropped_rows > 0:
                    st.info(f"🧹 **Auto-Clean:** Removed **{dropped_rows:,}** invalid rows (missing IDs or negative/refunded amounts) to protect AI accuracy.")
                # ─────────────────────────────────────────────────────────────

                # Identify the "current" date in the dataset to calculate Recency
                recent_date = raw_data[date_col].max()

                # Build the RFM Features
                rfm = raw_data.groupby(id_col).agg(
                    Recency  = (date_col,  lambda x: (recent_date - x.max()).days),
                    Frequency= (id_col,    'count'),
                    Monetary = (spend_col, 'sum')
                ).reset_index().dropna()

                if len(rfm) < 3:
                    st.error("❌ Not enough customers to form 3 clusters. Upload a larger dataset.")
                    st.stop()

                # QuantileTransformer — exactly matching the main training pipeline
                qt = QuantileTransformer(output_distribution='normal', random_state=42)
                scaled = qt.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

                # K-Means Clustering
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                rfm['Cluster'] = kmeans.fit_predict(scaled)

                # Dynamically assign labels based on Monetary value
                cluster_means = rfm.groupby('Cluster')['Monetary'].mean().sort_values()
                labels = {
                    cluster_means.index[0]: "At-Risk Sleepers",
                    cluster_means.index[1]: "Promising Newcomers",
                    cluster_means.index[2]: "Top-Tier Customers"
                }
                rfm['Assigned_Persona'] = rfm['Cluster'].map(labels)

                # Save to session state so other pages could theoretically access it
                st.session_state['dynamic_rfm'] = rfm

                st.success("✅ Dynamic Segmenting Complete!")

                # --- 3. Persona Distribution ---
                st.markdown("### 3. Persona Distribution")
                dist = rfm['Assigned_Persona'].value_counts().reset_index()
                dist.columns = ['Persona', 'Count']
                st.dataframe(dist, use_container_width=True, hide_index=True)

                # --- 4. 3D Cluster Visualisation ---
                st.markdown("### 4. Cluster Visualisation (3D RFM Space)")
                fig = px.scatter_3d(
                    rfm, x='Recency', y='Frequency', z='Monetary',
                    color='Assigned_Persona',
                    opacity=0.8,
                    color_discrete_map={
                        "Top-Tier Customers":  "#00CC96",
                        "Promising Newcomers": "#636EFA",
                        "At-Risk Sleepers":    "#EF553B"
                    }
                )
                fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
                st.plotly_chart(fig, use_container_width=True)

                # --- 5. Full Downloadable Table ---
                st.markdown("### 5. Full RFM Table")
                st.dataframe(rfm, use_container_width=True, hide_index=True)

                csv = rfm.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Segmented Data as CSV",
                    data=csv,
                    file_name='live_segmented_customers.csv',
                    mime='text/csv'
                )

            except Exception as e:
                st.error(f"ETL Error: {e}")