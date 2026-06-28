import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans

st.set_page_config(page_title="Dynamic ETL Pipeline", layout="wide")

st.title("🚀 Automated RFM & Clustering Engine")
st.markdown(
    "Upload any raw transaction file, map your columns, and the pipeline will "
    "compute RFM features and assign customer personas — using the same algorithm "
    "as the core CRM training pipeline."
)

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
    st.info("💡 **Tip:** If your file has a 'Total Amount', map it to Spend and leave Quantity blank. If your file has 'Unit Price', map it to Spend and map the 'Quantity' column.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        id_col    = st.selectbox("Customer ID", raw_data.columns, index=len(raw_data.columns) - 2)
    with col2:
        date_col  = st.selectbox("Transaction Date", raw_data.columns, index=len(raw_data.columns) - 4)
    with col3:
        spend_col = st.selectbox("Spend / Unit Price", raw_data.columns, index=len(raw_data.columns) - 3)
    with col4:
        qty_options = ["-- None (Already a Total) --"] + list(raw_data.columns)
        qty_col   = st.selectbox("Quantity (Optional)", qty_options)

    if st.button("🚀 Execute Live Clustering Pipeline", type="primary", use_container_width=True):
        with st.spinner("Processing ETL Pipeline & Running Live K-Means..."):
            try:
                # 1. Clean the Data
                raw_data[date_col] = pd.to_datetime(raw_data[date_col], errors='coerce')
                initial_rows = len(raw_data)
                raw_data = raw_data.dropna(subset=[id_col])
                
                # 2. Dynamic Financial Logic 
                if qty_col != "-- None (Already a Total) --":
                    raw_data['Calculated_Spend'] = raw_data[spend_col] * raw_data[qty_col]
                else:
                    raw_data['Calculated_Spend'] = raw_data[spend_col]
                
                # Filter out negative or zero spend
                raw_data = raw_data[raw_data['Calculated_Spend'] > 0]
                
                final_rows = len(raw_data)
                dropped_rows = initial_rows - final_rows
                
                if dropped_rows > 0:
                    st.info(f"🧹 **Auto-Clean:** Removed **{dropped_rows:,}** invalid rows (missing IDs or negative/refunded amounts) to protect AI accuracy.")
                
                recent_date = raw_data[date_col].max()

                # 3. RFM Aggregation 
                rfm = raw_data.groupby(id_col).agg({
                    date_col: lambda x: (recent_date - x.max()).days,
                    id_col: 'count',
                    'Calculated_Spend': 'sum'
                }).rename(columns={
                    date_col: 'Recency',
                    id_col: 'Frequency',
                    'Calculated_Spend': 'Monetary'
                }).reset_index().dropna()

                if len(rfm) < 3:
                    st.error("❌ Not enough customers to form 3 clusters. Upload a larger dataset.")
                    st.stop()
                
                # 4. Run K-Means Clustering
                qt = QuantileTransformer(output_distribution='normal', random_state=42)
                scaled = qt.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
                
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                rfm['Cluster'] = kmeans.fit_predict(scaled)
                
                # Assign Personas based on Average Monetary Value
                cluster_means = rfm.groupby('Cluster')['Monetary'].mean().sort_values()
                labels = {
                    cluster_means.index[0]: "At-Risk Sleepers",
                    cluster_means.index[1]: "Promising Newcomers",
                    cluster_means.index[2]: "Top-Tier Customers"
                }
                rfm['Assigned_Persona'] = rfm['Cluster'].map(labels)
                
                # Save to session state
                st.session_state['dynamic_rfm'] = rfm
                st.success("✅ Dynamic Segmenting Complete!")
                
                # 5. Visualisations
                st.markdown("### 3. Persona Distribution")
                dist = rfm['Assigned_Persona'].value_counts().reset_index()
                dist.columns = ['Persona', 'Count']
                st.dataframe(dist, use_container_width=True, hide_index=True)
                
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
                
                st.markdown("### 5. Full RFM Table")
                st.dataframe(rfm, use_container_width=True, hide_index=True)
                
                # Download Button
                csv = rfm.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Segmented Data as CSV",
                    data=csv,
                    file_name='live_segmented_customers.csv',
                    mime='text/csv'
                )

            except Exception as e:
                st.error(f"ETL Error: {e}")