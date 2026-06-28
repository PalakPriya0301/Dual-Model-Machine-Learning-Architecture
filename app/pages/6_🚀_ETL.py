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
    if file.name.endswith(".csv"):
        try:
            return pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, encoding="latin1")
    else:
        return pd.read_excel(file)


def assign_persona_composite(rfm_df):
    """
    FIX: Assign personas using composite RFM score — same logic as training pipeline.
    High Frequency + High Monetary + Low Recency = Top-Tier.
    """
    cluster_stats = rfm_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    composite = norm(cluster_stats["Frequency"]) + norm(cluster_stats["Monetary"]) - norm(cluster_stats["Recency"])
    ranked    = composite.sort_values()

    return {
        int(ranked.index[0]): "At-Risk Sleepers",
        int(ranked.index[1]): "Promising Newcomers",
        int(ranked.index[2]): "Top-Tier Customers",
    }


uploaded_file = st.file_uploader("Upload Raw Transaction Data", type=["csv", "xlsx"])

if uploaded_file:
    raw_data = load_data(uploaded_file)

    st.write("### 1. Data Preview")
    st.dataframe(raw_data.head(), use_container_width=True)

    st.write("### 2. Map Your Data Columns")
    st.info("💡 **Tip:** If your file has a 'Total Amount', map it to Spend and leave Quantity blank. "
            "If your file has 'Unit Price', map it to Spend and also map the 'Quantity' column.")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        id_col      = st.selectbox("Customer ID", raw_data.columns)
    with col2:
        date_col    = st.selectbox("Transaction Date", raw_data.columns)
    with col3:
        invoice_col = st.selectbox("Invoice / Order ID (for Frequency count)", raw_data.columns)
    with col4:
        spend_col   = st.selectbox("Spend / Unit Price", raw_data.columns)
    with col5:
        qty_options = ["-- None (Already a Total) --"] + list(raw_data.columns)
        qty_col     = st.selectbox("Quantity (Optional)", qty_options)

    if st.button("🚀 Execute Live Clustering Pipeline", type="primary", use_container_width=True):
        with st.spinner("Processing ETL Pipeline & Running Live K-Means..."):
            try:
                raw_data[date_col] = pd.to_datetime(raw_data[date_col], errors="coerce")
                initial_rows = len(raw_data)
                raw_data     = raw_data.dropna(subset=[id_col])

                if qty_col != "-- None (Already a Total) --":
                    raw_data["Calculated_Spend"] = raw_data[spend_col] * raw_data[qty_col]
                else:
                    raw_data["Calculated_Spend"] = raw_data[spend_col]

                raw_data     = raw_data[raw_data["Calculated_Spend"] > 0]
                final_rows   = len(raw_data)
                dropped_rows = initial_rows - final_rows

                if dropped_rows > 0:
                    st.info(f"🧹 **Auto-Clean:** Removed **{dropped_rows:,}** invalid rows.")

                recent_date = raw_data[date_col].max()

                
                rfm = raw_data.groupby(id_col).agg(
                    Recency=   (date_col,    lambda x: (recent_date - x.max()).days),
                    Frequency= (invoice_col, "nunique"),  
                    Monetary=  ("Calculated_Spend", "sum"),
                ).reset_index().dropna()

                if len(rfm) < 3:
                    st.error("❌ Not enough customers to form 3 clusters. Upload a larger dataset.")
                    st.stop()

                qt     = QuantileTransformer(output_distribution="normal", random_state=42)
                scaled = qt.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

                kmeans          = KMeans(n_clusters=3, random_state=42, n_init=10)
                rfm["Cluster"]  = kmeans.fit_predict(scaled)

               
                labels               = assign_persona_composite(rfm)
                rfm["Assigned_Persona"] = rfm["Cluster"].map(labels)

                st.session_state["dynamic_rfm"] = rfm
                st.success("✅ Dynamic Segmenting Complete!")

                st.markdown("### 3. Persona Distribution")
                dist          = rfm["Assigned_Persona"].value_counts().reset_index()
                dist.columns  = ["Persona", "Count"]
                st.dataframe(dist, use_container_width=True, hide_index=True)

                st.markdown("### 4. Cluster Visualisation (3D RFM Space)")
                st.caption("Axes: Recency (days since last purchase), Frequency (unique orders), Monetary (total spend)")
                fig = px.scatter_3d(
                    rfm, x="Recency", y="Frequency", z="Monetary",
                    color="Assigned_Persona",
                    opacity=0.8,
                    color_discrete_map={
                        "Top-Tier Customers":  "#00CC96",
                        "Promising Newcomers": "#636EFA",
                        "At-Risk Sleepers":    "#EF553B",
                    },
                )
                fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 5. Full RFM Table")
                st.dataframe(rfm, use_container_width=True, hide_index=True)

                csv_out = rfm.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Segmented Data as CSV",
                    data=csv_out,
                    file_name="live_segmented_customers.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"ETL Error: {e}")
