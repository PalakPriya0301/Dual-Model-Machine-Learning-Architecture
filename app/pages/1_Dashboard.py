import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Dashboard", layout="wide")

if "historical_df" not in st.session_state:
    st.error("Please go to the Home page first to load the database.")
    st.stop()

df = st.session_state["historical_df"]

st.title("📊 E-Commerce Customer Persona Dashboard")

with st.expander("📖 What is RFM? (click to expand)"):
    st.markdown("""
    **RFM** is a behaviour-based segmentation framework:

    | Dimension | Meaning | Good value |
    |---|---|---|
    | **Recency** | Days since last purchase | Low (bought recently) |
    | **Frequency** | Number of unique orders | High (buys often) |
    | **Monetary** | Total spend (£/\$) | High (spends more) |

    Customers are clustered into 3 personas using **K-Means** on these three dimensions.
    The 3D chart below shows how well-separated the clusters are in RFM space.
    """)

st.markdown("---")

selected_persona = st.selectbox("Filter by Persona:", ["All"] + list(df["Persona"].unique()))
filtered_df = df if selected_persona == "All" else df[df["Persona"] == selected_persona]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers",    f"{len(filtered_df):,}")
col2.metric("Avg Spend",    f"${filtered_df['Monetary'].mean():,.2f}")
col3.metric("Avg Recency",  f"{filtered_df['Recency'].mean():.0f} days")
col4.metric("Avg Orders",   f"{filtered_df['Frequency'].mean():.1f}")

st.markdown("### 🌐 Mathematical Cluster Separation (3D RFM Space)")
st.caption("Each dot is one customer. Axes = Recency (days), Frequency (orders), Monetary (spend $)")

fig_3d = px.scatter_3d(
    df, x="Recency", y="Frequency", z="Monetary",
    color="Persona",
    opacity=0.8,
    color_discrete_map={
        "Top-Tier Customers":  "#00CC96",
        "Promising Newcomers": "#636EFA",
        "At-Risk Sleepers":    "#EF553B",
    },
)
fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_3d, use_container_width=True)
