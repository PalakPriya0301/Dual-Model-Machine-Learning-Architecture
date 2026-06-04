import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Dashboard", layout="wide")

if 'historical_df' not in st.session_state:
    st.error("Please go to the Home page first to load the database.")
    st.stop()

df = st.session_state['historical_df']

st.title("E-Commerce Customer Persona Dashboard")
    
selected_persona = st.selectbox("Select Persona:", df['Persona'].unique())
filtered_df = df[df['Persona'] == selected_persona]

col1, col2, col3 = st.columns(3)
col1.metric("Customers", len(filtered_df))
col2.metric("Avg Spend", f"${filtered_df['Monetary'].mean():.2f}")
col3.metric("Avg Recency", f"{filtered_df['Recency'].mean():.0f} days")

st.write("### 🌐 Mathematical Cluster Separation (3D View)")
fig_3d = px.scatter_3d(
    df, x='Recency', y='Frequency', z='Monetary',
    color='Persona',
    opacity=0.8,
    color_discrete_map={
        "Top-Tier Customers": "#00CC96",      
        "Promising Newcomers": "#636EFA",     
        "At-Risk Sleepers": "#EF553B"      
    }
)
fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_3d, use_container_width=True)