import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import joblib

st.set_page_config(page_title="Predict New Customer", layout="wide")

st.title("🔮 Predict Customer Persona & Churn Risk")
st.write("Enter metrics below to classify a new customer and calculate their churn probability.")


_required_keys = ['historical_model', 'churn_model', 'scaler', 'label_map']
if any(k not in st.session_state for k in _required_keys):
    st.error("⚠️ Models not loaded. Please go to the **Home** page first to initialise the system.")
    st.stop()

persona_model = st.session_state['historical_model']
churn_model   = st.session_state['churn_model']
scaler        = st.session_state['scaler']
label_map     = st.session_state['label_map']

# ── INPUTS ────────────────────────────────────────────────────

st.markdown("### 📝 Input Customer Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    recency       = st.number_input("Recency (Days)", min_value=0, max_value=365, value=184)
with col2:
    frequency     = st.number_input("Frequency (Orders)", min_value=1, max_value=100, value=11)
with col3:
    monetary      = st.number_input("Monetary (Spend $)", min_value=0.0, max_value=10000.0, value=5563.56, format="%.2f")



avg_order_value = monetary / frequency if frequency > 0 else 0


input_data = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])

churn_input_data = pd.DataFrame([[frequency, monetary, avg_order_value]], 
                                columns=['Frequency', 'Monetary', 'AvgOrderValue'])

st.markdown("---")
if st.button("🚀 Run AI Diagnosis", type="primary", use_container_width=True):

    input_scaled       = scaler.transform(input_data)
    cluster_idx        = int(persona_model.predict(input_scaled)[0])
    persona_prediction = label_map.get(cluster_idx, f"Unknown Cluster {cluster_idx}")

    churn_proba = churn_model.predict_proba(churn_input_data)[0][1] * 100

    st.markdown("### 🔍 Diagnosis Results")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.success(f"**Predicted Segment:**\n## {persona_prediction}")

    with res_col2:
        st.markdown(f"**Churn Risk:**\n## {churn_proba:.1f}%")
        if churn_proba > 50:
            st.error("🚨 CRITICAL: High risk of abandonment.")
        elif churn_proba > 35:
            st.warning("⚠️ MODERATE: Exceeds 35% action threshold. Monitor behaviour.")
        else:
            st.info("✅ LOW: Customer is loyal.")
        st.progress(min(int(churn_proba), 100))

    st.markdown("---")
    st.subheader("🧠 Model Decision Logic (SHAP)")
    st.caption("Red = increases churn risk | Blue = reduces churn risk")

    try:
        explainer       = shap.TreeExplainer(churn_model)
        shap_values_raw = explainer.shap_values(churn_input_data)

        if isinstance(shap_values_raw, list):
            sv = np.array(shap_values_raw[1]).flatten()
        elif len(shap_values_raw.shape) == 3:
            sv = shap_values_raw[0, :, 1].flatten()
        else:
            sv = np.array(shap_values_raw).flatten()

        shap_df = pd.DataFrame({
            'Feature': ['Frequency', 'Monetary'],
            'Impact':  sv
        }).sort_values(by='Impact')

        shap_df['Color'] = ['#EF553B' if x > 0 else '#636EFA' for x in shap_df['Impact']]

        fig_shap = px.bar(
            shap_df, x='Impact', y='Feature', orientation='h',
            color='Color', color_discrete_map="identity",
            labels={'Impact': 'Impact on Churn Probability'}
        )
        fig_shap.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    except Exception as e:
        st.warning(f"SHAP visualisation unavailable: {e}")