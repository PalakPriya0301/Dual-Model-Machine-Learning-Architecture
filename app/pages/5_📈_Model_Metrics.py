import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Model Metrics", layout="wide")

st.title("📈 Model Architecture & Justification")
st.markdown("""
This section provides full transparency into the machine learning models powering the
Enterprise CRM platform. It details the mathematical reasoning behind customer segmentation
and the performance metrics of the predictive churn engine.
""")

# --- THE FIX IS HERE: CLOUD-SAFE PATHING ---
# 1. Current file: app/pages/5_📈_Model_Metrics.py
PAGES_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up to 'app' folder
APP_DIR = os.path.dirname(PAGES_DIR)

# 3. Go up to root repository folder (where 'models' is actually located)
ROOT_DIR = os.path.dirname(APP_DIR)

# 4. Connect to the models folder
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def _img(filename):
    path = os.path.join(MODELS_DIR, filename)
    return path if os.path.exists(path) else None

_missing_msg = "⚠️ Chart not found. Run `models/3_model_training.py` locally and push PNGs to GitHub."

st.markdown("---")

st.header("1. Customer Segmentation (Unsupervised Learning)")
st.markdown("""
**Algorithm Chosen:** K-Means Clustering

**Why K-Means?** Highly efficient for partitioning large continuous numerical datasets
(like RFM features) into distinct non-overlapping groups. Cluster centroids map directly
to interpretable customer archetypes.

To find the optimal number of personas (K), two industry-standard techniques were used:

- **Elbow Method:** Plots inertia vs K. The "elbow" point indicates diminishing returns
  from adding more clusters.
- **Silhouette Score:** Measures how similar each point is to its own cluster vs others.
  Ranges from -1 to 1 — higher is better.
""")

path = _img("clustering_validation.png")
if path:
    st.image(Image.open(path), caption="K-Means Validation: Elbow Method & Silhouette Score (K=3 chosen)", use_container_width=True)
else:
    st.info(_missing_msg)

st.success(
    "**Conclusion:** Both metrics confirm **K=3** as the optimal partition, producing: "
    "*Top-Tier Customers*, *Promising Newcomers*, and *At-Risk Sleepers*."
)

st.markdown("---")

st.header("2. Churn Prediction (Supervised Learning)")
st.markdown("""
**Algorithm Chosen:** Random Forest Classifier

**Why Random Forest?**
- Handles non-linear feature interactions without manual engineering.
- Robust to class imbalance via `class_weight='balanced'`.
- Provides Gini feature importance for business explainability.
- Resistant to overfitting through ensemble averaging (300 trees).

**Custom Decision Threshold = 0.35** (default is 0.50)

> **Business justification:** A false negative (missing a real churner) costs far more
> than a false positive (sending a coupon to a loyal customer). Lowering the threshold
> to 0.35 increases Recall — catching more at-risk customers — at the acceptable cost
> of slightly more marketing spend.
""")

path = _img("model_evaluation.png")
if path:
    st.image(Image.open(path), caption="Random Forest: Confusion Matrix & ROC Curve", use_container_width=True)
else:
    st.info(_missing_msg)

st.markdown("---")

st.subheader("3. What Drives Churn?")
st.markdown("""
**Gini Importance** extracted from the Random Forest ensemble shows which features
most strongly predict customer abandonment.

> **Why is Recency not in the churn model?**
> The churn label is derived directly from Recency (customers inactive beyond the
> 70th percentile Recency threshold are labelled churned). Including Recency as a
> model input would be **data leakage** — the model would achieve near-perfect accuracy
> for a trivial reason (it already knows the answer). The model instead learns to infer
> churn risk purely from purchase behaviour: Frequency and Monetary value.
""")

path = _img("feature_importance.png")
if path:
    st.image(Image.open(path), caption="Gini Feature Importance — Churn Prediction Model", use_container_width=True)
else:
    st.info(_missing_msg)