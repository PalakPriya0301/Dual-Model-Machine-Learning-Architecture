import json
import os

import streamlit as st
from PIL import Image

st.set_page_config(page_title="Model Metrics", layout="wide")

st.title("📈 Model Architecture & Justification")
st.markdown("""
This section provides full transparency into the machine learning models powering the
Enterprise CRM platform — including algorithm choices, validation methodology, and
live performance numbers.
""")

PAGES_DIR  = os.path.dirname(os.path.abspath(__file__))
APP_DIR    = os.path.dirname(PAGES_DIR)
ROOT_DIR   = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def _img(filename):
    path = os.path.join(MODELS_DIR, filename)
    return path if os.path.exists(path) else None


_missing_msg = "⚠️ Chart not found. Run `models/3_model_training.py` locally and push PNGs to GitHub."

# ── LIVE METRICS ─────────────────────────────────────────────

metrics_path = os.path.join(APP_DIR, "metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        m = json.load(f)

    st.markdown("### 🎯 Live Model Performance (from last training run)")
    st.caption(f"Based on {m.get('n_customers', '—'):,} customers | Aggressive threshold = {m.get('threshold_aggressive', 0.35)}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",      f"{m.get('accuracy', '—'):.1%}")
    c2.metric("Precision",     f"{m.get('precision', '—'):.1%}")
    c3.metric("Recall",        f"{m.get('recall', '—'):.1%}")
    c4.metric("F1 Score",      f"{m.get('f1_score', '—'):.1%}")
    c5.metric("ROC-AUC",       f"{m.get('roc_auc', '—'):.4f}")

    st.markdown("")
    cv_col1, cv_col2 = st.columns(2)
    st.markdown("")
    cv_col1, cv_col2 = st.columns(2)
    
    cv_mean = m.get('cv_f1_mean')
    cv_std = m.get('cv_f1_std')
    baseline = m.get('baseline_f1_050')

    cv_text = f"{cv_mean:.4f} ± {cv_std:.4f}" if cv_mean is not None and cv_std is not None else "—"
    base_text = f"{baseline:.4f}" if baseline is not None else "—"

    cv_col1.metric(
        "5-Fold CV F1 (mean ± std)",
        cv_text,
        help="Cross-validation gives a more reliable estimate than a single train/test split.",
    )
    cv_col2.metric(
        "Baseline F1 (threshold 0.50)",
        base_text,
        help="F1 at standard 0.50 threshold — lower than aggressive threshold because recall is not maximised.",
    )
    st.markdown("---")
else:
    st.info("ℹ️ `metrics.json` not found. Run `models/3_model_training.py` to generate live metrics.")
    st.markdown("---")

# ── SECTION 1: K-MEANS ───────────────────────────────────────
st.header("1. Customer Segmentation (Unsupervised Learning)")
st.markdown("""
**Algorithm:** K-Means Clustering

**Why K-Means?** Highly efficient for partitioning large continuous numerical datasets
(like RFM features) into distinct non-overlapping groups. Cluster centroids map directly
to interpretable customer archetypes.

Two industry-standard validation techniques were used to find the optimal K:

- **Elbow Method:** Plots inertia vs K. The "elbow" point marks diminishing returns from adding clusters.
- **Silhouette Score:** Measures how similar each point is to its own cluster vs others (higher = better).
""")

path = _img("clustering_validation.png")
if path:
    st.image(Image.open(path), caption="K-Means Validation: Elbow & Silhouette (K=3 chosen)", use_container_width=True)
else:
    st.info(_missing_msg)

st.success(
    "**Conclusion:** Both metrics confirm **K=3** as the optimal partition, producing: "
    "*Top-Tier Customers*, *Promising Newcomers*, and *At-Risk Sleepers*. "
    "Personas are assigned using a composite RFM score (not Monetary alone) to ensure "
    "all three dimensions contribute to labelling."
)

st.markdown("---")

# ── SECTION 2: RANDOM FOREST ─────────────────────────────────
st.header("2. Churn Prediction (Supervised Learning)")
st.markdown("""
**Algorithm:** Random Forest Classifier

**Why Random Forest?**
- Handles non-linear feature interactions without manual engineering.
- Robust to class imbalance via `class_weight='balanced'`.
- Provides Gini feature importance for business explainability.
- Resistant to overfitting through ensemble averaging (300 trees).

**Custom Decision Threshold = 0.35** (default is 0.50)

> **Business justification:** A false negative (missing a real churner) costs far more
> than a false positive (sending a coupon to a loyal customer). Lowering the threshold
> to 0.35 increases Recall — catching more at-risk customers — at the acceptable cost
> of slightly more marketing spend on non-churners.
""")

path = _img("model_evaluation.png")
if path:
    st.image(Image.open(path), caption="Random Forest: Confusion Matrix & ROC Curve", use_container_width=True)
else:
    st.info(_missing_msg)

st.markdown("---")

# ── SECTION 3: FEATURE IMPORTANCE ────────────────────────────
st.subheader("3. What Drives Churn?")
st.markdown("""
**Gini Importance** extracted from the Random Forest ensemble shows which features
most strongly predict customer abandonment.
""")

path = _img("feature_importance.png")
if path:
    st.image(Image.open(path), caption="Gini Feature Importance — Churn Prediction Model", use_container_width=True)
else:
    st.info(_missing_msg)


with st.expander("🔬 Why are Recency, TotalQuantity, and AvgOrderValue not in the churn model?"):
    st.markdown("""
    | Feature | Status | Reason |
    |---|---|---|
    | **Recency** | ❌ Excluded | **Data leakage** — the churn label is derived directly from Recency (customers beyond 70th percentile are labelled churned). Including it gives the model near-perfect accuracy for a trivial reason. |
    | **TotalQuantity** | ❌ Excluded | Pearson correlation with Frequency > 0.90 — it adds no marginal predictive information beyond what Frequency already captures. |
    | **AvgOrderValue** | ❌ Excluded | Low Gini importance in feature selection trials. Spend per order adds little signal beyond total Monetary value when churn is the target. |
    | **Frequency** | ✅ Used | How often a customer buys is a strong independent signal of engagement. |
    | **Monetary** | ✅ Used | Total lifetime value captures the economic relationship with the customer. |
    """)

# ── SECTION 4: CHURN LABEL METHODOLOGY ──────────────────────
st.markdown("---")
st.subheader("4. Churn Label Methodology")
st.markdown("""
The churn label is engineered from Recency using a **70th percentile threshold**:

> Customers whose last purchase was more than **X days ago** (X = 70th percentile of Recency)
> are labelled **Churned = 1**. All others are **Loyal = 0**.

**Why 70th percentile?** A sensitivity analysis across three thresholds confirms this choice
produces the most balanced and business-realistic split:

| Threshold | Churned | Loyal | Notes |
|---|---|---|---|
| 60th percentile | ~40% churned | ~60% loyal | Too many customers flagged; over-spends on win-back campaigns |
| **70th percentile** | **~30% churned** | **~70% loyal** | **Matches realistic e-commerce churn rates (25–35%)** ✅ |
| 80th percentile | ~20% churned | ~80% loyal | Under-flags churn; misses too many at-risk customers |
""")
