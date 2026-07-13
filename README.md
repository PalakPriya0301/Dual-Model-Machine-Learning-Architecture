# 📊 Enterprise AI-Driven CRM Platform: Dual-Model Machine Learning Architecture

An end-to-end Data Science and Predictive Analytics solution built as a graduation major project for the Bachelor of Computer Applications (BCA) program. This platform engineers raw transaction logs into actionable customer profiles and deploys a predictive engine to forecast and mitigate customer churn in real time.

## 🚀 Live Application

The platform is fully deployed and accessible via Streamlit Community Cloud:
https://dual-model-machine-learning-architecture.streamlit.app/

---

## 🛠️ System Architecture & Workflow

The platform follows a modular, industry-standard data science pipeline distributed across three core operational phases:

1. **ETL & Data Engineering (`etl/`):** Consolidates raw e-commerce transaction data, handles system-specific order anomalies, and persists data into a structured SQLite database (`enterprise_crm.db`). Executes an RFM (Recency, Frequency, Monetary) feature extraction pipeline.

2. **Machine Learning Pipeline (`models/`):**
   - **Unsupervised Learning:** Automatically segments customers into distinct operational personas using an optimised K-Means Clustering algorithm (K=3, validated via Elbow Method and Silhouette Analysis).
   - **Supervised Learning:** Trains an ensemble Random Forest Classifier (300 estimators) to identify high-risk customer abandonment before it occurs.

3. **Enterprise UI Dashboard (`app/`):** A production-ready Streamlit multi-page interface that surfaces customer health metrics, provides interactive single-customer inference, and connects to an automated email marketing delivery node.

---

## 📂 Project Structure

```text
├── app/
│   ├── Home.py                        # Main multi-page entry point & MLOps panel
│   ├── enterprise_crm.db              # Live SQLite application database
│   ├── churn_model.pkl                # Trained Random Forest classifier
│   ├── persona_model.pkl              # Trained K-Means clustering model
│   ├── scaler.pkl                     # QuantileTransformer instance for feature scaling
│   ├── persona_label_map.pkl          # Cluster ID → persona name mapping
│   ├── historical_data.pkl            # Pre-scored customer profiles (cached)
│   ├── metrics.json                   # Live model performance numbers (auto-generated)
│   └── pages/
│       ├── 1_Dashboard.py             # High-level CRM descriptive analytics
│       ├── 2_Predict.py               # Real-time single-customer inference + SHAP
│       ├── 3_Training_Data.py         # Training data archive & download
│       ├── 4_Marketing_Assistant.py   # Persona-based email campaign dispatcher
│       ├── 5_Model_Metrics.py         # ML performance evaluation & methodology
│       └── 6_ETL.py                   # Dynamic upload & live clustering engine
├── etl/
│   ├── 1_database_setup.py            # Database initialisation and order cleaning
│   └── 2_feature_engineering.py       # RFM aggregation and churn labelling
├── models/
│   ├── 3_model_training.py            # Training script for K-Means and Random Forest
│   ├── clustering_validation.png      # Elbow and Silhouette evaluation charts
│   ├── feature_importance.png         # Gini importance bar chart
│   └── model_evaluation.png           # Confusion Matrix and ROC-AUC curve
├── requirements.txt                   # Pinned dependency manifest
└── README.md                          # Project documentation
```

---

## 🧠 Machine Learning & Methodology

### 1. Customer Segmentation (K-Means Clustering)

- **Features Used:** Recency, Frequency, and Monetary value — scaled using `QuantileTransformer` (output distribution: normal).
- **Hyperparameter Optimisation:** K validated via the **Elbow Method** (inertia) and **Silhouette Analysis** across K = 2–8. Both metrics confirm **K=3** as optimal.
- **Persona Assignment:** Clusters are labelled using a **composite RFM score** — weighting high Frequency, high Monetary, and low Recency together. This ensures all three dimensions inform persona naming rather than Monetary value alone.
- **Identified Archetypes (K=3):**
  - **Top-Tier Customers:** High frequency, high monetary value, low recency (recently active).
  - **Promising Newcomers:** Moderate recency, moderate frequency, lower initial spend.
  - **At-Risk Sleepers:** High recency (long-absent), declining frequency.

### 2. Predictive Churn Engine (Random Forest)

- **Algorithm:** Random Forest Classifier — 300 estimators, `max_depth=8`, `min_samples_split=10`, `class_weight='balanced'`.
- **Features Used:** `Frequency` and `Monetary` only.
- **Feature Exclusion Rationale:**

  | Feature | Decision | Reason |
  |---|---|---|
  | Recency | Excluded | Data leakage — churn label is derived from Recency |
  | TotalQuantity | Excluded | Pearson correlation with Frequency > 0.90; no marginal gain |
  | AvgOrderValue | Excluded | Low Gini importance in feature selection trials |
  | Frequency | Used | Strong independent signal of customer engagement |
  | Monetary | Used | Captures economic relationship with the customer |

- **Churn Label Definition:** Customers whose Recency exceeds the **70th percentile** are labelled Churned (=1). All others are Loyal (=0).

  | Threshold | Churned | Loyal | Notes |
  |---|---|---|---|
  | 60th percentile | ~40% | ~60% | Over-flags; too many win-back campaigns |
  | **70th percentile** | **~30%** | **~70%** | **Matches real-world e-commerce churn rates ✅** |
  | 80th percentile | ~20% | ~80% | Under-flags; misses too many at-risk customers |

- **Validation:** 5-fold cross-validation (F1 mean ± std) used alongside a single 80/20 stratified train-test split for robust evaluation.

- **Operational Decision Threshold (0.35):** Lowered from the default 0.50 because a **False Negative** (missing a real churner) costs far more than a **False Positive** (sending a discount to a loyal customer). This maximises **Recall** at an acceptable precision trade-off.

---

## 💻 Tech Stack & Dependencies

All versions are pinned in `requirements.txt` for reproducible deployments.

| Package | Purpose |
|---|---|
| `streamlit` | Multi-page web application framework |
| `pandas`, `numpy` | Data engineering and numerical computation |
| `scikit-learn` | K-Means clustering and Random Forest classifier |
| `plotly` | Interactive 3D and 2D visualisations |
| `matplotlib` | Static training charts (Elbow, ROC, Confusion Matrix) |
| `shap` | Model explainability (SHAP waterfall on Predict page) |
| `joblib` | Model serialisation (`.pkl` files) |
| `python-dotenv` | Local environment variable management |
| `openpyxl` | Excel file ingestion support |
| `Pillow` | PNG chart rendering in Streamlit |

---

## 🔧 Installation & Local Setup

**1. Clone the repository:**
```bash
git clone https://github.com/PalakPriya0301/Dual-Model-Machine-Learning-Architecture.git
cd Dual-Model-Machine-Learning-Architecture
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Place the raw dataset in the `data/` folder:**

Download `Online Retail.xlsx` from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/352/online+retail) and place it at:
```
data/Online Retail.xlsx
```

**4. Run the ETL pipeline:**
```bash
python etl/1_database_setup.py
python etl/2_feature_engineering.py
```

**5. Train the models:**
```bash
python models/3_model_training.py
```
This generates all `.pkl` files and `metrics.json` in `app/`.

**6. Launch the application:**
```bash
streamlit run app/Home.py
```

---

## 🔒 Security & Deployment Notes

- **Secrets Management:** `SENDER_EMAIL` and `APP_PASSWORD` (Gmail SMTP credentials) are managed via Streamlit Secrets on cloud and a `.env` file locally. Neither is committed to version control.
- **Raw Data:** `Online Retail.xlsx` is excluded from GitHub (file size). The compiled `.db` database and all `.pkl` model files are committed instead so the cloud deployment is fully functional without the source file.

---

## 📁 Generated Assets (after running training)

After running all scripts, the following files are auto-generated and **should be committed to GitHub** so the Streamlit Cloud deployment works:

```
app/enterprise_crm.db
app/churn_model.pkl
app/persona_model.pkl
app/scaler.pkl
app/persona_label_map.pkl
app/historical_data.pkl
app/metrics.json
models/clustering_validation.png
models/feature_importance.png
models/model_evaluation.png
```

---

## 👩‍💻 Author

**Palak Priya** — BCA Major Project
GitHub: [PalakPriya0301](https://github.com/PalakPriya0301)
