# 📊 Enterprise AI-Driven CRM Platform: Dual-Model Machine Learning Architecture

An end-to-end Data Science and Predictive Analytics solution built as a graduation major project for the Bachelor of Computer Applications (BCA) program. This platform engineers raw transaction logs into actionable customer profiles and deploys a predictive engine to forecast and mitigate customer churn in real time.

## 🚀 Live Application

The platform is fully deployed and accessible via Streamlit Community Cloud:  
https://dual-model-machine-learning-architecture.streamlit.app/

---

## 🛠️ System Architecture & Workflow

The platform follows a modular, industry-standard data science pipeline distributed across three core operational phases:

1. **ETL & Data Engineering (`etl/`):** Consolidates raw e-commerce transaction data, handles system-specific order anomalies, and persists data into a structured SQLite database (`enterprise_crm.db`). It executes an RFM (Recency, Frequency, Monetary) feature extraction pipeline.

2. **Machine Learning Pipeline (`models/`):** * **Unsupervised Learning:** Automatically segments customers into distinct operational personas using an optimized K-Means Clustering algorithm.
   * **Supervised Learning:** Trains an ensemble Random Forest Classifier to identify high-risk customer abandonment before it occurs.

3. **Enterprise UI Dashboard (`app/`):** A production-ready Streamlit interface that surfaces customer health metrics, provides interactive single-customer inference, and connects to an automated email marketing delivery node.


## 📂 Project Structure

```text
├── app/
│   ├── Home.py                       # Main multi-page entry point
│   ├── enterprise_crm.db             # Live SQLite application database
│   ├── churn_model.pkl               # Trained Random Forest classifier
│   ├── persona_model.pkl             # Trained K-Means clustering model
│   ├── scaler.pkl                    # RobustScaler instance for feature scaling
│   └── pages/
│       ├── 1_📋_Dashboard.py         # High-level CRM descriptive analytics
│       ├── 2_🔍_Customer_Lookup.py   # Single-customer RFM & history tracking
│       ├── 3_🔮_Predict_Churn.py     # Real-time inference engine
│       ├── 4_👤_Customer_Personas.py # K-Means behavioral archetype explorer
│       └── 5_📈_Model_Metrics.py     # Machine learning performance evaluation
├── etl/
│   ├── 1_database_setup.py           # Database initialization and order cleaning
│   └── 2_feature_engineering.py      # RFM aggregation and data preparation
├── models/
│   ├── 3_model_training.py           # Training script for K-Means and Random Forest
│   ├── clustering_validation.png     # Elbow and Silhouette evaluation charts
│   ├── feature_importance.png        # Gini importance bar chart
│   └── model_evaluation.png          # Confusion Matrix and ROC-AUC curve
├── requirements.txt                  # Cloud server installation manifest
└── README.md                         # Project documentation

```
## 🧠 Machine Learning & Methodology

### 1. Customer Segmentation (K-Means)
* **Features Used:** Log-transformed Recency, Frequency, and Monetary value.
* **Hyperparameter Optimization:** Validated via the **Elbow Method** (inertia metrics) and **Silhouette Analysis** across multiple initialization runs.
* **Identified Archetypes ($K=3$):**
  * **Top-Tier Customers:** High frequency, high monetary value, low recency.
  * **Promising Newcomers:** Low recency, moderate frequency, lower initial spend.
  * **At-Risk Sleepers:** Extremely high recency, declining frequency.

### 2. Predictive Churn Engine (Random Forest)
* **Algorithm Choice:** Random Forest Classifier utilizing 300 estimators, restricted max depth to prevent overfitting, and engineered with `class_weight='balanced'` to counter class imbalance.
* **Data Leakage Mitigation:** Recency metrics were strictly excluded from the classifier features because the operational churn label is derived from timeline boundaries. The engine is trained to predict risk purely based on purchasing behavior (Frequency, Monetary Value, and total quantity volume).
* **Operational Decision Threshold ($0.35$):** The classification threshold was intentionally lowered from 0.50 to 0.35. From a business standpoint, a **False Negative** (missing a customer who will churn) is exponentially more expensive than a **False Positive** (incentivizing a loyal customer). This operational optimization maximizes model **Recall**.

---

## 💻 Tech Stack & Dependencies

The complete core environment configuration is detailed in `requirements.txt`:

* **Core UI Framework:** `streamlit`
* **Data Engineering:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Interactive Visualization:** `plotly`, `matplotlib`
* **Model Explainability:** `shap`
* **Serialization & Environment:** `joblib`, `python-dotenv`

---

## 🔧 Installation & Local Setup

To clone and execute this architecture locally on your system:

**1. Clone the repository:**
```bash
git clone [https://github.com/PalakPriya0301/Dual-Model-Machine-Learning-Architecture.git](https://github.com/PalakPriya0301/Dual-Model-Machine-Learning-Architecture.git)
cd Dual-Model-Machine-Learning-Architecture
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the local database setup and feature engineering pipeline to generate the database:**
```bash
python etl/1_database_setup.py
python etl/2_feature_engineering.py
```


**4. Train the models locally:**
```bash
python models/3_model_training.py
```

**5. Launch the Streamlit application interface:**
```bash
streamlit run app/Home.py
```

---

## 🔒 Security & Deployment Notes

* **Secrets Management:** Environment variables (such as `SENDER_EMAIL` and `APP_PASSWORD`) are omitted from version control to prevent credential exposure. Deployment relies on protected cloud-injected secrets.
* **Storage Footprint:** Raw dataset sheets are excluded from GitHub storage tracking. Data persistency layer targets are compiled natively into the deployed `.db` SQLite target.