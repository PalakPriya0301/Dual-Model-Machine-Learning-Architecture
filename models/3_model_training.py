import os
import sqlite3
import json
import logging
import joblib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

# Import your new modules
from benchmark_models import benchmark_models
from hyperparameter_tuning import tune_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def run():
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    APP_DIR    = os.path.join(ROOT_DIR, "app")
    DB_PATH    = os.path.join(APP_DIR, "enterprise_crm.db")

    log.info("Starting Advanced Training Pipeline...")

    # 1. Load Data
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM customer_features", conn)
    conn.close()
    
    # 2. Setup Features
    y = df['Churn_Label']
    X_rfm = df[['Recency', 'Frequency', 'Monetary']]
    X_churn = df[['Frequency', 'Monetary']] # Strict 2-feature limit for app stability

    # 3. K-Means
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_scaled = scaler.fit_transform(X_rfm)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
    df['Cluster'] = kmeans.labels_

    # 4. Benchmark Models
    X_train, X_test, y_train, y_test = train_test_split(X_churn, y, test_size=0.2, random_state=42, stratify=y)
    best_model, best_name, benchmark_df = benchmark_models(X_train, y_train, SCRIPT_DIR)
    
    # Save Benchmark results
    benchmark_df.to_csv(os.path.join(APP_DIR, "benchmark_results.csv"), index=False)
    benchmark_df.to_json(os.path.join(APP_DIR, "benchmark_results.json"), orient="records", indent=4)

    # 5. Hyperparameter Tuning
    try:
        final_model, best_params, best_score = tune_model(best_name, best_model, X_train, y_train, SCRIPT_DIR)
    except:
        final_model = best_model.fit(X_train, y_train)

    # 6. Evaluation & SHAP
    y_proba = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.35).astype(int)
    
    # Save SHAP
    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_test)
        plt.figure()
        shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, X_test, show=False)
        plt.savefig(os.path.join(SCRIPT_DIR, 'shap_summary.png'))
        plt.close()
    except: pass

    # 7. Save Assets
    joblib.dump(kmeans, os.path.join(APP_DIR, "persona_model.pkl"))
    joblib.dump(final_model, os.path.join(APP_DIR, "churn_model.pkl"))
    joblib.dump(scaler, os.path.join(APP_DIR, "scaler.pkl"))
    
    # Generate Metrics JSON
    metrics = {
        "best_model_architecture": best_name,
        "threshold_aggressive": 0.35,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "n_customers": len(df)
    }
    with open(os.path.join(APP_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("Training complete. Assets saved.")

if __name__ == "__main__":
    run()