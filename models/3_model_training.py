import os
import sqlite3
import json
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

# Import custom supervised modules
from benchmark_models import benchmark_models
from hyperparameter_tuning import tune_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def run_clustering_benchmark(scaled_rfm):
    """Benchmarks unsupervised models using a sample to prevent RAM memory crashes."""
    log.info("Running Unsupervised Clustering Benchmarks...")
    optimal_k = 3
    cluster_results = {}

    # === MEMORY FIX: Downsample for distance-based benchmarking ===
    MAX_SAMPLE = 3000 
    if len(scaled_rfm) > MAX_SAMPLE:
        log.info(f"Dataset too large for distance matrices. Downsampling to {MAX_SAMPLE} for benchmark comparison.")
        np.random.seed(42)
        indices = np.random.permutation(len(scaled_rfm))[:MAX_SAMPLE]
        benchmark_data = scaled_rfm[indices]
    else:
        benchmark_data = scaled_rfm

    # 1. K-Means
    kmeans_test = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    labels_km = kmeans_test.fit_predict(benchmark_data)
    cluster_results['K-Means'] = round(float(silhouette_score(benchmark_data, labels_km)), 4)

    # 2. Agglomerative (Hierarchical)
    agglo = AgglomerativeClustering(n_clusters=optimal_k)
    labels_agg = agglo.fit_predict(benchmark_data)
    cluster_results['Agglomerative'] = round(float(silhouette_score(benchmark_data, labels_agg)), 4)

    # 3. Gaussian Mixture
    gmm = GaussianMixture(n_components=optimal_k, random_state=42)
    labels_gmm = gmm.fit_predict(benchmark_data)
    cluster_results['Gaussian Mixture'] = round(float(silhouette_score(benchmark_data, labels_gmm)), 4)

    # 4. DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels_db = dbscan.fit_predict(benchmark_data)
    if len(set(labels_db)) > 1:
        cluster_results['DBSCAN'] = round(float(silhouette_score(benchmark_data, labels_db)), 4)
    else:
        cluster_results['DBSCAN'] = "Failed (Only Noise)"

    return cluster_results

def run():
    # --- 1. Path Configuration ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    APP_DIR    = os.path.join(ROOT_DIR, "app")
    DB_PATH    = os.path.join(APP_DIR, "enterprise_crm.db")

    log.info("Starting Fully Integrated Advanced Training Pipeline...")

    # --- 2. Data Ingestion ---
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM customer_features", conn)
    conn.close()
    
    y = df['Churn_Label']
    X_rfm = df[['Recency', 'Frequency', 'Monetary']] 
    X_churn = df[['Frequency', 'Monetary']] # Drop Recency to prevent data leakage

    # --- 3. Normalization ---
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_scaled = scaler.fit_transform(X_rfm)

    # --- 4. Unsupervised Benchmark & Execution ---
    clustering_scores = run_clustering_benchmark(X_scaled)
    log.info(f"Clustering Benchmark Results (Silhouette): {clustering_scores}")

    # Production run using the optimal algorithm (K-Means) on the FULL dataset
    log.info("Training final K-Means model on full dataset...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
    df['Cluster'] = kmeans.labels_

    # --- 5. Supervised Benchmarking ---
    log.info("Executing Supervised Algorithm Benchmarking...")
    X_train, X_test, y_train, y_test = train_test_split(X_churn, y, test_size=0.2, random_state=42, stratify=y)
    
    best_model, best_name, benchmark_df = benchmark_models(X_train, y_train, SCRIPT_DIR)
    benchmark_df.to_csv(os.path.join(APP_DIR, "benchmark_results.csv"), index=False)

    # --- 6. Hyperparameter Tuning ---
    log.info(f"Optimizing Selected Architecture: {best_name}")
    try:
        final_model, best_params, best_score = tune_model(best_name, best_model, X_train, y_train, SCRIPT_DIR)
    except Exception as e:
        log.warning(f"Tuning bypassed: {e}. Utilizing baseline configuration.")
        final_model = best_model.fit(X_train, y_train)

    # --- 7. Evaluation & Threshold Optimization ---
    y_proba = final_model.predict_proba(X_test)[:, 1]
    custom_threshold = 0.35
    y_pred_custom = (y_proba >= custom_threshold).astype(int)
    
    # --- 8. Explainable AI (SHAP) ---
    log.info("Generating SHAP feature explanations...")
    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_test)
        plt.figure(figsize=(10,6))
        shap_data = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(shap_data, X_test, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(APP_DIR, 'shap_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e: 
        log.error(f"SHAP generation skipped: {e}")

    # --- 9. Export Production Artifacts & Combined JSON ---
    log.info("Exporting models and JSON metrics to /app directory...")
    joblib.dump(kmeans, os.path.join(APP_DIR, "persona_model.pkl"))
    joblib.dump(final_model, os.path.join(APP_DIR, "churn_model.pkl"))
    joblib.dump(scaler, os.path.join(APP_DIR, "scaler.pkl"))
    
    metrics = {
        "best_model_architecture": best_name,
        "decision_threshold": custom_threshold,
        "n_customers": len(df),
        "clustering_benchmarks": clustering_scores,
        "performance": {
            "accuracy": round(float(accuracy_score(y_test, y_pred_custom)), 4),
            "precision": round(float(precision_score(y_test, y_pred_custom, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_custom, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred_custom, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4)
        }
    }
    
    with open(os.path.join(APP_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    log.info("Integrated execution complete. All systems go.")

if __name__ == "__main__":
    run()