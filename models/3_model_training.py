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
from sklearn.metrics import (
    silhouette_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    roc_curve, 
    auc
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

from benchmark_models import benchmark_models
from hyperparameter_tuning import tune_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def generate_cluster_validation_plot(scaled_rfm, output_dir):
    log.info("Generating cluster validation graph...")
    try:
        k_range = range(2, 9)
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            lbls = km.fit_predict(scaled_rfm)
            inertias.append(km.inertia_)
            silhouette_scores.append(silhouette_score(scaled_rfm, lbls))
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(k_range, inertias, marker='o', color='#2b7bba', linewidth=2)
        ax1.axvline(x=3, color='red', linestyle='--', label='K=3 chosen')
        ax1.set_title('Elbow Method (Inertia)')
        ax1.set_xlabel('Number of Clusters (K)')
        ax1.set_ylabel('Inertia')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(k_range, silhouette_scores, marker='s', color='#ff8c00', linewidth=2)
        ax2.axvline(x=3, color='red', linestyle='--', label='K=3 chosen')
        ax2.set_title('Silhouette Score')
        ax2.set_xlabel('Number of Clusters (K)')
        ax2.set_ylabel('Score')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'clustering_validation.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        log.error(f"Cluster validation graph generation skipped: {e}")


def generate_feature_importance_plot(model, feature_names, output_dir):
    log.info("Generating Gini feature importance graph...")
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            log.warning("Model does not have feature_importances_ attribute. Skipping plot.")
            return
            
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        imp_df = imp_df.sort_values("Importance", ascending=True)
        
        plt.figure(figsize=(8, 5))
        bars = plt.barh(imp_df["Feature"], imp_df["Importance"], color=['#ff6f59', '#5b73e8'])
        plt.title("Gini Feature Importance")
        plt.xlabel("")
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va='center')
            
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        log.error(f"Feature importance graph generation skipped: {e}")


def generate_evaluation_plots(y_test, y_pred, y_proba, output_dir, model_name):
    log.info("Generating evaluation graphs (Confusion Matrix & ROC)...")
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Active", "Churned"])
        disp.plot(ax=ax1, cmap=plt.cm.Blues, colorbar=False)
        ax1.set_title(f"Confusion Matrix ({model_name})")
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax2.set_xlim([-0.05, 1.05])
        ax2.set_ylim([-0.05, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve Analysis')
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_evaluation.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        log.error(f"Evaluation graph generation skipped: {e}")


def run_clustering_benchmark(scaled_rfm):
    log.info("Running Unsupervised Clustering Benchmarks...")
    optimal_k = 3
    cluster_results = {}
    MAX_SAMPLE = 3000
    
    if len(scaled_rfm) > MAX_SAMPLE:
        np.random.seed(42)
        indices = np.random.permutation(len(scaled_rfm))[:MAX_SAMPLE]
        benchmark_data = scaled_rfm[indices]
    else:
        benchmark_data = scaled_rfm
        
    kmeans_test = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    labels_km = kmeans_test.fit_predict(benchmark_data)
    cluster_results['K-Means'] = round(float(silhouette_score(benchmark_data, labels_km)), 4)
    
    agglo = AgglomerativeClustering(n_clusters=optimal_k)
    labels_agg = agglo.fit_predict(benchmark_data)
    cluster_results['Agglomerative'] = round(float(silhouette_score(benchmark_data, labels_agg)), 4)
    
    gmm = GaussianMixture(n_components=optimal_k, random_state=42)
    labels_gmm = gmm.fit_predict(benchmark_data)
    cluster_results['Gaussian Mixture'] = round(float(silhouette_score(benchmark_data, labels_gmm)), 4)
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels_db = dbscan.fit_predict(benchmark_data)
    if len(set(labels_db)) > 1:
        cluster_results['DBSCAN'] = round(float(silhouette_score(benchmark_data, labels_db)), 4)
    else:
        cluster_results['DBSCAN'] = "Failed (Only Noise)"
        
    return cluster_results


def run():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    APP_DIR = os.path.join(ROOT_DIR, "app")
    DB_PATH = os.path.join(APP_DIR, "enterprise_crm.db")
    
    log.info("Starting Fully Integrated Advanced Training Pipeline...")
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM customer_features", conn)
    conn.close()
    
    y = df['Churn_Label']
    X_rfm = df[['Recency', 'Frequency', 'Monetary']]
    X_churn = df[['Frequency', 'Monetary']]
    
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_scaled = scaler.fit_transform(X_rfm)
    
    clustering_scores = run_clustering_benchmark(X_scaled)
    log.info(f"Clustering Benchmark Results (Silhouette): {clustering_scores}")
    
    generate_cluster_validation_plot(X_scaled, SCRIPT_DIR)
    
    log.info("Training final K-Means model on full dataset...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
    df['Cluster'] = kmeans.labels_
    
    log.info("Executing Supervised Algorithm Benchmarking...")
    X_train, X_test, y_train, y_test = train_test_split(X_churn, y, test_size=0.2, random_state=42, stratify=y)
    
    best_model, best_name, benchmark_df = benchmark_models(X_train, y_train, SCRIPT_DIR)
    benchmark_df.to_csv(os.path.join(APP_DIR, "benchmark_results.csv"), index=False)
    
    log.info(f"Training model: {best_name}")
    final_model = best_model.fit(X_train, y_train)
    
    generate_feature_importance_plot(final_model, X_churn.columns, SCRIPT_DIR)
    
    y_proba = final_model.predict_proba(X_test)[:, 1]
    custom_threshold = 0.35
    y_pred_custom = (y_proba >= custom_threshold).astype(int)
    
    generate_evaluation_plots(y_test, y_pred_custom, y_proba, SCRIPT_DIR, best_name)
    
    log.info("Generating SHAP feature explanations...")
    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_test)
        
        plt.figure(figsize=(10,6))
        shap_data = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(shap_data, X_test, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(SCRIPT_DIR, 'shap_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        log.error(f"SHAP generation skipped: {e}")
        
    log.info("Exporting models and JSON metrics to /app directory...")
    joblib.dump(kmeans, os.path.join(APP_DIR, "persona_model.pkl"))
    joblib.dump(final_model, os.path.join(APP_DIR, "churn_model.pkl"))
    joblib.dump(scaler, os.path.join(APP_DIR, "scaler.pkl"))
    df.to_pickle(os.path.join(APP_DIR, "historical_data.pkl"))


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