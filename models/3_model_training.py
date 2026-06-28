import json
import logging
import os
import sqlite3
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # must be before pyplot import
import matplotlib.pyplot as plt

import joblib
import shap
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import QuantileTransformer

# Import the custom benchmarking script
from benchmark_models import benchmark_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def assign_persona(cluster_id, cluster_stats):
    """
    Assign persona using all 3 RFM dimensions, not just Monetary.
    Each cluster gets a composite score: high Frequency + high Monetary + low Recency = best.
    """
    def norm(series):
        rng = series.max() - series.min()
        return (series - series.min()) / rng if rng > 0 else series * 0

    freq_n  =  norm(cluster_stats["Frequency"])   
    mon_n   =  norm(cluster_stats["Monetary"])     
    rec_n   = -norm(cluster_stats["Recency"])      

    composite = freq_n + mon_n + rec_n  

    ranked = composite.sort_values()
    labels = {
        int(ranked.index[0]): "At-Risk Sleepers",
        int(ranked.index[1]): "Promising Newcomers",
        int(ranked.index[2]): "Top-Tier Customers",
    }
    return labels


def run():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    APP_DIR    = os.path.join(ROOT_DIR, "app")
    DB_PATH    = os.path.join(APP_DIR, "enterprise_crm.db")
    
    # Ensure SCRIPT_DIR is in the path to import benchmark_models if run from root
    if SCRIPT_DIR not in sys.path:
        sys.path.append(SCRIPT_DIR)

    log.info("=" * 60)
    log.info("  ENTERPRISE AI — ADVANCED DUAL MODEL TRAINING PIPELINE")
    log.info("=" * 60)

    # ── [1/8] Load Data ─────────────────────────────────────────
    log.info("[1/8] Loading database...")
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}.")

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM customer_features", conn)
    log.info("Loaded %d customers.", len(df))

    # ── [2/8] Scale Features & Cluster ──────────────────────────
    log.info("[2/8] Scaling features & K-Means Clustering...")
    rfm_features = ["Recency", "Frequency", "Monetary"]
    X_rfm        = df[rfm_features]
    y            = df["Churn_Label"]

    scaler   = QuantileTransformer(output_distribution="normal", random_state=42)
    X_scaled = scaler.fit_transform(X_rfm)

    optimal_k = 3
    kmeans    = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_stats  = df.groupby("Cluster")[rfm_features].mean()
    persona_labels = assign_persona(None, cluster_stats)
    df["Persona"]  = df["Cluster"].map(persona_labels)

    # ── [3/8] Benchmarking Models ───────────────────────────────
    log.info("[3/8] Running Automated Model Benchmarking...")
    # Strictly 2 features to prevent Streamlit crashes
    X_churn = df[["Frequency", "Monetary"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X_churn, y, test_size=0.2, random_state=42, stratify=y
    )

    best_model, best_name, benchmark_df = benchmark_models(X_train, y_train, SCRIPT_DIR)
    log.info("🏆 Best Model Chosen: %s", best_name)

    # ── [4/8] Hyperparameter Tuning ─────────────────────────────
    log.info("[4/8] Tuning hyperparameters for %s...", best_name)
    if best_name in ["Random Forest", "Decision Tree", "Extra Trees"]:
        param_grid = {'max_depth': [5, 8, 12], 'min_samples_split': [5, 10]}
        grid = GridSearchCV(best_model, param_grid, cv=3, scoring='f1')
        grid.fit(X_train, y_train)
        final_model = grid.best_estimator_
        log.info("Tuned Params: %s", grid.best_params_)
    else:
        final_model = best_model
        final_model.fit(X_train, y_train)
        log.info("Skipped deep tuning for %s (using defaults).", best_name)

    # ── [5/8] SHAP Explainability ───────────────────────────────
    log.info("[5/8] Generating SHAP Explainability Plot...")
    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_test)
        
        plt.figure(figsize=(8, 5))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_test, show=False)
        else:
             shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(SCRIPT_DIR, "shap_summary.png"), dpi=150)
        plt.close()
        log.info("shap_summary.png saved.")
    except Exception as e:
        log.warning("SHAP generation skipped (Model %s may not be tree-based): %s", best_name, e)

    # ── [6/8] Error Analysis (Hard Customers) ───────────────────
    log.info("[6/8] Performing Error Analysis...")
    y_pred = final_model.predict(X_test)
    error_df = X_test.copy()
    error_df['Actual'] = y_test
    error_df['Predicted'] = y_pred

    misclassified = error_df[error_df['Actual'] != error_df['Predicted']]
    misclassified.to_csv(os.path.join(SCRIPT_DIR, 'misclassified_customers.csv'), index=False)
    log.info("Found %d misclassified customers. Saved to misclassified_customers.csv", len(misclassified))

    # ── [7/8] Evaluation Metrics & Visuals ──────────────────────
    log.info("[7/8] Generating evaluation metrics & visuals...")
    y_proba          = final_model.predict_proba(X_test)[:, 1]
    custom_threshold = 0.35
    y_pred_custom    = (y_proba >= custom_threshold).astype(int)
    roc_auc          = roc_auc_score(y_test, y_proba)

    accuracy  = accuracy_score(y_test, y_pred_custom)
    precision = precision_score(y_test, y_pred_custom, zero_division=0)
    recall    = recall_score(y_test, y_pred_custom, zero_division=0)
    f1        = f1_score(y_test, y_pred_custom, zero_division=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm   = confusion_matrix(y_test, y_pred_custom)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Active", "Churned"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"Confusion Matrix ({best_name})", fontsize=12)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve Analysis")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "model_evaluation.png"), dpi=150)
    plt.close()
    
    # Generate Feature Importance if supported
    if hasattr(final_model, 'feature_importances_'):
        fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
        bars = ax_fi.barh(
            ["Frequency", "Monetary"],
            final_model.feature_importances_,
            color=["#636EFA", "#EF553B"],
        )
        ax_fi.set_title(f"Gini Feature Importance ({best_name})")
        ax_fi.bar_label(bars, fmt="%.3f", padding=4)
        ax_fi.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SCRIPT_DIR, "feature_importance.png"), dpi=150)
        plt.close()

    # ── [8/8] Save All Assets ────────────────────────────────────
    log.info("[8/8] Saving AI assets to app/ folder...")

    joblib.dump(kmeans,         os.path.join(APP_DIR, "persona_model.pkl"))
    joblib.dump(final_model,    os.path.join(APP_DIR, "churn_model.pkl")) # Saved as the expected name
    joblib.dump(scaler,         os.path.join(APP_DIR, "scaler.pkl"))
    joblib.dump(persona_labels, os.path.join(APP_DIR, "persona_label_map.pkl"))
    df.to_pickle(               os.path.join(APP_DIR, "historical_data.pkl"))

    metrics = {
        "best_model_architecture": best_name,
        "threshold_aggressive": custom_threshold,
        "accuracy":             round(accuracy, 4),
        "precision":            round(precision, 4),
        "recall":               round(recall, 4),
        "f1_score":             round(f1, 4),
        "roc_auc":              round(roc_auc, 4),
        "n_customers":          len(df),
    }
    with open(os.path.join(APP_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("✅ Pipeline complete! Check %s for the generated CSVs and PNGs.", SCRIPT_DIR)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        log.error("Training pipeline failed: %s", e)
        sys.exit(1)