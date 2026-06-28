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
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import QuantileTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def assign_persona(cluster_id, cluster_stats):
    """
    FIX: Assign persona using all 3 RFM dimensions, not just Monetary.
    Each cluster gets a composite score: high Frequency + high Monetary + low Recency = best.
    The cluster with the highest composite score = Top-Tier, lowest = At-Risk.
    """
    # Normalise each stat to 0-1 range across clusters
    def norm(series):
        rng = series.max() - series.min()
        return (series - series.min()) / rng if rng > 0 else series * 0

    freq_n  =  norm(cluster_stats["Frequency"])   # higher = better
    mon_n   =  norm(cluster_stats["Monetary"])     # higher = better
    rec_n   = -norm(cluster_stats["Recency"])      # lower recency = better (negate)

    composite = freq_n + mon_n + rec_n  # range roughly -1 to 3

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

    log.info("=" * 60)
    log.info("  ENTERPRISE AI — DUAL MODEL TRAINING PIPELINE")
    log.info("=" * 60)
    log.info("DB  : %s", DB_PATH)
    log.info("PKL : %s", APP_DIR)
    log.info("PNG : %s", SCRIPT_DIR)

    # ── [1/7] Load Data ─────────────────────────────────────────
    log.info("[1/7] Loading database...")
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Run etl/1_database_setup.py then etl/2_feature_engineering.py first."
        )

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM customer_features", conn)
    log.info("Loaded %d customers.", len(df))

    # ── [2/7] Scale Features ─────────────────────────────────────
    log.info("[2/7] Scaling features with QuantileTransformer...")
    # NOTE: This produces scaler.pkl — it is a QuantileTransformer, NOT a RobustScaler.
    # The README previously stated RobustScaler incorrectly; that has been corrected.
    rfm_features = ["Recency", "Frequency", "Monetary"]
    X_rfm        = df[rfm_features]
    y            = df["Churn_Label"]

    scaler   = QuantileTransformer(output_distribution="normal", random_state=42)
    X_scaled = scaler.fit_transform(X_rfm)

    # ── [3/7] K-Means Clustering ─────────────────────────────────
    log.info("[3/7] Training K-Means & generating validation chart...")
    inertia_vals    = []
    silhouette_vals = []
    K_range         = range(2, 9)

    for k in K_range:
        km  = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        lbl = km.fit_predict(X_scaled)
        inertia_vals.append(km.inertia_)
        silhouette_vals.append(silhouette_score(X_scaled, lbl))

    fig_val, ax_val = plt.subplots(1, 2, figsize=(12, 4))
    optimal_k = 3

    ax_val[0].plot(K_range, inertia_vals, marker="o", color="steelblue")
    ax_val[0].axvline(x=optimal_k, color="red", linestyle="--", label=f"K={optimal_k} chosen")
    ax_val[0].set_title("Elbow Method (Inertia)")
    ax_val[0].set_xlabel("Number of Clusters (K)")
    ax_val[0].set_ylabel("Inertia")
    ax_val[0].legend()
    ax_val[0].grid(True, alpha=0.3)

    ax_val[1].plot(K_range, silhouette_vals, marker="s", color="darkorange")
    ax_val[1].axvline(x=optimal_k, color="red", linestyle="--", label=f"K={optimal_k} chosen")
    ax_val[1].set_title("Silhouette Score")
    ax_val[1].set_xlabel("Number of Clusters (K)")
    ax_val[1].set_ylabel("Score")
    ax_val[1].legend()
    ax_val[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "clustering_validation.png"), dpi=150)
    plt.close()
    log.info("clustering_validation.png saved.")

    kmeans       = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # FIX: persona assignment now uses all 3 RFM dimensions via composite score
    cluster_stats  = df.groupby("Cluster")[rfm_features].mean()
    persona_labels = assign_persona(None, cluster_stats)
    df["Persona"]  = df["Cluster"].map(persona_labels)

    log.info("Persona distribution:\n%s", df["Persona"].value_counts().to_string())

    # ── [4/7] Random Forest Churn Predictor ──────────────────────
    log.info("[4/7] Training Random Forest Churn Predictor...")
    # Recency excluded: churn label is derived from Recency → data leakage if included.
    # TotalQuantity excluded: correlates strongly with Frequency (r > 0.9), no marginal gain.
    # AvgOrderValue excluded: low Gini importance in feature selection trials.
    X_churn = df[["Frequency", "Monetary"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X_churn, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        random_state=42,
        class_weight="balanced",
    )
    rf_model.fit(X_train, y_train)

    # FIX: 5-fold cross-validation for robust evaluation
    log.info("[4b/7] Running 5-fold cross-validation...")
    cv_f1_scores = cross_val_score(rf_model, X_churn, y, cv=5, scoring="f1")
    log.info(
        "Cross-Validation F1: %.4f ± %.4f (per fold: %s)",
        cv_f1_scores.mean(),
        cv_f1_scores.std(),
        np.round(cv_f1_scores, 4),
    )

    # ── [5/7] Evaluation Metrics ─────────────────────────────────
    log.info("[5/7] Generating evaluation metrics...")

    y_proba          = rf_model.predict_proba(X_test)[:, 1]
    custom_threshold = 0.35
    y_pred_custom    = (y_proba >= custom_threshold).astype(int)
    y_pred_default   = (y_proba >= 0.50).astype(int)
    roc_auc          = roc_auc_score(y_test, y_proba)

    accuracy  = accuracy_score(y_test, y_pred_custom)
    precision = precision_score(y_test, y_pred_custom, zero_division=0)
    recall    = recall_score(y_test, y_pred_custom, zero_division=0)
    f1        = f1_score(y_test, y_pred_custom, zero_division=0)

    default_report = classification_report(y_test, y_pred_default, output_dict=True)
    base_f1        = default_report["weighted avg"]["f1-score"]

    log.info("Standard Threshold (0.50) — Weighted Avg F1: %.4f", base_f1)
    log.info("Aggressive Threshold (0.35) — Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | AUC: %.4f",
             accuracy, precision, recall, f1, roc_auc)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm   = confusion_matrix(y_test, y_pred_custom)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Active", "Churned"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"Confusion Matrix (Threshold: {custom_threshold})", fontsize=12)

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
    log.info("model_evaluation.png saved.")

    # ── [6/7] Feature Importance Chart ───────────────────────────
    log.info("[6/7] Generating feature importance chart...")

    fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
    bars = ax_fi.barh(
        ["Frequency", "Monetary"],
        rf_model.feature_importances_,
        color=["#636EFA", "#EF553B"],
    )
    ax_fi.set_title("Gini Feature Importance (Churn Model)")
    ax_fi.set_xlabel("Importance Score")
    ax_fi.bar_label(bars, fmt="%.3f", padding=4)
    ax_fi.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    log.info("feature_importance.png saved.")

    # ── [7/7] Save All Assets ────────────────────────────────────
    log.info("[7/7] Saving AI assets to app/ folder...")

    joblib.dump(kmeans,         os.path.join(APP_DIR, "persona_model.pkl"))
    joblib.dump(rf_model,       os.path.join(APP_DIR, "churn_model.pkl"))
    joblib.dump(scaler,         os.path.join(APP_DIR, "scaler.pkl"))
    joblib.dump(persona_labels, os.path.join(APP_DIR, "persona_label_map.pkl"))
    df.to_pickle(               os.path.join(APP_DIR, "historical_data.pkl"))

    # FIX: save metrics as JSON so the Streamlit UI can display live numbers
    metrics = {
        "threshold_aggressive": custom_threshold,
        "accuracy":             round(accuracy, 4),
        "precision":            round(precision, 4),
        "recall":               round(recall, 4),
        "f1_score":             round(f1, 4),
        "roc_auc":              round(roc_auc, 4),
        "baseline_f1_050":      round(base_f1, 4),
        "cv_f1_mean":           round(float(cv_f1_scores.mean()), 4),
        "cv_f1_std":            round(float(cv_f1_scores.std()), 4),
        "n_customers":          len(df),
        "optimal_k":            optimal_k,
    }
    metrics_path = os.path.join(APP_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("metrics.json saved to app/")

    log.info("=" * 60)
    log.info("All assets saved to app/:")
    log.info("  persona_model.pkl, churn_model.pkl, scaler.pkl")
    log.info("  persona_label_map.pkl, historical_data.pkl, metrics.json")
    log.info("Charts saved to models/:")
    log.info("  clustering_validation.png, model_evaluation.png, feature_importance.png")
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        log.error("Training pipeline failed: %s", e)
        sys.exit(1)
