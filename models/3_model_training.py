import pandas as pd
import sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import warnings
import os
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

print("=" * 60)
print("  ENTERPRISE AI — DUAL MODEL TRAINING PIPELINE")
print("=" * 60)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
APP_DIR    = os.path.join(ROOT_DIR, "app")
DB_PATH    = os.path.join(APP_DIR, "enterprise_crm.db")

print(f"\n   DB  : {DB_PATH}")
print(f"   PKL : {APP_DIR}")
print(f"   PNG : {SCRIPT_DIR}")

print(f"\n [1/7] Loading database...")

if not os.path.exists(DB_PATH):
    print(f" ERROR: Database not found at {DB_PATH}")
    print("   Run etl/1_database_setup.py then etl/2_feature_engineering.py first.")
    exit()

try:
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM customer_features", conn)
    conn.close()
    print(f"   ✅ Loaded {len(df)} customers.")
except Exception as e:
    print(f"  ERROR: {e}")
    exit()


print("\n[2/7] Scaling features...")
features = ['Recency', 'Frequency', 'Monetary']
X = df[features]
y = df['Churn_Label']

scaler   = QuantileTransformer(output_distribution='normal', random_state=42)
X_scaled = scaler.fit_transform(X)

print("\n🧠 [3/7] Training K-Means & generating validation chart...")

inertia_vals    = []
silhouette_vals = []
K_range         = range(2, 9)

for k in K_range:
    km  = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    lbl = km.fit_predict(X_scaled)
    inertia_vals.append(km.inertia_)
    silhouette_vals.append(silhouette_score(X_scaled, lbl))

fig_val, ax_val = plt.subplots(1, 2, figsize=(12, 4))

ax_val[0].plot(K_range, inertia_vals, marker='o', color='steelblue')
ax_val[0].set_title('Elbow Method (Inertia)')
ax_val[0].set_xlabel('Number of Clusters (K)')
ax_val[0].set_ylabel('Inertia')
ax_val[0].grid(True, alpha=0.3)

ax_val[1].plot(K_range, silhouette_vals, marker='s', color='darkorange')
ax_val[1].set_title('Silhouette Score')
ax_val[1].set_xlabel('Number of Clusters (K)')
ax_val[1].set_ylabel('Score')
ax_val[1].grid(True, alpha=0.3)

optimal_k = 3

ax_val[0].axvline(
    x=optimal_k,
    color='red',
    linestyle='--',
    label=f'K={optimal_k} chosen'
)

ax_val[1].axvline(
    x=optimal_k,
    color='red',
    linestyle='--',
    label=f'K={optimal_k} chosen'
)

ax_val[0].legend()
ax_val[1].legend()

kmeans = KMeans(
    n_clusters=optimal_k,
    init='k-means++',
    random_state=42,
    n_init=10
)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'clustering_validation.png'), dpi=150)
plt.close()

print("   clustering_validation.png saved.")

df['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_means = df.groupby('Cluster')['Monetary'].mean().sort_values()

persona_labels = {
    int(cluster_means.index[0]): "At-Risk Sleepers",
    int(cluster_means.index[1]): "Promising Newcomers",
    int(cluster_means.index[2]): "Top-Tier Customers"
}
df['Persona'] = df['Cluster'].map(persona_labels)
print(df['Persona'].value_counts())


print("\n [4/7] Training Random Forest Churn Predictor...")
X_churn = df[['Frequency', 'Monetary', 'TotalQuantity', 'AvgOrderValue']]
X_train, X_test, y_train, y_test = train_test_split(
    X_churn, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=10,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)


print("\n [5/7] Generating evaluation metrics...")

y_proba          = rf_model.predict_proba(X_test)[:, 1]
custom_threshold = 0.35  # selected to improve recall and identify more at-risk customers
y_pred_custom    = (y_proba >= custom_threshold).astype(int)
roc_auc          = roc_auc_score(y_test, y_proba)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm   = confusion_matrix(y_test, y_pred_custom)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Active', 'Churned'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f'Confusion Matrix (Threshold: {custom_threshold})', fontsize=12)

fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve Analysis')
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'model_evaluation.png'), dpi=150)
plt.close()
print("   model_evaluation.png saved.")

print("\n [6/7] Generating feature importance chart...")

fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
bars = ax_fi.barh(
    ['Frequency', 'Monetary', 'Total Quantity', 'Avg Order Value'],
    rf_model.feature_importances_,
    color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
)
ax_fi.set_title('Gini Feature Importance (Churn Model)')
ax_fi.set_xlabel('Importance Score')
ax_fi.bar_label(bars, fmt='%.3f', padding=4)
ax_fi.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'feature_importance.png'), dpi=150)
plt.close()
print("   feature_importance.png saved.")

print("\n [7/7] Saving AI assets to app/ folder...")

joblib.dump(kmeans,         os.path.join(APP_DIR, "persona_model.pkl"))
joblib.dump(rf_model,       os.path.join(APP_DIR, "churn_model.pkl"))
joblib.dump(scaler,         os.path.join(APP_DIR, "scaler.pkl"))
joblib.dump(persona_labels, os.path.join(APP_DIR, "persona_label_map.pkl"))
df.to_pickle(               os.path.join(APP_DIR, "historical_data.pkl"))


y_pred_default = (y_proba >= 0.50).astype(int)
base_report = classification_report(y_test, y_pred_default, output_dict=True)
base_f1 = base_report['weighted avg']['f1-score']


accuracy  = accuracy_score(y_test, y_pred_custom)
precision = precision_score(y_test, y_pred_custom)
recall    = recall_score(y_test, y_pred_custom)
f1        = f1_score(y_test, y_pred_custom)



print("\n" + "═" * 60)
print(f" Standard Threshold: 0.50")
print("═" * 60)
print(f"   Weighted Avg F1     : {base_f1:.4f}")

print("\n" + "═" * 60)
print(f"(Aggressive Threshold: {custom_threshold})")
print("═" * 60)
print(f"   Accuracy            : {accuracy:.4f}")
print(f"   Precision           : {precision:.4f}")
print(f"   Recall              : {recall:.4f}")
print(f"   F1-Score            : {f1:.4f}")
print("═" * 60)
print("\n Assets saved to app/:")
print("   persona_model.pkl, churn_model.pkl, scaler.pkl")
print("   persona_label_map.pkl, historical_data.pkl")
print("\nCharts saved to models/:")
print("   clustering_validation.png, model_evaluation.png, feature_importance.png")