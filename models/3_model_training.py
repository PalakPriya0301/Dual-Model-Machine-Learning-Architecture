import pandas as pd
import sqlite3
import joblib
import json
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

def train_and_export_models(db_path="enterprise_crm.db"):
    print("--- Initiating Model Training Pipeline ---")
    
    # 1. Load Data
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM customer_features", conn)
    conn.close()

    # ==========================================
    # PHASE 1: UNSUPERVISED LEARNING (K-MEANS)
    # ==========================================
    print("Training K-Means Clustering Model...")
    rfm_features = df[['Recency', 'Frequency', 'Monetary']]
    
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    scaled_rfm = scaler.fit_transform(rfm_features)

    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_rfm)

    # Save Clustering Artifacts
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(kmeans, 'persona_model.pkl')
    print("K-Means Model and Scaler saved.")

    # ==========================================
    # PHASE 2: SUPERVISED LEARNING (RANDOM FOREST)
    # ==========================================
    print("\nTraining Random Forest Churn Classifier...")
    
    # Feature Selection: STRICTLY drop Recency to prevent data leakage
    X = df[['Frequency', 'Monetary']]
    y = df['Churn_Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Setup GridSearchCV for Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 8, 12],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=5, scoring='recall', n_jobs=-1)
    
    print("Running Grid Search Optimization (Optimizing for Recall)...")
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    print(f"Best Parameters Found: {grid_search.best_params_}")

    # Evaluate with lowered threshold (0.35) for aggressive churn catching
    probas = best_rf.predict_proba(X_test)[:, 1]
    y_pred_custom = (probas >= 0.35).astype(int)

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred_custom), 4),
        "Recall": round(recall_score(y_test, y_pred_custom), 4),
        "ROC-AUC": round(roc_auc_score(y_test, probas), 4)
    }

    # Save Classification Artifacts
    joblib.dump(best_rf, 'churn_model.pkl')
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)
        
    print(f"Random Forest Model saved. Metrics: {metrics}")
    print("--- Pipeline Complete ---")

if __name__ == "__main__":
    train_and_export_models()