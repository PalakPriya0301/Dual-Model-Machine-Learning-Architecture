import pandas as pd
import sqlite3
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def run_clustering_benchmark(db_path="enterprise_crm.db"):
    print("--- Starting Clustering Benchmark ---")
    
    # 1. Load Data
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM customer_features", conn)
    conn.close()

    # We only cluster on RFM features
    rfm_features = df[['Recency', 'Frequency', 'Monetary']]

    # 2. Scale the Data (Crucial for distance-based algorithms)
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    scaled_rfm = scaler.fit_transform(rfm_features)

    # We already know K=3 from our Elbow Method analysis
    optimal_k = 3
    results = {}

    # 3. Model 1: K-Means (Centroid-based)
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(scaled_rfm)
    results['K-Means'] = silhouette_score(scaled_rfm, kmeans_labels)

    # 4. Model 2: Agglomerative (Hierarchical)
    agglo = AgglomerativeClustering(n_clusters=optimal_k)
    agglo_labels = agglo.fit_predict(scaled_rfm)
    results['Agglomerative'] = silhouette_score(scaled_rfm, agglo_labels)

    # 5. Model 3: Gaussian Mixture Model (Probabilistic)
    gmm = GaussianMixture(n_components=optimal_k, random_state=42)
    gmm_labels = gmm.fit_predict(scaled_rfm)
    results['Gaussian Mixture'] = silhouette_score(scaled_rfm, gmm_labels)

    # 6. Model 4: DBSCAN (Density-based)
    # DBSCAN doesn't take 'k', it takes eps and min_samples. We estimate standard params.
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(scaled_rfm)
    
    # DBSCAN might assign everything to noise (-1), so we handle that error
    if len(set(dbscan_labels)) > 1:
        results['DBSCAN'] = silhouette_score(scaled_rfm, dbscan_labels)
    else:
        results['DBSCAN'] = "Failed to find distinct clusters (Only Noise)"

    # 7. Print Results
    print("\n--- Silhouette Scores (Higher is Better) ---")
    for model_name, score in sorted(results.items(), key=lambda item: str(item[1]), reverse=True):
        if isinstance(score, float):
            print(f"{model_name}: {score:.4f}")
        else:
            print(f"{model_name}: {score}")

if __name__ == "__main__":
    run_clustering_benchmark()