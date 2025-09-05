
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import dump

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "app" / "models"
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

def generate_dummy_customers(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    centers = [
        [25, 30000, 70],   # young, low income, high spenders
        [45, 70000, 40],   # mid-age, mid income, moderate spending
        [65, 120000, 20],  # older, high income, low spending
    ]
    X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=[5, 8, 6], random_state=random_state)
    df = pd.DataFrame(X, columns=["age", "income", "spending_score"]).round(2)
    return df

def main():
    df = generate_dummy_customers()
    df.to_csv(DATA_DIR / "customers.csv", index=False)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[["age", "income", "spending_score"]].values)

    # Choose K via elbow/silhouette heuristics (fixed to 3 for this synthetic data)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    kmeans.fit(X_scaled)

    sil = silhouette_score(X_scaled, kmeans.labels_)
    print(f"Silhouette Score: {sil:.3f}")

    dump(scaler, MODELS_DIR / "scaler.pkl")
    dump(kmeans, MODELS_DIR / "kmeans_model.pkl")
    print("Artifacts saved to:", MODELS_DIR)

if __name__ == "__main__":
    main()
