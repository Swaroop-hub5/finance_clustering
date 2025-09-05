
from __future__ import annotations
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "customers.csv"
MODELS = ROOT / "app" / "models"

def main():
    df = pd.read_csv(DATA)
    scaler: StandardScaler = load(MODELS / "scaler.pkl")
    kmeans = load(MODELS / "kmeans_model.pkl")

    X_scaled = scaler.transform(df[["age", "income", "spending_score"]].values)
    labels = kmeans.predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)

    sizes = pd.Series(labels).value_counts().sort_index()
    print("Cluster sizes:\n", sizes.to_string())
    print(f"Silhouette Score: {sil:.3f}")
    print(f"Davies-Bouldin Index: {dbi:.3f}")

if __name__ == "__main__":
    main()
