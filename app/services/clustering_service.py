
from __future__ import annotations
from joblib import load
from pathlib import Path
import os
import numpy as np

MODELS_DIR = Path(os.getenv("MODELS_DIR", Path(__file__).resolve().parents[1] / "models"))

class ClusterService:
    def __init__(self) -> None:
        kmeans_path = MODELS_DIR / "kmeans_model.pkl"
        if not kmeans_path.exists():
            raise FileNotFoundError(f"KMeans model not found at {kmeans_path}. Run scripts/train.py first.")
        self.model = load(kmeans_path)

    def predict_cluster(self, x_scaled: np.ndarray) -> int:
        cluster = int(self.model.predict(x_scaled)[0])
        return cluster

    def nearest_centroid_distance(self, x_scaled: np.ndarray) -> float:
        # Distance to assigned centroid (useful for confidence)
        centroid = self.model.cluster_centers_[self.predict_cluster(x_scaled)]
        return float(np.linalg.norm(x_scaled - centroid))
