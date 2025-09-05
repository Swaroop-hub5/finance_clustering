
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load
from pathlib import Path
import os

MODELS_DIR = Path(os.getenv("MODELS_DIR", Path(__file__).resolve().parents[1] / "models"))

def get_scaler() -> StandardScaler:
    scaler_path = MODELS_DIR / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run scripts/train.py first.")
    return load(scaler_path)

def transform_features(payload: dict) -> np.ndarray:
    # Expect keys: age, income, spending_score
    x = np.array([[
        payload["age"],
        payload["income"],
        payload["spending_score"]
    ]], dtype=float)
    scaler = get_scaler()
    return scaler.transform(x)
