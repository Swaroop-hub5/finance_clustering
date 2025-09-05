
from pathlib import Path
from joblib import load
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "app" / "models"

def test_artifacts_exist():
    assert (MODELS / "scaler.pkl").exists(), "Missing scaler.pkl. Run scripts/train.py."
    assert (MODELS / "kmeans_model.pkl").exists(), "Missing kmeans_model.pkl. Run scripts/train.py."

def test_prediction_shape():
    scaler = load(MODELS / "scaler.pkl")
    kmeans = load(MODELS / "kmeans_model.pkl")
    x = np.array([[30, 40000, 60]], dtype=float)
    x_scaled = scaler.transform(x)
    pred = kmeans.predict(x_scaled)
    assert pred.shape == (1,)
