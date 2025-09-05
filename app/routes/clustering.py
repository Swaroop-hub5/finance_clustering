
from fastapi import APIRouter
from pydantic import BaseModel, Field, conint, confloat
from app.services.clustering_service import ClusterService
from app.utils.preprocessing import transform_features

router = APIRouter(prefix="/api", tags=["clustering"])
service = ClusterService()

class CustomerFeatures(BaseModel):
    age: conint(ge=18, le=100) = Field(..., description="Customer age in years")
    income: confloat(ge=0) = Field(..., description="Annual income in USD")
    spending_score: confloat(ge=0, le=100) = Field(..., description="Spending score (0-100)")

@router.post("/predict_cluster")
def predict_cluster(features: CustomerFeatures):
    x_scaled = transform_features(features.model_dump())
    cluster = service.predict_cluster(x_scaled)
    distance = service.nearest_centroid_distance(x_scaled)
    return {"cluster": cluster, "distance_to_centroid": distance}
