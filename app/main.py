
from fastapi import FastAPI
from app.routes.clustering import router as clustering_router

app = FastAPI(title="Clustering API", version="1.0.0")
app.include_router(clustering_router)

@app.get("/health")
def health():
    return {"status": "ok"}
