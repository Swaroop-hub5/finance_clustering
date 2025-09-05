
# Clustering Project (End-to-End)

An end-to-end example of deploying a KMeans clustering model via FastAPI, containerized with Docker, and ready for Kubernetes.

## Project Layout
See the folder structure and key components in the repository.

## Quickstart (Local)
1. Create artifacts and dummy data:
   ```bash
   python scripts/train.py
   python scripts/evaluate.py
   ```
2. Run API locally:
   ```bash
   uvicorn app.main:app --reload --port 8000
   # Test:
   curl -X POST http://localhost:8000/api/predict_cluster -H "Content-Type: application/json" -d '{"age":30, "income":40000, "spending_score":60}'
   ```

## Docker
1. Build image:
   ```bash
   docker build -t clustering-api .
   ```
2. (Option A) Bake artifacts into the image (copy models dir) — simple but static.  
   (Option B) Mount models at runtime — more flexible.
   For simplicity, bake artifacts:
   ```dockerfile
   # After running scripts/train.py, you can add this line to Dockerfile before CMD:
   # COPY app/models ./app/models
   ```
   Rebuild the image after training.

3. Run:
   ```bash
   docker run -p 8000:8000 clustering-api
   ```

## Kubernetes (Demo)
- Apply manifests (assuming image is available to the cluster):
  ```bash
  kubectl apply -f k8s/deployment.yaml
  kubectl apply -f k8s/deployment.yaml
  ```
- Expose via Ingress or change Service to `LoadBalancer` in cloud environments.

## Tests
```bash
pip install -r requirements.txt
pytest -q
```

## Notes
- To regenerate training artifacts, run `python scripts/train.py`.
- The API expects fields: `age`, `income`, `spending_score`.
