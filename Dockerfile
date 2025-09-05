
FROM python:3.10-slim

WORKDIR /app

# System deps (optional, kept minimal for slim images)
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only runtime code (exclude notebooks/tests/data via .dockerignore)
COPY app ./app

# Expose FastAPI port
EXPOSE 8000

# Start API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
