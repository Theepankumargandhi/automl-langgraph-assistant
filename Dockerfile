FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make entrypoint executable
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create necessary directories for application, MLflow, and Chroma
RUN mkdir -p /app/data /app/outputs /app/chroma_store /app/mlflow-artifacts

ENV PYTHONPATH=/app
ENV ALLOW_IO=1
ENV ALLOW_TUNING=0
ENV ALLOWED_DATA_DIR=/app/data

# MLflow environment variables
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/app/mlflow-artifacts
ENV MLFLOW_EXPERIMENT_NAME=automl-assistant-experiments
ENV MLFLOW_REGISTRY_URI=sqlite:///mlflow.db
ENV MLFLOW_AUTO_LOG=false
ENV MLFLOW_LOG_ARTIFACTS=true
ENV MLFLOW_LOG_MODELS=true

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]