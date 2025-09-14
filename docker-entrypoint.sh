#!/bin/bash
set -e

echo "Starting AutoML Assistant initialization..."

# Initialize Chroma vector store if it doesn't exist
if [ ! -d "/app/chroma_store/rules" ]; then
    echo "Initializing vector store..."
    python ingest_rules.py
else
    echo "Vector store already exists, skipping initialization."
fi

# Initialize MLflow directories and database if they don't exist
if [ ! -d "/app/mlflow-artifacts" ]; then
    echo "Creating MLflow artifacts directory..."
    mkdir -p /app/mlflow-artifacts
fi

# Initialize MLflow tracking database if it doesn't exist
if [ ! -f "/app/mlflow.db" ]; then
    echo "Initializing MLflow tracking database..."
    python -c "
try:
    from mlflow_config import initialize_mlflow
    initialize_mlflow()
    print('MLflow initialized successfully')
except Exception as e:
    print(f'MLflow initialization failed: {e}')
    print('Application will continue without MLflow tracking')
"
else
    echo "MLflow database already exists."
fi

echo "Initialization complete. Starting application..."

# Start the application
exec "$@"