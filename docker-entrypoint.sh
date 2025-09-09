#!/bin/bash
set -e

# Initialize Chroma vector store if it doesn't exist
if [ ! -d "/app/chroma_store/rules" ]; then
    echo "Initializing vector store..."
    python ingest_rules.py
fi

# Start the application
exec "$@"