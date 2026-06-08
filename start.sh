#!/bin/bash
echo "Starting FastAPI application on port ${PORT:-10000}..."

# Print environment variables for debugging (remove this later)
echo "Checking environment variables..."
echo "DATABASE_URL is set: ${DATABASE_URL:+YES}"
echo "PORT is set to: ${PORT:-10000}"

# Try to run uvicorn and print any errors
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --log-level debug