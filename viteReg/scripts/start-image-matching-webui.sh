#!/bin/bash
# Quick start script for image-matching-webui service

set -e

echo "Starting image-matching-webui service..."
cd "$(dirname "$0")/.."

# Create data directory if it doesn't exist
mkdir -p image-matching-data

# Start the service
docker-compose up -d image-matching-webui

echo ""
echo "✅ Image Matching WebUI is starting..."
echo ""
echo "Access the web UI at: http://localhost:7860"
echo ""
echo "To view logs: docker-compose logs -f image-matching-webui"
echo "To stop: docker-compose stop image-matching-webui"
echo ""
