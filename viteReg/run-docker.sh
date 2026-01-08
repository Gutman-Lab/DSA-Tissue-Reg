#!/bin/bash
# Docker stack startup script for DSA Tissue Registration

set -e

echo "DSA Tissue Registration - Docker Stack"
echo "======================================"
echo ""

# Get current user's UID and GID for proper file permissions
# This ensures files created in containers match host user ownership
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

echo "Using user ID: $USER_ID, group ID: $GROUP_ID"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found"
    echo "Creating .env from .env.docker template..."
    if [ -f .env.docker ]; then
        cp .env.docker .env
        echo "Please edit .env with your DSA credentials"
    else
        echo "Error: .env.docker template not found"
        exit 1
    fi
fi

# Parse command line arguments
MODE=${1:-prod}

if [ "$MODE" == "dev" ]; then
    echo "Starting in DEVELOPMENT mode..."
    echo "Frontend: http://localhost:5173"
    echo "Backend: http://localhost:8000"
    echo ""
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
elif [ "$MODE" == "prod" ]; then
    echo "Starting in PRODUCTION mode..."
    echo "Frontend: http://localhost"
    echo "Backend: http://localhost:8000"
    echo ""
    docker-compose up --build
else
    echo "Usage: $0 [dev|prod]"
    echo "  dev  - Development mode with hot reload"
    echo "  prod - Production mode (default)"
    exit 1
fi

