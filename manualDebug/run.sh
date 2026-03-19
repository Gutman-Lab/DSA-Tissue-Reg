#!/bin/bash
# Run the manual debug tool with uv

cd "$(dirname "$0")"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv
fi

# Activate the environment
source .venv/bin/activate

# Install/update dependencies if needed
if [ ! -f ".venv/.installed" ] || [ "pyproject.toml" -nt ".venv/.installed" ]; then
    echo "Installing dependencies..."
    uv pip install -e .
    touch .venv/.installed
fi

# Set default port (can be overridden with PORT environment variable)
export PORT=${PORT:-8002}

# Run the app
echo "Starting server on port $PORT..."
python3 app.py
