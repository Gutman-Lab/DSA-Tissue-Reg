#!/bin/bash
# Run the DSA Tissue Registration Docker stack from repo root.
# Delegates to viteReg/run-docker.sh (must be run from viteReg).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/viteReg" || { echo "Error: viteReg/ not found"; exit 1; }
exec ./run-docker.sh "$@"
