#!/bin/bash
# Run e2e tests for hpc_bench

set -e

cd "$(dirname "$0")/.."

echo "Running hpc_bench e2e tests..."
echo "================================"

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "pytest not found, trying python -m pytest..."
    PYTHON="${PYTHON:-python}"
else
    PYTHON="python -m"
fi

# Run tests
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Run e2e tests (skip CUDA tests if not available)
$PYTHON pytest tests/e2e -v -m "not cuda" "$@"

echo ""
echo "================================"
echo "Tests complete!"
echo ""
echo "To run with CUDA tests:"
echo "  export PYTHONPATH=src:$PYTHONPATH && python -m pytest tests/e2e -v"
