#!/bin/bash

# Quick demo script for the LLM Inference Server
# This runs the server in demo mode (no actual model required)

set -e

echo "========================================="
echo "LLM Inference Server - Demo Mode"
echo "========================================="
echo ""
echo "Building the server..."
cargo build --release

echo ""
echo "Starting server on http://localhost:8080"
echo ""
echo "Available endpoints:"
echo "  - POST http://localhost:8080/v1/completions"
echo "  - POST http://localhost:8080/v1/chat/completions"
echo "  - GET  http://localhost:8080/v1/models"
echo "  - GET  http://localhost:8080/health"
echo "  - GET  http://localhost:8080/metrics"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run in demo mode
./target/release/llm-server \
  --model-path dummy \
  --model-arch llama \
  --port 8080 \
  --log-level info \
  --workers 2 \
  --max-batch-size 4
