#!/usr/bin/env bash
#
# Example of running the SPIRAL image directly with Docker.
# Requirements: Docker >= 19.03, NVIDIA Container Toolkit installed,
# NVIDIA driver >= 510.39.01, linux/amd64 architecture.
#
# Usage: adjust the paths below, then run this script.

set -euo pipefail

# ── Local paths to adjust ───────────────────────────────────────────────
EMBEDDINGS_DIR="/path/to/embeddings"
CSV_FILE="${EMBEDDINGS_DIR}/proteins.csv"
OUTPUT_DIR="/path/to/output"

# ── Public Docker image ─────────────────────────────────────────────────
DOCKER_IMAGE="zulied/spiral:v1.0"

mkdir -p "$OUTPUT_DIR"

docker run --rm --gpus all \
    -v "${OUTPUT_DIR}:${OUTPUT_DIR}" \
    -v "${PROTT5_DIR}:${PROTT5_DIR}" \
    "${DOCKER_IMAGE}" \
    prediction \
    --embeddings "${EMBEDDINGS_DIR}" \
    --csv "${CSV_FILE}" \
    --output "${OUTPUT_DIR}" 
