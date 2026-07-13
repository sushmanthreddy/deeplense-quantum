#!/usr/bin/env bash
set -euo pipefail

: "${DEVELOPMENT_ROOT:?Set DEVELOPMENT_ROOT to the Model_I development directory}"
: "${TEST_ROOT:?Set TEST_ROOT to the Model_I_test directory}"
: "${CACHE_ROOT:?Set CACHE_ROOT to the d4-orqb cache directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new output directory}"
: "${BACKBONE_CHECKPOINT:?Set BACKBONE_CHECKPOINT to the pretrained best.pt}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  exit 1
fi

python -m d4_orqb.train \
  --development-root "${DEVELOPMENT_ROOT}" \
  --test-root "${TEST_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --image-size 96 \
  --encoder-variant tiny \
  --physics-variant base \
  --core quantum \
  --heads 4 \
  --reuploads 2 \
  --epochs 40 \
  --patience 41 \
  --batch-size 256 \
  --workers 4 \
  --io-workers 8 \
  --encoder-learning-rate 0.0005 \
  --learning-rate 0.003 \
  --core-learning-rate 0.005 \
  --init-backbone-checkpoint "${BACKBONE_CHECKPOINT}" \
  --seed 2 \
  --split-seed 42
