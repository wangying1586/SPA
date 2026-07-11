#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

for RUN_SEED in 1 2 3 4 5; do
  rm -f ./data/training.pt ./data/test.pt
  TAG="2class_official_seed${RUN_SEED}"
  "$PYTHON" main.py \
    "${COMMON_DATA_ARGS[@]}" \
    "${COMMON_TRAIN_ARGS[@]}" \
    "${COMMON_PAFA_ARGS[@]}" \
    --n_cls 2 \
    --test_fold official \
    --seed "$RUN_SEED" \
    --tag "$TAG" \
    --save_dir "$SAVE_ROOT/pafa"
done
