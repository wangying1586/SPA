#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

for FOLD in 0 1 2 3 4; do
  rm -f ./data/training.pt ./data/test.pt
  TAG="4class_fold${FOLD}_seed${SEED}"
  "$PYTHON" main.py \
    "${COMMON_DATA_ARGS[@]}" \
    "${COMMON_TRAIN_ARGS[@]}" \
    "${COMMON_PAFA_ARGS[@]}" \
    --n_cls 4 \
    --test_fold "$FOLD" \
    --seed "$SEED" \
    --tag "$TAG" \
    --save_dir "$SAVE_ROOT/pafa"
done
