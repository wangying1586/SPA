#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

for FOLD in 0 1 2 3 4; do
  "$PYTHON" main_spa.py \
    "${COMMON_DATA_ARGS[@]}" \
    "${COMMON_TRAIN_ARGS[@]}" \
    "${COMMON_SPA_ARGS[@]}" \
    --n_cls 2 \
    --test_fold "$FOLD" \
    --seed "$SEED" \
    --save_dir "$SAVE_ROOT/spa"
done
