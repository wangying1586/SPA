#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

for RUN_SEED in 1 2 3 4 5; do
  "$PYTHON" main_spa.py \
    "${COMMON_DATA_ARGS[@]}" \
    "${COMMON_TRAIN_ARGS[@]}" \
    "${COMMON_SPA_ARGS[@]}" \
    --n_cls 4 \
    --test_fold official \
    --seed "$RUN_SEED" \
    --save_dir "$SAVE_ROOT/spa"
done
