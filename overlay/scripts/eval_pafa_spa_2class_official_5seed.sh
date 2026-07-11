#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

for RUN_SEED in 1 2 3 4 5; do
  RUN_DIR="$SAVE_ROOT/spa/icbhi_beats_pafa_spa_2class_official_seed${RUN_SEED}"
  "$PYTHON" main_spa.py \
    "${COMMON_DATA_ARGS[@]}" \
    "${COMMON_SPA_ARGS[@]}" \
    --n_cls 2 \
    --test_fold official \
    --seed "$RUN_SEED" \
    --save_dir "$SAVE_ROOT/spa" \
    --eval \
    --checkpoint "$RUN_DIR/best.pth"
done
