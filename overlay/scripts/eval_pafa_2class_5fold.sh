#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

for FOLD in 0 1 2 3 4; do
  rm -f ./data/training.pt ./data/test.pt
  RUN_DIR="$SAVE_ROOT/pafa/icbhi_beats_pafa_2class_fold${FOLD}_seed${SEED}"
  "$PYTHON" eval_pafa_checkpoint.py \
    --checkpoint "$RUN_DIR/best.pth" \
    --output "$RUN_DIR/metrics_eval.json" \
    --data_folder "$DATA_FOLDER" \
    --n_cls 2 \
    --test_fold "$FOLD" \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS"
done
