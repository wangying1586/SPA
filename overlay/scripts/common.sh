#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
DATA_FOLDER="${DATA_FOLDER:-./data}"
SAVE_ROOT="${SAVE_ROOT:-./experiments}"
GPU="${GPU:-0}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRINT_FREQ="${PRINT_FREQ:-20}"

export CUDA_VISIBLE_DEVICES="$GPU"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

COMMON_DATA_ARGS=(
  --dataset icbhi
  --data_folder "$DATA_FOLDER"
  --class_split lungsound
  --sample_rate 16000
  --desired_length 5
  --n_mels 128
  --pad_types repeat
  --raw_augment 0
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --model beats
  --nospec
)

COMMON_TRAIN_ARGS=(
  --epochs "$EPOCHS"
  --optimizer adam
  --learning_rate 5e-5
  --weight_decay 1e-6
  --cosine
  --warm
  --warm_epochs 10
  --print_freq "$PRINT_FREQ"
  --save_freq 100
  --ma_update
  --ma_beta 0.5
  --from_sl_official
  --audioset_pretrained
)

COMMON_PAFA_ARGS=(
  --method pafa
  --w_ce 1.0
  --w_pafa 1.0
  --lambda_pcsl 50
  --lambda_gpal 0.0005
  --norm_type ln
  --output_dim 768
)

COMMON_SPA_ARGS=(
  --w_pafa 1.0
  --lambda_pcsl 50
  --lambda_gpal 0.0005
  --norm_type ln
  --output_dim 768
  --spa_momentum 0.99
  --spa_temperature 2.0
  --spa_lambda 0.5
  --spa_gamma 1.0
  --spa_fusion_weight 0.5
  --spa_scale_init 16.0
  --spa_label_smoothing 0.1
  --spa_kd_direction geo_to_sphere
  --spa_update_after_all_classes
  --w_spa 1.0
  --ece_bins 15
)
