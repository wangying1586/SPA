#!/usr/bin/env bash
set -euo pipefail
TASK="${1:-4}"

case "$TASK" in
  4)
    bash "$(dirname "$0")/train_pafa_4class_5fold.sh"
    bash "$(dirname "$0")/eval_pafa_4class_5fold.sh"
    bash "$(dirname "$0")/train_pafa_spa_4class_5fold.sh"
    bash "$(dirname "$0")/eval_pafa_spa_4class_5fold.sh"
    ;;
  2)
    bash "$(dirname "$0")/train_pafa_2class_5fold.sh"
    bash "$(dirname "$0")/eval_pafa_2class_5fold.sh"
    bash "$(dirname "$0")/train_pafa_spa_2class_5fold.sh"
    bash "$(dirname "$0")/eval_pafa_spa_2class_5fold.sh"
    ;;
  *)
    echo "Usage: bash scripts/run_all_5fold.sh {4|2}" >&2
    exit 2
    ;;
esac

bash "$(dirname "$0")/summarize_5fold.sh"
