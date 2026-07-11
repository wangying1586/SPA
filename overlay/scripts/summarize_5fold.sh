#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
"$PYTHON" tools/summarize_results.py \
  --experiments "$SAVE_ROOT" \
  --output_dir "$SAVE_ROOT/summary"
