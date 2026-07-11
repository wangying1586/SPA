#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-$PACKAGE_ROOT/PAFA-SPA}"

if [[ -e "$TARGET" ]]; then
  echo "Target already exists: $TARGET" >&2
  echo "Use install_overlay.py to install into an existing PAFA clone." >&2
  exit 2
fi

git clone https://github.com/wa976/pafa.git "$TARGET"
python "$PACKAGE_ROOT/install_overlay.py" --target "$TARGET"

echo "PAFA-SPA repository created at: $TARGET"
echo "Continue with the environment and data preparation steps in README.md."
