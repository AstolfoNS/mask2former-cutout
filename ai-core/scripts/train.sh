#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment (if not already activated)
if [ -z "${VIRTUAL_ENV:-}" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

cd "$PROJECT_DIR"

case "${OMP_NUM_THREADS:-}" in
    ""|0|*[!0-9]*) export OMP_NUM_THREADS=4 ;;
esac
case "${MKL_NUM_THREADS:-}" in
    ""|0|*[!0-9]*) export MKL_NUM_THREADS=4 ;;
esac

# Launch training. Arguments are forwarded to src.train, for example:
#   ./scripts/train.sh --batch_size 8 --epochs 40 --report_to wandb

python -u -m src.train \
    "$@"
