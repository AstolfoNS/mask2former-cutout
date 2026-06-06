#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

case "${OMP_NUM_THREADS:-}" in
    ""|0|*[!0-9]*) export OMP_NUM_THREADS=4 ;;
esac
case "${MKL_NUM_THREADS:-}" in
    ""|0|*[!0-9]*) export MKL_NUM_THREADS=4 ;;
esac

# Launch training. Arguments are forwarded to src.train, for example:
#   ./scripts/train.sh --batch_size 8 --epochs 40 --report_to wandb

if [ -z "${VIRTUAL_ENV:-}" ] && command -v uv >/dev/null 2>&1; then
    exec uv run --project "$PROJECT_DIR" python -u -m src.train "$@"
fi

if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
        source "$PROJECT_DIR/.venv/bin/activate"
    elif [ -f "$PROJECT_DIR/../.venv/bin/activate" ]; then
        source "$PROJECT_DIR/../.venv/bin/activate"
    else
        echo "No virtual environment found. Run: uv sync --project $PROJECT_DIR" >&2
        exit 1
    fi
fi

python -u -m src.train \
    "$@"
