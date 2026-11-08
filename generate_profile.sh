
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <script.py> [args...]" >&2
    exit 1
fi

SCRIPT="$1"
shift

SCRIPT="$(readlink -f "$SCRIPT")"

if [[ ! -f "$SCRIPT" ]]; then
    echo "Error: Python script not found: $SCRIPT" >&2
    exit 1
fi

# Resolve the project root relative to this wrapper.
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$HERE/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Error: Python venv not found: $PYTHON" >&2
    exit 1
fi

NCU="$(command -v ncu)"
PROFILE_DIR="$HERE/profiles"
mkdir -p "$PROFILE_DIR"

NAME="$(basename "$SCRIPT" .py)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="$PROFILE_DIR/${NAME}_${TIMESTAMP}"

sudo "$NCU" \
    --profile-from-start on \
    --kernel-name "regex:${KERNEL_FILTER:-.*}" \
    --launch-count 1 \
    --set full \
    --import-source yes \
    -o "$OUTPUT" \
    /usr/bin/env \
        TRITON_DISABLE_LINE_INFO=0 \
        TRITON_ALWAYS_COMPILE=1 \
        TRITON_DEFAULT_FP_FUSION=1 \
        "$PYTHON" "$SCRIPT" "$@"

echo
echo "Profile: ${OUTPUT}.ncu-rep"
echo "Open with: ncu-ui ${OUTPUT}.ncu-rep"
