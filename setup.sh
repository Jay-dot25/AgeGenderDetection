#!/usr/bin/env bash
# Sets up a virtual environment and installs dependencies.  (macOS / Linux --
# on Windows skip this file and use the PowerShell commands in the README;
# this is a bash script and there is no .bat/.ps1 equivalent.)
#
# Usage:
#   bash setup.sh            # inference-only deps
#   bash setup.sh --train     # also install training deps (pandas, sklearn, jupyter, ...)

# Re-exec under bash if invoked as `sh setup.sh`: the `[[ ]]` test below and
# `source` are bashisms and would abort with a confusing "not found" error.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "error: '$PYTHON_BIN' not found. Install Python 3.9-3.12, or set PYTHON_BIN=/path/to/python." >&2
    exit 1
fi

# tensorflow>=2.16 (what requirements.txt now pins) has no wheel for 3.13+.
if ! "$PYTHON_BIN" -c 'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)'; then
    echo "error: need Python 3.9-3.12 for the pinned TensorFlow; found:" >&2
    "$PYTHON_BIN" --version >&2
    exit 1
fi

echo "Creating virtual environment in $VENV_DIR ..."
if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    echo "error: could not create the venv." >&2
    echo "       On Debian/Ubuntu install the venv module first: sudo apt-get install -y python3-venv" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing inference requirements..."
# All three packages resolve together, which is what keeps numpy on 1.x
# (tensorflow 2.16 requires numpy<2; a separate install can leave numpy 2 in place).
python -m pip install -r requirements.txt

for arg in "$@"; do
    if [[ "$arg" == "--train" ]]; then
        echo "Installing training requirements..."
        python -m pip install -r requirements-train.txt
    fi
done

# Import everything up front, so a headless box missing libGL (or a numpy 2.x
# clash) is reported here instead of mid-run the first time you use the model.
if ! python -c "import cv2, numpy, tensorflow as tf; print(f'  tensorflow {tf.__version__} | numpy {numpy.__version__} | opencv {cv2.__version__}'); print('  imports OK')"; then
    echo "warning: deps installed but importing them failed -- see above." >&2
    echo "         On a headless Linux box, 'libGL.so.1' errors mean: pip install opencv-python-headless" >&2
    exit 1
fi

echo ""
echo "Setup complete. Activate the environment with:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "Then run real-time detection with:"
echo "    python realtime_detection.py"
echo "(needs a webcam and a display; otherwise use scripts/predict_image.py --save)"
