#!/usr/bin/env bash
# Sets up a virtual environment and installs dependencies.
#
# Usage:
#   bash setup.sh            # inference-only deps
#   bash setup.sh --train    # also install training deps (pandas, sklearn, jupyter, ...)

set -e

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

echo "Creating virtual environment in $VENV_DIR ..."
$PYTHON_BIN -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing inference requirements..."
pip install -r requirements.txt

if [[ "$1" == "--train" ]]; then
    echo "Installing training requirements..."
    pip install -r requirements-train.txt
fi

echo ""
echo "Setup complete. Activate the environment with:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "Then run real-time detection with:"
echo "    python realtime_detection.py"
