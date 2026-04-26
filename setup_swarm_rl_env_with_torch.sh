#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${1:-swarm-rl-obstacles}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11.11}"
TORCH_VERSION="${TORCH_VERSION:-2.5.0}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found in PATH" >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

conda create -y -n "$ENV_NAME" --override-channels -c conda-forge "python=${PYTHON_VERSION}"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel

# This repo pins torch==2.5.0 in setup.py. Install the CUDA wheel explicitly
# before installing the editable package to avoid pip choosing a CPU build.
python -m pip install \
    "torch==${TORCH_VERSION}" \
    --index-url "${PYTORCH_INDEX_URL}"

python -m pip install \
    numpy==1.26.4 \
    matplotlib==3.9.2 \
    numba==0.60.0 \
    pyglet==1.5.23 \
    gym==0.26.2 \
    gymnasium==0.28.1 \
    transforms3d==0.4.2 \
    noise==1.2.2 \
    tqdm==4.66.5 \
    Cython==3.0.11 \
    scipy==1.14.1 \
    sample-factory==2.1.1 \
    plotly==5.24.1 \
    attrdict==2.0.1 \
    pandas==2.2.3 \
    bezier==2023.7.28 \
    typeguard==4.3.0 \
    osqp==0.6.7.post3 \
    seaborn==0.13.2

python -m pip install -e "$REPO_ROOT" --no-deps

cat <<EOF
Environment '$ENV_NAME' is ready.

Recommended validation commands:
  conda activate $ENV_NAME
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
  python -m swarm_rl.train --env=quadrotor_multi --help
  ./run_tests.sh

Notes:
  - This script installs torch from: $PYTORCH_INDEX_URL
  - Render-related tests require an available X11 display.
  - Training should target an idle GPU because all 4 RTX 4090 cards are
    currently occupied by other Python processes.
EOF
