#!/usr/bin/env bash
# ============================================================
# DemoGen Installation Script
# Sets up the complete environment for training and inference
# ============================================================
#
# Usage:
#   bash scripts/install.sh          # Full install (conda + all deps)
#   bash scripts/install.sh --pip    # pip-only install (skip conda)
#
# Prerequisites:
#   - NVIDIA GPU with CUDA 11.8+
#   - conda (Miniconda or Anaconda), unless using --pip

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PIP_ONLY=false
ENV_NAME="demogen"

for arg in "$@"; do
    case $arg in
        --pip) PIP_ONLY=true ;;
    esac
done

echo "============================================"
echo "  DemoGen Environment Setup"
echo "============================================"
echo ""

# ---- Step 0: Conda Environment ----
if [ "$PIP_ONLY" = false ]; then
    echo "[1/4] Creating conda environment '${ENV_NAME}' (Python 3.8)..."
    conda create -n "${ENV_NAME}" python=3.8 -y
    echo "      Activating environment..."

    # Activate in subshell-compatible way
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
else
    echo "[1/4] Skipping conda setup (--pip mode)"
fi

# ---- Step 1: Install PyTorch with CUDA ----
echo "[2/4] Installing PyTorch 2.0.1 with CUDA support..."
pip install torch==2.0.1 torchvision torchaudio

# ---- Step 2: Install pip dependencies ----
echo "[3/4] Installing pip dependencies..."
pip install -r "${REPO_ROOT}/requirements.txt"

# ---- Step 3: Install project packages ----
echo "[4/4] Installing DemoGen packages (editable mode)..."
cd "${REPO_ROOT}/demo_generation" && pip install -e . && cd "${REPO_ROOT}"
cd "${REPO_ROOT}/diffusion_policies" && pip install -e . && cd "${REPO_ROOT}"
cd "${REPO_ROOT}/pcd_visualizer" && pip install -e . && cd "${REPO_ROOT}"

echo ""
echo "============================================"
echo "  Installation complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Activate the environment:  conda activate ${ENV_NAME}"
echo "  2. Generate demos:            cd demo_generation && bash run_gen_demo.sh"
echo "  3. Train policies:            cd diffusion_policies && bash train.sh <demo> <algo> <task> <seed>"
echo "  4. Run experiments:           cd experiments/segment_weighting && bash run_experiments.sh"
echo ""
