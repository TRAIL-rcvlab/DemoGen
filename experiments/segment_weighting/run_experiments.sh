#!/bin/bash
# Run segment weighting experiments with different weight configurations.
#
# This script trains policies with different sampling weights for motion
# vs skill segments to test their relative contribution to model performance.
#
# Prerequisites:
#   1. Generate synthetic data using DemoGen:
#      cd demo_generation && python gen_demo.py --config-name=flower
#   2. Update zarr_path in config/task/segment_weight_exp.yaml
#
# Usage:
#   bash run_experiments.sh
#
# The script runs 5 configurations:
#   - balanced:   motion=1.0, skill=1.0 (baseline)
#   - motion_2x:  motion=2.0, skill=1.0
#   - motion_3x:  motion=3.0, skill=1.0
#   - skill_2x:   motion=1.0, skill=2.0
#   - skill_3x:   motion=1.0, skill=3.0

set -e

# Navigate to the diffusion_policies directory (training entry point)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}/diffusion_policies"

CONFIG_PATH="../experiments/segment_weighting/config"

echo "============================================"
echo "  Segment Weighting Experiment"
echo "============================================"
echo ""

# Define experiment configurations: name, motion_weight, skill_weight
declare -a EXPERIMENTS=(
    "balanced 1.0 1.0"
    "motion_2x 2.0 1.0"
    "motion_3x 3.0 1.0"
    "skill_2x 1.0 2.0"
    "skill_3x 1.0 3.0"
)

for exp in "${EXPERIMENTS[@]}"; do
    read -r exp_name motion_w skill_w <<< "$exp"

    echo "--------------------------------------------"
    echo "  Running: ${exp_name}"
    echo "  Motion weight: ${motion_w}, Skill weight: ${skill_w}"
    echo "--------------------------------------------"

    python train.py \
        --config-path="${CONFIG_PATH}" \
        --config-name=segment_weight_exp \
        task.segment_weights.motion="${motion_w}" \
        task.segment_weights.skill="${skill_w}" \
        exp_name="${exp_name}" \
        logging.name="${exp_name}" \
        logging.tags="[${exp_name},segment_weighting]"

    echo "  Finished: ${exp_name}"
    echo ""
done

echo "============================================"
echo "  All experiments completed!"
echo "============================================"
