#!/bin/bash
# 运行片段加权实验，使用不同的权重配置。
#
# 本脚本使用不同的 motion（运动）和 skill（技能/规划）片段采样权重训练策略，
# 以测试不同数据片段对模型性能的贡献差异。
#
# 前置条件:
#   1. 使用 DemoGen 生成合成数据：
#      cd demo_generation && python gen_demo.py --config-name=flower
#   2. 更新 config/task/segment_weight_exp.yaml 中的 zarr_path
#
# 使用方法:
#   bash run_experiments.sh
#
# 本脚本运行 5 种配置：
#   - balanced:   motion=1.0, skill=1.0（基准）
#   - motion_2x:  motion=2.0, skill=1.0
#   - motion_3x:  motion=3.0, skill=1.0
#   - skill_2x:   motion=1.0, skill=2.0
#   - skill_3x:   motion=1.0, skill=3.0

set -e

# 进入 diffusion_policies 目录（训练入口）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}/diffusion_policies"

CONFIG_PATH="../experiments/segment_weighting/config"

echo "============================================"
echo "  片段加权实验"
echo "============================================"
echo ""

# 定义实验配置：名称, motion权重, skill权重
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
    echo "  运行: ${exp_name}"
    echo "  Motion 权重: ${motion_w}, Skill 权重: ${skill_w}"
    echo "--------------------------------------------"

    python train.py \
        --config-path="${CONFIG_PATH}" \
        --config-name=segment_weight_exp \
        task.segment_weights.motion="${motion_w}" \
        task.segment_weights.skill="${skill_w}" \
        exp_name="${exp_name}" \
        logging.name="${exp_name}" \
        logging.tags="[${exp_name},segment_weighting]"

    echo "  完成: ${exp_name}"
    echo ""
done

echo "============================================"
echo "  所有实验已完成！"
echo "============================================"
