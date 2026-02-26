#!/bin/bash
# 评估片段加权实验中训练好的策略，在仿真中验证平均成功率。
#
# 本脚本使用 diffusion_policies/eval.py 对已训练的策略进行推理（inference），
# 在仿真环境中运行多个 episode 并报告平均成功率。
#
# 前置条件:
#   1. 已使用 run_experiments.sh 完成训练
#   2. 训练输出目录中存在 checkpoints/latest.ckpt
#
# 使用方法:
#   # 方式一：批量评估（自动搜索 checkpoint）
#   bash eval_experiments.sh <train_output_base_dir> [env_runner] [eval_episodes]
#
#   # 方式二：评估单个实验（指定 checkpoint 目录）
#   bash eval_experiments.sh <checkpoint_dir> [env_runner] [eval_episodes] <exp_name>
#
# 参数:
#   train_output_base_dir  训练输出的基础目录，其中包含各实验的子目录
#                          例如: data/outputs/2024.01.01
#   env_runner             (可选) 环境类型: "robosuite" 或 "metaworld"，默认 "robosuite"
#   eval_episodes          (可选) 每个实验评估的 episode 数量，默认 20
#   exp_name               (可选) 单个实验名称，例如 "balanced"
#
# 示例:
#   # 评估目录下所有含 checkpoint 的子目录（使用 robosuite 环境，20 episodes）
#   bash eval_experiments.sh data/outputs/2024.01.01
#
#   # 评估目录下所有含 checkpoint 的子目录（使用 metaworld 环境，50 episodes）
#   bash eval_experiments.sh data/outputs/2024.01.01 metaworld 50
#
#   # 评估单个实验（直接指定 checkpoint 目录）
#   bash eval_experiments.sh data/outputs/2024.01.01/12.00.00_train_xxx robosuite 20 balanced

set -e

# ---- 参数解析 ----
TRAIN_OUTPUT_BASE="${1:?请提供训练输出目录，例如: data/outputs/2024.01.01}"
ENV_RUNNER="${2:-robosuite}"
EVAL_EPISODES="${3:-20}"
SINGLE_EXP="${4:-}"

# ---- 路径设置 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}/diffusion_policies"

CONFIG_PATH="../experiments/segment_weighting/config"

# ---- 选择 env_runner ----
case "${ENV_RUNNER}" in
    robosuite)
        ENV_RUNNER_TARGET="diffusion_policies.env_runner.robosuite_runner.RobosuiteRunner"
        ;;
    metaworld)
        ENV_RUNNER_TARGET="diffusion_policies.env_runner.metaworld_runner.MetaworldRunner"
        ;;
    *)
        echo "错误: 不支持的 env_runner 类型 '${ENV_RUNNER}'，请使用 'robosuite' 或 'metaworld'"
        exit 1
        ;;
esac

echo "============================================"
echo "  片段加权实验 - 策略评估（推理）"
echo "============================================"
echo "  训练输出目录: ${TRAIN_OUTPUT_BASE}"
echo "  环境类型:     ${ENV_RUNNER}"
echo "  评估 Episodes: ${EVAL_EPISODES}"
echo ""

# ---- 评估单个 checkpoint 目录的函数 ----
eval_checkpoint() {
    local ckpt_dir="$1"
    local exp_name="$2"

    echo "  Checkpoint 目录: ${ckpt_dir}"

    export HYDRA_FULL_ERROR=1
    python eval.py \
        --config-path="${CONFIG_PATH}" \
        --config-name=segment_weight_exp \
        hydra.run.dir="${ckpt_dir}" \
        task.env_runner._target_="${ENV_RUNNER_TARGET}" \
        task.env_runner.eval_episodes="${EVAL_EPISODES}" \
        task.env_runner.n_obs_steps='${n_obs_steps}' \
        task.env_runner.n_action_steps='${n_action_steps}' \
        task.env_runner.shape_meta='${shape_meta}' \
        exp_name="${exp_name}" \
    && echo "  完成: ${exp_name}" \
    || echo "  失败: ${exp_name}"

    echo ""
}

# ---- 单个实验模式：直接使用给定目录 ----
if [ -n "${SINGLE_EXP}" ]; then
    echo "--------------------------------------------"
    echo "  评估: ${SINGLE_EXP}"
    echo "--------------------------------------------"

    # 如果给定目录本身包含 checkpoint，直接使用
    if [ -f "${TRAIN_OUTPUT_BASE}/checkpoints/latest.ckpt" ]; then
        eval_checkpoint "${TRAIN_OUTPUT_BASE}" "${SINGLE_EXP}"
    else
        echo "  错误: 未在 '${TRAIN_OUTPUT_BASE}/checkpoints/latest.ckpt' 找到 checkpoint"
        exit 1
    fi

    echo "============================================"
    echo "  评估已完成！"
    echo "============================================"
    exit 0
fi

# ---- 批量模式：遍历基础目录下所有包含 checkpoint 的子目录 ----
if [ ! -d "${TRAIN_OUTPUT_BASE}" ]; then
    echo "错误: 目录 '${TRAIN_OUTPUT_BASE}' 不存在"
    exit 1
fi

FOUND_ANY=false

for dir in "${TRAIN_OUTPUT_BASE}"/*/; do
    [ -d "${dir}" ] || continue
    [ -f "${dir}/checkpoints/latest.ckpt" ] || continue

    FOUND_ANY=true
    dir_name="$(basename "${dir}")"

    echo "--------------------------------------------"
    echo "  评估: ${dir_name}"
    echo "--------------------------------------------"

    eval_checkpoint "${dir}" "${dir_name}"
done

if [ "${FOUND_ANY}" = false ]; then
    echo "  警告: 在 '${TRAIN_OUTPUT_BASE}' 下未找到任何包含 checkpoints/latest.ckpt 的子目录"
    echo "  请确认训练已完成，或直接指定 checkpoint 目录："
    echo "    bash eval_experiments.sh <checkpoint_dir> ${ENV_RUNNER} ${EVAL_EPISODES} <exp_name>"
    exit 1
fi

echo "============================================"
echo "  所有评估已完成！"
echo "============================================"
