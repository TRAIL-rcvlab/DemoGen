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
#   bash eval_experiments.sh <train_output_base_dir> [env_runner] [eval_episodes]
#
# 参数:
#   train_output_base_dir  训练输出的基础目录，其中包含各实验的子目录
#                          例如: data/outputs/2024.01.01
#   env_runner             (可选) 环境类型: "robosuite" 或 "metaworld"，默认 "robosuite"
#   eval_episodes          (可选) 每个实验评估的 episode 数量，默认 20
#
# 示例:
#   # 评估所有实验（使用 robosuite 环境，20 episodes）
#   bash eval_experiments.sh data/outputs/2024.01.01
#
#   # 评估所有实验（使用 metaworld 环境，50 episodes）
#   bash eval_experiments.sh data/outputs/2024.01.01 metaworld 50
#
#   # 评估单个实验
#   bash eval_experiments.sh data/outputs/2024.01.01 robosuite 20 balanced

set -e

# ---- 参数解析 ----
TRAIN_OUTPUT_BASE="${1:?请提供训练输出的基础目录，例如: data/outputs/2024.01.01}"
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

# ---- 定义实验名称（与 run_experiments.sh 一致）----
declare -a EXPERIMENT_NAMES=(
    "balanced"
    "motion_2x"
    "motion_3x"
    "skill_2x"
    "skill_3x"
)

# 如果指定了单个实验，则只评估该实验
if [ -n "${SINGLE_EXP}" ]; then
    EXPERIMENT_NAMES=("${SINGLE_EXP}")
fi

# ---- 收集评估结果 ----
declare -A RESULTS

for exp_name in "${EXPERIMENT_NAMES[@]}"; do
    echo "--------------------------------------------"
    echo "  评估: ${exp_name}"
    echo "--------------------------------------------"

    # 查找该实验的训练输出目录
    # 训练时 hydra.run.dir 格式: data/outputs/{date}/{time}_{name}_{task_name}
    # 在基础目录中搜索匹配的子目录
    EXP_DIR=""
    if [ -d "${TRAIN_OUTPUT_BASE}" ]; then
        # 搜索包含实验名称的目录（匹配 *_train_diffusion_unet_hybrid_*exp_name* 模式）
        for dir in "${TRAIN_OUTPUT_BASE}"/*/; do
            if [ -d "${dir}" ]; then
                dir_name="$(basename "${dir}")"
                # 训练目录名包含 exp_name 作为 hydra override
                if [ -f "${dir}/checkpoints/latest.ckpt" ]; then
                    # 检查目录名或配置中是否匹配实验名称
                    if echo "${dir_name}" | grep -q "${exp_name}\|train_diffusion_unet_hybrid"; then
                        EXP_DIR="${dir}"
                        break
                    fi
                fi
            fi
        done

        # 如果没找到精确匹配，尝试通过 exp_name 子目录查找
        if [ -z "${EXP_DIR}" ] && [ -d "${TRAIN_OUTPUT_BASE}/${exp_name}" ]; then
            EXP_DIR="${TRAIN_OUTPUT_BASE}/${exp_name}"
        fi
    fi

    if [ -z "${EXP_DIR}" ] || [ ! -f "${EXP_DIR}/checkpoints/latest.ckpt" ]; then
        echo "  警告: 未找到实验 '${exp_name}' 的 checkpoint，跳过"
        echo "  搜索路径: ${TRAIN_OUTPUT_BASE}"
        echo ""
        RESULTS["${exp_name}"]="跳过（未找到 checkpoint）"
        continue
    fi

    echo "  Checkpoint 目录: ${EXP_DIR}"

    export HYDRA_FULL_ERROR=1
    python eval.py \
        --config-path="${CONFIG_PATH}" \
        --config-name=segment_weight_exp \
        hydra.run.dir="${EXP_DIR}" \
        task.env_runner._target_="${ENV_RUNNER_TARGET}" \
        task.env_runner.eval_episodes="${EVAL_EPISODES}" \
        task.env_runner.n_obs_steps='${n_obs_steps}' \
        task.env_runner.n_action_steps='${n_action_steps}' \
        task.env_runner.shape_meta='${shape_meta}' \
        exp_name="${exp_name}" \
    && echo "  完成: ${exp_name}" \
    || echo "  失败: ${exp_name}"

    echo ""
done

echo "============================================"
echo "  所有评估已完成！"
echo "============================================"
