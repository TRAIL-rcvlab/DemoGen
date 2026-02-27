#!/bin/bash
# 评估片段加权实验中训练好的策略，在仿真中验证平均成功率。
#
# 本脚本使用 diffusion_policies/eval.py 对已训练的策略进行推理（inference），
# 在仿真环境中运行多个 episode 并报告平均成功率。
#
# 前置条件:
#   1. 已使用 run_experiments.sh 完成训练
#   2. 训练输出目录中存在 checkpoints/*.ckpt（latest.ckpt 可选）
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
#                          例如: ../diffusion_policies/data/outputs/2026.02.26
#                          或在容器中: /workspace/DemoGen/diffusion_policies/data/outputs/2026.02.26
#   env_runner             (可选) 环境类型: "robosuite" 或 "metaworld"，默认 "robosuite"
#   eval_episodes          (可选) 每个实验评估的 episode 数量，默认 20
#   exp_name               (可选) 单个实验名称，例如 "balanced"
#
# 示例:
#   # 评估目录下所有含 checkpoint 的子目录（使用 robosuite 环境，20 episodes）
#   bash eval_experiments.sh ../diffusion_policies/data/outputs/2026.02.26
#
#   # 评估目录下所有含 checkpoint 的子目录（使用 metaworld 环境，50 episodes）
#   bash eval_experiments.sh ../diffusion_policies/data/outputs/2026.02.26 metaworld 50
#
#   # 评估单个实验（直接指定 checkpoint 目录）
#   bash eval_experiments.sh ../diffusion_policies/data/outputs/2026.02.26/17.09.36_train_xxx robosuite 20 balanced

set -e

# ---- 参数解析 ----
TRAIN_OUTPUT_BASE_RAW="${1:?请提供训练输出目录，例如: ../diffusion_policies/data/outputs/2026.02.26}"
ENV_RUNNER="${2:-robosuite}"
EVAL_EPISODES="${3:-20}"
SINGLE_EXP="${4:-}"

# ---- 路径设置 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CALLER_PWD="$(pwd)"

# ---- 解析训练输出目录（支持绝对路径/当前目录相对路径/仓库根目录相对路径）----
resolve_train_output_dir() {
    local raw_path="$1"

    if [ -z "${raw_path}" ]; then
        return 1
    fi

    if [ -d "${raw_path}" ]; then
        readlink -f "${raw_path}"
        return 0
    fi

    if [ -d "${CALLER_PWD}/${raw_path}" ]; then
        readlink -f "${CALLER_PWD}/${raw_path}"
        return 0
    fi

    if [ -d "${REPO_ROOT}/${raw_path}" ]; then
        readlink -f "${REPO_ROOT}/${raw_path}"
        return 0
    fi

    if [ -d "${REPO_ROOT}/diffusion_policies/${raw_path}" ]; then
        readlink -f "${REPO_ROOT}/diffusion_policies/${raw_path}"
        return 0
    fi

    return 1
}

if ! TRAIN_OUTPUT_BASE="$(resolve_train_output_dir "${TRAIN_OUTPUT_BASE_RAW}")"; then
    echo "错误: 目录 '${TRAIN_OUTPUT_BASE_RAW}' 不存在"
    echo "提示: 可尝试以下路径形式之一"
    echo "  1) 绝对路径: ${REPO_ROOT}/diffusion_policies/data/outputs/<date>"
    echo "  2) 相对仓库根: diffusion_policies/data/outputs/<date>"
    echo "  3) 从当前目录: ../diffusion_policies/data/outputs/<date>"
    exit 1
fi

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
echo "  训练输出目录(输入): ${TRAIN_OUTPUT_BASE_RAW}"
echo "  训练输出目录(解析): ${TRAIN_OUTPUT_BASE}"
echo "  环境类型:     ${ENV_RUNNER}"
echo "  评估 Episodes: ${EVAL_EPISODES}"
echo ""

# ---- 解析 checkpoint 文件 ----
resolve_ckpt_file() {
    local ckpt_dir="$1"

    # 优先使用 latest.ckpt（若训练流程创建了该文件）
    if [ -f "${ckpt_dir}/latest.ckpt" ]; then
        echo "${ckpt_dir}/latest.ckpt"
        return 0
    fi

    # 回退：选择该目录下最新编号的 *.ckpt
    local latest_numbered
    latest_numbered=$(find "${ckpt_dir}" -maxdepth 1 -type f -name '*.ckpt' ! -name 'latest.ckpt' -printf '%f\n' \
        | sort -V \
        | tail -n 1)

    if [ -n "${latest_numbered}" ]; then
        echo "${ckpt_dir}/${latest_numbered}"
        return 0
    fi

    return 1
}

# ---- 评估单个 checkpoint 目录的函数 ----
eval_checkpoint() {
    local ckpt_dir="$1"
    local exp_name="$2"
    local ckpt_file="$3"

    echo "  Checkpoint 目录: ${ckpt_dir}"
    echo "  Checkpoint 文件: ${ckpt_file}"

    export HYDRA_FULL_ERROR=1
    python3 eval.py \
        --config-path="${CONFIG_PATH}" \
        --config-name=segment_weight_exp \
        hydra.run.dir="${ckpt_dir}" \
        training.resume="${ckpt_file}" \
        +task.env_runner._target_="${ENV_RUNNER_TARGET}" \
        +task.env_runner.eval_episodes="${EVAL_EPISODES}" \
        +task.env_runner.n_obs_steps='${n_obs_steps}' \
        +task.env_runner.n_action_steps='${n_action_steps}' \
        +task.env_runner.shape_meta='${shape_meta}' \
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

    # 如果给定目录本身包含 checkpoint，自动解析 checkpoint 文件
    if CKPT_FILE="$(resolve_ckpt_file "${TRAIN_OUTPUT_BASE}/checkpoints")"; then
        eval_checkpoint "${TRAIN_OUTPUT_BASE}" "${SINGLE_EXP}" "${CKPT_FILE}"
    else
        echo "  错误: 未在 '${TRAIN_OUTPUT_BASE}/checkpoints' 下找到可用 checkpoint（latest.ckpt 或 *.ckpt）"
        exit 1
    fi

    echo "============================================"
    echo "  评估已完成！"
    echo "============================================"
    exit 0
fi

# ---- 批量模式：遍历基础目录下所有包含 checkpoint 的子目录 ----
FOUND_ANY=false

for dir in "${TRAIN_OUTPUT_BASE}"/*/; do
    [ -d "${dir}" ] || continue
    CKPT_FILE="$(resolve_ckpt_file "${dir}/checkpoints")" || continue

    FOUND_ANY=true
    dir_name="$(basename "${dir}")"

    echo "--------------------------------------------"
    echo "  评估: ${dir_name}"
    echo "--------------------------------------------"

    eval_checkpoint "${dir}" "${dir_name}" "${CKPT_FILE}"
done

if [ "${FOUND_ANY}" = false ]; then
    echo "  警告: 在 '${TRAIN_OUTPUT_BASE}' 下未找到任何包含可用 checkpoint（latest.ckpt 或 *.ckpt）的子目录"
    echo "  请确认训练已完成，或直接指定 checkpoint 目录："
    echo "    bash eval_experiments.sh <checkpoint_dir> ${ENV_RUNNER} ${EVAL_EPISODES} <exp_name>"
    exit 1
fi

echo "============================================"
echo "  所有评估已完成！"
echo "============================================"
