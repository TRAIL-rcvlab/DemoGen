"""
Utility functions for Metaworld teleoperation.
Provides environment creation and task listing helpers.
"""

import metaworld
import gymnasium as gym
import numpy as np


# All 50 Metaworld v3 task names
METAWORLD_TASKS = [
    "assembly-v3",
    "basketball-v3",
    "bin-picking-v3",
    "box-close-v3",
    "button-press-topdown-v3",
    "button-press-topdown-wall-v3",
    "button-press-v3",
    "button-press-wall-v3",
    "coffee-button-v3",
    "coffee-pull-v3",
    "coffee-push-v3",
    "dial-turn-v3",
    "disassemble-v3",
    "door-close-v3",
    "door-lock-v3",
    "door-open-v3",
    "door-unlock-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "faucet-close-v3",
    "faucet-open-v3",
    "hammer-v3",
    "hand-insert-v3",
    "handle-press-side-v3",
    "handle-press-v3",
    "handle-pull-side-v3",
    "handle-pull-v3",
    "lever-pull-v3",
    "peg-insert-side-v3",
    "peg-unplug-side-v3",
    "pick-out-of-hole-v3",
    "pick-place-v3",
    "pick-place-wall-v3",
    "plate-slide-back-side-v3",
    "plate-slide-back-v3",
    "plate-slide-side-v3",
    "plate-slide-v3",
    "push-back-v3",
    "push-v3",
    "push-wall-v3",
    "reach-v3",
    "reach-wall-v3",
    "shelf-place-v3",
    "soccer-v3",
    "stick-pull-v3",
    "stick-push-v3",
    "sweep-into-v3",
    "sweep-v3",
    "window-close-v3",
    "window-open-v3",
]


# Chinese labels for Metaworld tasks (used by teleop task dropdown).
METAWORLD_TASK_LABELS = {
    "assembly-v3": "装配",
    "basketball-v3": "投篮",
    "bin-picking-v3": "料箱抓取",
    "box-close-v3": "关闭盒盖",
    "button-press-topdown-v3": "俯按按钮",
    "button-press-topdown-wall-v3": "靠墙俯按按钮",
    "button-press-v3": "按按钮",
    "button-press-wall-v3": "靠墙按按钮",
    "coffee-button-v3": "咖啡机按键",
    "coffee-pull-v3": "拉咖啡杯",
    "coffee-push-v3": "推咖啡杯",
    "dial-turn-v3": "旋钮旋转",
    "disassemble-v3": "拆卸",
    "door-close-v3": "关门",
    "door-lock-v3": "锁门",
    "door-open-v3": "开门",
    "door-unlock-v3": "开锁",
    "drawer-close-v3": "关抽屉",
    "drawer-open-v3": "开抽屉",
    "faucet-close-v3": "关水龙头",
    "faucet-open-v3": "开水龙头",
    "hammer-v3": "锤击",
    "hand-insert-v3": "手柄插入",
    "handle-press-side-v3": "侧向按把手",
    "handle-press-v3": "按把手",
    "handle-pull-side-v3": "侧向拉把手",
    "handle-pull-v3": "拉把手",
    "lever-pull-v3": "拉杆",
    "peg-insert-side-v3": "侧向插销",
    "peg-unplug-side-v3": "侧向拔销",
    "pick-out-of-hole-v3": "孔中取物",
    "pick-place-v3": "抓取放置",
    "pick-place-wall-v3": "靠墙抓取放置",
    "plate-slide-back-side-v3": "侧向拉回滑板",
    "plate-slide-back-v3": "拉回滑板",
    "plate-slide-side-v3": "侧向推滑板",
    "plate-slide-v3": "推滑板",
    "push-back-v3": "向后推",
    "push-v3": "前推",
    "push-wall-v3": "靠墙推",
    "reach-v3": "到达",
    "reach-wall-v3": "靠墙到达",
    "shelf-place-v3": "置于搁板",
    "soccer-v3": "推球",
    "stick-pull-v3": "用杆拉",
    "stick-push-v3": "用杆推",
    "sweep-into-v3": "扫入目标区",
    "sweep-v3": "清扫",
    "window-close-v3": "关窗",
    "window-open-v3": "开窗",
}


def list_available_tasks():
    """Return list of all available Metaworld task names."""
    return METAWORLD_TASKS.copy()


def get_task_label(task_name):
    """Return Chinese task label, or task name if no label exists."""
    return METAWORLD_TASK_LABELS.get(task_name, task_name)


def create_metaworld_env(task_name="pick-place-v3", render_mode="human", seed=42,
                          max_episode_steps=None):
    """
    Create a Metaworld environment via Gymnasium API.

    Args:
        task_name: Name of the Metaworld task (e.g. 'pick-place-v3')
        render_mode: 'human' for GUI window, 'rgb_array' for offscreen
        seed: Random seed for reproducibility
        max_episode_steps: Override max steps. None = no limit (teleop-friendly).

    Returns:
        gymnasium.Env: The created environment
    """
    if task_name not in METAWORLD_TASKS:
        raise ValueError(
            f"Unknown task '{task_name}'. "
            f"Available tasks: {METAWORLD_TASKS}"
        )

    env = gym.make(
        "Meta-World/MT1",
        env_name=task_name,
        seed=seed,
        render_mode=render_mode,
    )

    # Strip the TimeLimit wrapper (default 500 steps) to prevent truncation
    # during teleoperation. The user controls reset timing manually (R key).
    env = _strip_timelimit(env)

    # Also override the base Metaworld env's internal max_path_length
    # (SawyerEnv checks curr_path_length >= max_path_length for truncation)
    inner = env.unwrapped
    if hasattr(inner, 'max_path_length'):
        inner.max_path_length = max_episode_steps if max_episode_steps else 50000

    # Optionally re-wrap with a higher limit
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    return env


def _strip_timelimit(env):
    """Remove TimeLimit wrapper from a gymnasium environment stack."""
    if isinstance(env, gym.wrappers.TimeLimit):
        return env.env  # unwrap one layer
    # Check if TimeLimit is nested deeper
    if hasattr(env, 'env'):
        env.env = _strip_timelimit(env.env)
    return env


def get_env_info(env):
    """Get useful information about the environment."""
    info = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "action_dim": env.action_space.shape[0],
        "obs_dim": env.observation_space.shape[0],
    }
    return info
