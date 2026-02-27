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


def list_available_tasks():
    """Return list of all available Metaworld task names."""
    return METAWORLD_TASKS.copy()


def create_metaworld_env(task_name="pick-place-v3", render_mode="human", seed=42):
    """
    Create a Metaworld environment via Gymnasium API.

    Args:
        task_name: Name of the Metaworld task (e.g. 'pick-place-v3')
        render_mode: 'human' for GUI window, 'rgb_array' for offscreen
        seed: Random seed for reproducibility

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
