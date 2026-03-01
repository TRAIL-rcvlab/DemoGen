"""
Utility functions for ManiSkill 3 teleoperation.
Provides environment creation and task listing helpers.

ManiSkill uses Vulkan for rendering (headless-compatible out of the box).
No DISPLAY or MUJOCO_GL configuration needed.
"""

import gymnasium as gym
import numpy as np


# Curated list of ManiSkill 3 tasks suitable for teleoperation.
# These are the most common manipulation tasks with well-defined success criteria.
MANISKILL_TASKS = [
    # --- Tabletop manipulation (Panda arm) ---
    "PickCube-v1",
    "PushCube-v1",
    "StackCube-v1",
    "PullCube-v1",
    "PokeCube-v1",
    "RollBall-v1",
    "PlaceSphere-v1",
    "PickSingleYCB-v1",
    "PickClutterYCB-v1",
    # Peg / insertion
    "PegInsertionSide-v1",
    "PlugCharger-v1",
    "LiftPegUpright-v1",
    # Tool use
    "PullCubeTool-v1",
    # Assembly
    "AssemblingKits-v1",
    "FMBAssembly1Easy-v1",
    "StackPyramid-v1",
    # Articulated objects
    "TurnFaucet-v1",
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    # Drawing
    "PushT-v1",
    "DrawTriangle-v1",
    "DrawSVG-v1",
    "TableTopFreeDraw-v1",
    # Scene manipulation
    "PutCarrotOnPlateInScene-v1",
    "PutEggplantInBasketScene-v1",
    "PutSpoonOnTableClothInScene-v1",
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
    # Flower arrangement
    "InsertFlower-v1",
    # --- Two-arm tasks ---
    "TwoRobotPickCube-v1",
    "TwoRobotStackCube-v1",
    # --- Alternative robots ---
    "PickCubeSO100-v1",          # SO-100 robot
    "SO100GraspCube-v1",         # SO-100 grasp
    # --- Dexterous hand ---
    "RotateValveLevel0-v1",
    "RotateValveLevel1-v1",
    "RotateValveLevel2-v1",
    "RotateValveLevel3-v1",
    "RotateValveLevel4-v1",
    "RotateSingleObjectInHandLevel0-v1",
    "RotateSingleObjectInHandLevel1-v1",
    "RotateSingleObjectInHandLevel2-v1",
    "RotateSingleObjectInHandLevel3-v1",
    # --- TriFinger ---
    "TriFingerRotateCubeLevel0-v1",
    "TriFingerRotateCubeLevel1-v1",
    "TriFingerRotateCubeLevel2-v1",
    "TriFingerRotateCubeLevel3-v1",
    "TriFingerRotateCubeLevel4-v1",
    # --- Humanoid / mobile ---
    "UnitreeG1PlaceAppleInBowl-v1",
    "UnitreeG1TransportBox-v1",
]

# Chinese labels for each task, used by the replay UI dropdown.
MANISKILL_TASK_LABELS = {
    # 桌面操作 (Panda 机械臂)
    "PickCube-v1":              "抓取方块",
    "PushCube-v1":              "推方块",
    "StackCube-v1":             "堆叠方块",
    "PullCube-v1":              "拉方块",
    "PokeCube-v1":              "戳方块",
    "RollBall-v1":              "滚球",
    "PlaceSphere-v1":           "放置球体",
    "PickSingleYCB-v1":         "抓取YCB物体",
    "PickClutterYCB-v1":        "杂乱中抓取YCB",
    # 插入 / 对准
    "PegInsertionSide-v1":      "侧面插钉",
    "PlugCharger-v1":           "插充电器",
    "LiftPegUpright-v1":        "竖起木钉",
    # 工具使用
    "PullCubeTool-v1":          "用工具拉方块",
    # 组装
    "AssemblingKits-v1":        "拼装套件",
    "FMBAssembly1Easy-v1":      "FMB组装(简单)",
    "StackPyramid-v1":          "堆金字塔",
    # 铰接物体
    "TurnFaucet-v1":            "拧水龙头",
    "OpenCabinetDoor-v1":       "开柜门",
    "OpenCabinetDrawer-v1":     "拉抽屉",
    # 绘画
    "PushT-v1":                 "推T形块",
    "DrawTriangle-v1":          "画三角形",
    "DrawSVG-v1":               "画SVG图案",
    "TableTopFreeDraw-v1":      "桌面自由绘画",
    # 场景操作
    "PutCarrotOnPlateInScene-v1":                       "把胡萝卜放盘子上",
    "PutEggplantInBasketScene-v1":                      "把茄子放篮子里",
    "PutSpoonOnTableClothInScene-v1":                   "把勺子放桌布上",
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1":     "绿方块叠黄方块(场景)",
    # 插花
    "InsertFlower-v1":          "插花",
    # 双臂任务
    "TwoRobotPickCube-v1":      "双臂抓方块",
    "TwoRobotStackCube-v1":     "双臂堆叠方块",
    # SO-100 机器人
    "PickCubeSO100-v1":         "SO100抓方块",
    "SO100GraspCube-v1":        "SO100夹方块",
    # 灵巧手
    "RotateValveLevel0-v1":     "旋转阀门 Lv0",
    "RotateValveLevel1-v1":     "旋转阀门 Lv1",
    "RotateValveLevel2-v1":     "旋转阀门 Lv2",
    "RotateValveLevel3-v1":     "旋转阀门 Lv3",
    "RotateValveLevel4-v1":     "旋转阀门 Lv4",
    "RotateSingleObjectInHandLevel0-v1":  "手内旋转物体 Lv0",
    "RotateSingleObjectInHandLevel1-v1":  "手内旋转物体 Lv1",
    "RotateSingleObjectInHandLevel2-v1":  "手内旋转物体 Lv2",
    "RotateSingleObjectInHandLevel3-v1":  "手内旋转物体 Lv3",
    # TriFinger
    "TriFingerRotateCubeLevel0-v1":  "三指旋转方块 Lv0",
    "TriFingerRotateCubeLevel1-v1":  "三指旋转方块 Lv1",
    "TriFingerRotateCubeLevel2-v1":  "三指旋转方块 Lv2",
    "TriFingerRotateCubeLevel3-v1":  "三指旋转方块 Lv3",
    "TriFingerRotateCubeLevel4-v1":  "三指旋转方块 Lv4",
    # 人形 / 移动
    "UnitreeG1PlaceAppleInBowl-v1":  "G1人形: 苹果放碗里",
    "UnitreeG1TransportBox-v1":      "G1人形: 搬运箱子",
}


def list_available_tasks():
    """Return list of available ManiSkill task names."""
    return MANISKILL_TASKS.copy()


def get_task_label(env_id):
    """Return the Chinese label for a task, or the env_id itself if no label exists."""
    return MANISKILL_TASK_LABELS.get(env_id, env_id)


def list_official_downloadable_tasks():
    """Return env IDs supported by ManiSkill's official download script."""
    try:
        from mani_skill.utils.download_demo import DATASET_SOURCES

        return sorted(DATASET_SOURCES.keys())
    except Exception:
        return []


def create_maniskill_env(
    task_name="PickCube-v1",
    render_mode="rgb_array",
    seed=42,
    control_mode="pd_ee_delta_pos",
    obs_mode="state",
    render_width=640,
    render_height=480,
    sensor_width=128,
    sensor_height=128,
    max_episode_steps=None,
):
    """
    Create a ManiSkill 3 environment via Gymnasium API.

    Args:
        task_name: Name of the ManiSkill task (e.g. 'PickCube-v1')
        render_mode: 'rgb_array' for offscreen rendering (always use this for web teleop)
        seed: Random seed for reproducibility
        control_mode: 'pd_ee_delta_pos' (4-DOF) or 'pd_ee_delta_pose' (7-DOF)
        obs_mode: 'state' (flat vector), 'rgbd' (dict with images), etc.
        render_width: Width of the human render camera (for streaming)
        render_height: Height of the human render camera (for streaming)
        sensor_width: Width of observation cameras (for data collection)
        sensor_height: Height of observation cameras (for data collection)
        max_episode_steps: Override max steps. None = use env default.

    Returns:
        gymnasium.Env: The created environment (wrapped with CPUGymWrapper)
    """
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

    env = gym.make(
        task_name,
        num_envs=1,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps if max_episode_steps is not None else 1000000,
        sensor_configs=dict(width=sensor_width, height=sensor_height),
        human_render_camera_configs=dict(width=render_width, height=render_height),
    )

    # CPUGymWrapper converts torch tensors to numpy and unbatches (removes batch dim)
    env = CPUGymWrapper(env)

    return env


def get_env_info(env):
    """Get useful information about the environment.

    Works with both flat (state) and dict (rgbd) observation spaces.
    """
    action_space = env.action_space
    obs_space = env.observation_space

    info = {
        "action_space": action_space,
        "observation_space": obs_space,
        "action_dim": action_space.shape[0],
    }

    # obs_dim: for flat spaces it's the shape; for dict spaces, sum all leaf dims
    if hasattr(obs_space, "shape") and obs_space.shape is not None and len(obs_space.shape) > 0:
        info["obs_dim"] = obs_space.shape[0]
    else:
        # Dict observation space — compute total flat dim for DemoCollector
        info["obs_dim"] = _compute_dict_obs_dim(obs_space)

    return info


def _compute_dict_obs_dim(obs_space):
    """Recursively compute total flat dimension of a Dict observation space."""
    if hasattr(obs_space, "shape") and obs_space.shape is not None and len(obs_space.shape) > 0:
        return int(np.prod(obs_space.shape))
    if hasattr(obs_space, "spaces"):
        return sum(_compute_dict_obs_dim(v) for v in obs_space.spaces.values())
    return 0
