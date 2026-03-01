"""
ManiSkill 3 Trajectory Replay Module.

Provides functions and a CLI for:
1. Downloading official ManiSkill demonstration datasets
2. Replaying trajectories (converting control modes, adding observations)
3. Loading replayed trajectory data for visualization or training

ManiSkill stores demos at ~/.maniskill/demos/<env_id>/<source>/trajectory.h5
Replay produces: trajectory.<obs_mode>.<control_mode>.<sim_backend>.h5

Reference: https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("maniskill_replay")

# Default demo root directory
DEMO_ROOT = Path.home() / ".maniskill" / "demos"

# Sources available from ManiSkill
DEMO_SOURCES = ["motionplanning", "rl", "teleop"]


def _to_numpy(x):
    """Convert torch/numpy/scalar to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _flatten_obs(obs):
    """Flatten nested observation dict into 1D float32 array."""
    if isinstance(obs, dict):
        parts = []

        def _walk(d):
            for key in sorted(d.keys()):
                val = d[key]
                if isinstance(val, dict):
                    _walk(val)
                else:
                    arr = _to_numpy(val)
                    # Unbatch common ManiSkill shape: (1, ...)
                    if arr.ndim > 0 and arr.shape[0] == 1:
                        arr = arr[0]
                    parts.append(arr.reshape(-1).astype(np.float32))

        _walk(obs)
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros((0,), dtype=np.float32)

    arr = _to_numpy(obs)
    if arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.reshape(-1).astype(np.float32)


def _extract_depth_from_obs(obs):
    """Extract depth map (meters) from ManiSkill rgbd observation dict."""
    if not isinstance(obs, dict):
        return None
    sensor_data = obs.get("sensor_data", {})
    for cam_name in ("base_camera", "hand_camera", "sensor_camera"):
        cam_data = sensor_data.get(cam_name, {})
        if "depth" not in cam_data:
            continue
        depth = _to_numpy(cam_data["depth"])
        # Common shapes: (H,W,1) or (1,H,W,1)
        if depth.ndim == 4 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        depth = depth.astype(np.float32)
        # ManiSkill depth may be int16 millimeters; normalize to meters.
        if depth.size > 0 and np.nanmax(depth) > 20.0:
            depth = depth / 1000.0
        return depth
    return None


def _farthest_point_sampling(points, n_points):
    """Farthest Point Sampling (same strategy as teleop_env_manager)."""
    n = len(points)
    if n == 0:
        return np.zeros((n_points, 3), dtype=np.float32)
    if n <= n_points:
        idx = np.arange(n)
        pad = np.random.choice(n, n_points - n, replace=True)
        idx = np.concatenate([idx, pad])
        return points[idx]

    selected = np.zeros(n_points, dtype=np.int64)
    selected[0] = np.random.randint(n)
    distances = np.full(n, np.inf)

    for i in range(1, n_points):
        last = points[selected[i - 1]]
        d = np.sum((points - last) ** 2, axis=1)
        distances = np.minimum(distances, d)
        selected[i] = np.argmax(distances)

    return points[selected]


def _depth_to_pointcloud_dp3(depth, fov=45.0, n_points=512):
    """Convert depth map to DP3-style point cloud.

    Pipeline matches teleop_env_manager in both simulators:
    1) depth->xyz via intrinsics, 2) workspace crop, 3) DBSCAN outlier removal,
    4) FPS to fixed point count.
    """
    if depth is None:
        return np.zeros((n_points, 3), dtype=np.float32)

    h, w = depth.shape
    f = 0.5 * h / np.tan(np.radians(fov) / 2)
    cx, cy = w / 2, h / 2
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - cx) * z / f
    y = (v - cy) * z / f
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    valid = (z.reshape(-1) > 0.1) & (z.reshape(-1) < 5.0)
    points = points[valid]
    if len(points) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)

    median = np.median(points, axis=0)
    std = np.std(points, axis=0)
    margin = np.maximum(std * 3, 0.1)
    ws_min = median - margin
    ws_max = median + margin
    in_ws = np.all((points >= ws_min) & (points <= ws_max), axis=1)
    points = points[in_ws]
    if len(points) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)

    try:
        from sklearn.cluster import DBSCAN

        if len(points) > 50:
            db = DBSCAN(eps=0.03, min_samples=5).fit(points)
            labels = db.labels_
            points = points[labels != -1]
    except Exception:
        pass

    if len(points) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)

    points = _farthest_point_sampling(points, n_points)
    return points.astype(np.float32)


# -----------------------------------------------------------------------
#  Discovery: list available demos on disk
# -----------------------------------------------------------------------

def get_demo_root():
    """Return the ManiSkill demo root directory."""
    return DEMO_ROOT


def list_local_demos(env_id=None):
    """List locally available demo datasets.

    Returns:
        list[dict]: Each dict has keys: env_id, source, traj_path, num_episodes, control_mode
    """
    demos = []
    root = DEMO_ROOT
    if not root.exists():
        return demos

    env_dirs = [root / env_id] if env_id else sorted(root.iterdir())

    for env_dir in env_dirs:
        if not env_dir.is_dir():
            continue
        for source_dir in sorted(env_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            # Find all trajectory h5 files in this source dir
            for h5_file in sorted(source_dir.glob("trajectory*.h5")):
                json_file = h5_file.with_suffix(".json")
                info = {
                    "env_id": env_dir.name,
                    "source": source_dir.name,
                    "filename": h5_file.name,
                    "traj_path": str(h5_file),
                    "num_episodes": 0,
                    "control_mode": "unknown",
                    "is_replayed": h5_file.name != "trajectory.h5",
                }
                if json_file.exists():
                    try:
                        with open(json_file) as f:
                            meta = json.load(f)
                        info["num_episodes"] = len(meta.get("episodes", []))
                        # control_mode from first episode
                        eps = meta.get("episodes", [])
                        if eps:
                            info["control_mode"] = eps[0].get("control_mode", "unknown")
                    except Exception:
                        pass
                demos.append(info)
    return demos


# -----------------------------------------------------------------------
#  Download: wraps `python -m mani_skill.utils.download_demo`
# -----------------------------------------------------------------------

def download_demos(env_id, source="all", verbose=True):
    """Download official ManiSkill demos for a given environment.

    Args:
        env_id: Environment name (e.g., 'PickCube-v1')
        source: 'motionplanning', 'rl', 'teleop', or 'all'
        verbose: Print progress
    """
    cmd = [sys.executable, "-m", "mani_skill.utils.download_demo", env_id]
    if verbose:
        logger.info(f"Downloading demos: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Download failed:\n{result.stderr}")
        return False
    if verbose:
        logger.info(f"Download complete for {env_id}")
        if result.stdout:
            logger.info(result.stdout.strip())
    return True


# -----------------------------------------------------------------------
#  Replay: wraps `python -m mani_skill.trajectory.replay_trajectory`
# -----------------------------------------------------------------------

def replay_trajectory(
    traj_path,
    obs_mode="state",
    control_mode="pd_ee_delta_pos",
    count=None,
    save_traj=True,
    save_video=False,
    verbose=True,
):
    """Replay a ManiSkill trajectory file, converting to desired obs/control mode.

    This wraps `python -m mani_skill.trajectory.replay_trajectory` which:
    - Re-executes each episode from recorded env_states
    - Converts actions to the target control_mode
    - Optionally records observations in the target obs_mode
    - Saves the result as a new .h5 file alongside the original

    Args:
        traj_path: Path to the source trajectory.h5 file
        obs_mode: Target observation mode ('state', 'rgbd', 'pointcloud')
        control_mode: Target control mode ('pd_ee_delta_pos', 'pd_ee_delta_pose', etc.)
        count: Number of episodes to replay (None = all)
        save_traj: Save replayed trajectory to new .h5 file
        save_video: Save video of replay
        verbose: Print progress

    Returns:
        str or None: Path to the output .h5 file if save_traj=True, else None
    """
    cmd = [
        sys.executable, "-m", "mani_skill.trajectory.replay_trajectory",
        "--traj-path", str(traj_path),
        "-o", obs_mode,
        "-c", control_mode,
    ]
    if count is not None:
        cmd += ["--count", str(count)]
    if save_traj:
        cmd.append("--save-traj")
    if save_video:
        cmd.append("--save-video")

    if verbose:
        logger.info(f"Replaying: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    if result.returncode != 0:
        stderr = result.stderr if hasattr(result, "stderr") and result.stderr else ""
        logger.error(f"Replay failed (exit code {result.returncode}):\n{stderr}")
        return None

    if save_traj:
        # Infer output path: trajectory.<obs_mode>.<control_mode>.physx_cpu.h5
        traj_dir = Path(traj_path).parent
        stem = Path(traj_path).stem  # e.g., "trajectory"
        # ManiSkill naming: <stem>.<obs_mode>.<control_mode>.<backend>.h5
        out_name = f"{stem}.{obs_mode}.{control_mode}.physx_cpu.h5"
        out_path = traj_dir / out_name
        if out_path.exists():
            if verbose:
                logger.info(f"Replayed trajectory saved: {out_path}")
            return str(out_path)
        # Fallback: search for recently created h5 files
        import glob
        pattern = str(traj_dir / f"{stem}.*.h5")
        candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if candidates:
            if verbose:
                logger.info(f"Replayed trajectory saved: {candidates[0]}")
            return candidates[0]
        logger.warning("Could not locate output file")
        return None

    return None


# -----------------------------------------------------------------------
#  Load: read trajectory data from h5 files
# -----------------------------------------------------------------------

def load_trajectory(traj_path, episode_ids=None):
    """Load trajectory data from an h5 file.

    Args:
        traj_path: Path to trajectory .h5 file
        episode_ids: List of episode indices to load (None = all)

    Returns:
        list[dict]: Each dict has keys: actions, env_states, rewards, success, etc.
    """
    import h5py

    episodes = []
    with h5py.File(traj_path, "r") as f:
        if episode_ids is None:
            episode_ids = sorted(
                [int(k.split("_")[1]) for k in f.keys() if k.startswith("traj_")]
            )

        for eid in episode_ids:
            key = f"traj_{eid}"
            if key not in f:
                logger.warning(f"Episode {key} not found in {traj_path}")
                continue

            g = f[key]
            ep = {"episode_id": eid}
            if "actions" in g:
                ep["actions"] = g["actions"][:]
            if "rewards" in g:
                ep["rewards"] = g["rewards"][:]
            if "success" in g:
                ep["success"] = g["success"][:]
            if "terminated" in g:
                ep["terminated"] = g["terminated"][:]
            if "truncated" in g:
                ep["truncated"] = g["truncated"][:]

            # env_states: actors and articulations
            if "env_states" in g:
                ep["env_states"] = {}
                es = g["env_states"]
                if "actors" in es:
                    ep["env_states"]["actors"] = {
                        name: es["actors"][name][:] for name in es["actors"]
                    }
                if "articulations" in es:
                    ep["env_states"]["articulations"] = {
                        name: es["articulations"][name][:] for name in es["articulations"]
                    }

            episodes.append(ep)

    return episodes


def load_trajectory_metadata(traj_path):
    """Load metadata (JSON sidecar) for a trajectory file.

    Args:
        traj_path: Path to trajectory .h5 file

    Returns:
        dict or None: Parsed JSON metadata
    """
    json_path = Path(traj_path).with_suffix(".json")
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


# -----------------------------------------------------------------------
#  Visual replay: re-execute episodes and yield rendered frames
# -----------------------------------------------------------------------

def _parse_trajectory_metadata(traj_path):
    """Parse metadata from a trajectory JSON sidecar.

    Extracts env_id, control_mode, sim_backend, and per-episode metadata.

    Returns:
        dict with keys: env_id, control_mode, sim_backend, env_kwargs, episodes, raw
    """
    meta = load_trajectory_metadata(traj_path)
    if meta is None:
        raise ValueError(f"No metadata (JSON sidecar) found for {traj_path}")

    env_info = meta.get("env_info", {})
    env_kwargs = env_info.get("env_kwargs", {})

    env_id = env_info.get("env_id")
    if env_id is None:
        env_id = meta.get("env_id")  # Fallback for old formats
    if env_id is None:
        # Last resort: infer from directory structure
        # ~/.maniskill/demos/<env_id>/<source>/trajectory.h5
        traj_p = Path(traj_path)
        if traj_p.parent.parent.parent.name == "demos":
            env_id = traj_p.parent.parent.name

    control_mode = env_kwargs.get("control_mode")
    # Also check per-episode control_mode (replayed files store it there)
    episodes_meta = meta.get("episodes", [])
    if control_mode is None and episodes_meta:
        control_mode = episodes_meta[0].get("control_mode")
    if control_mode is None:
        control_mode = "pd_ee_delta_pos"  # Safe default

    sim_backend = env_kwargs.get("sim_backend", "auto")
    # Normalize: "auto" defaults to physx_cpu for single-env replay
    if sim_backend == "auto":
        sim_backend = "physx_cpu"

    return {
        "env_id": env_id,
        "control_mode": control_mode,
        "sim_backend": sim_backend,
        "env_kwargs": env_kwargs,
        "max_episode_steps": env_info.get("max_episode_steps"),
        "episodes": episodes_meta,
        "raw": meta,
    }


def visual_replay_generator(
    traj_path,
    episode_id=0,
    control_mode=None,
    render_width=640,
    render_height=480,
):
    """Generator that replays a trajectory episode and yields rendered frames.

    Uses env_states (not actions) to set the simulation state at each step,
    following ManiSkill's recommended approach for faithful replay. This avoids
    action dimension mismatches, truncation issues, and non-determinism.

    The control_mode and sim_backend are auto-detected from trajectory metadata
    unless explicitly overridden.

    Args:
        traj_path: Path to the trajectory .h5 file (must have env_states)
        episode_id: Which episode to replay
        control_mode: Control mode override (None = auto-detect from metadata)
        render_width: Render resolution width
        render_height: Render resolution height

    Yields:
        dict: {
            'frame': np.ndarray (H,W,3) uint8,
            'step': int,
            'total_steps': int,
            'reward': float,
            'success': bool,
        }
    """
    import h5py

    # Parse metadata (auto-detect env_id, control_mode, sim_backend)
    tmeta = _parse_trajectory_metadata(traj_path)

    env_id = tmeta["env_id"]
    if env_id is None:
        raise ValueError("Cannot determine env_id from trajectory metadata")

    if control_mode is None:
        control_mode = tmeta["control_mode"]

    # Always use physx_cpu for visual replay, regardless of what the trajectory
    # was collected with. GPU PhysX (physx_cuda) can only be initialized once
    # per process and conflicts with the teleop server's own PhysX instance.
    # ManiSkill docs recommend using --use-first-env-state for cross-backend replay.
    sim_backend = "physx_cpu"

    # Find episode metadata
    ep_meta = None
    for em in tmeta["episodes"]:
        if em.get("episode_id") == episode_id:
            ep_meta = em
            break

    seed = 0
    if ep_meta:
        seed = ep_meta.get("episode_seed", ep_meta.get("reset_kwargs", {}).get("seed", 0))

    # Load episode data from H5
    with h5py.File(traj_path, "r") as f:
        key = f"traj_{episode_id}"
        if key not in f:
            raise ValueError(f"Episode {key} not found in {traj_path}")
        g = f[key]

        rewards = g["rewards"][:] if "rewards" in g else None
        success = g["success"][:] if "success" in g else None

        # Prefer env_states for faithful replay (ManiSkill recommended approach)
        has_env_states = "env_states" in g
        if has_env_states:
            env_states_group = g["env_states"]
            # Preload all state data into memory so we can close the file
            env_states_data = {}
            if "actors" in env_states_group:
                env_states_data["actors"] = {
                    name: env_states_group["actors"][name][:]
                    for name in env_states_group["actors"]
                }
            if "articulations" in env_states_group:
                env_states_data["articulations"] = {
                    name: env_states_group["articulations"][name][:]
                    for name in env_states_group["articulations"]
                }
            # Number of states = number of steps + 1 (initial state included)
            first_key = list(env_states_data.get("actors", env_states_data.get("articulations", {})).keys())[0]
            container = env_states_data.get("actors", env_states_data.get("articulations", {}))
            total_steps = container[first_key].shape[0] - 1  # subtract initial state
        else:
            # Fallback: use actions (less faithful but works for replayed files)
            actions = g["actions"][:] if "actions" in g else None
            if actions is None:
                raise ValueError(f"No env_states or actions in episode {episode_id}")
            total_steps = len(actions)
            env_states_data = None

    logger.info(
        f"Visual replay: env={env_id}, episode={episode_id}, "
        f"control_mode={control_mode}, sim_backend={sim_backend}, "
        f"steps={total_steps}, use_env_states={has_env_states}"
    )

    # Create environment for visual replay
    # Use a very large max_episode_steps to prevent truncation during replay
    import gymnasium as gym
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state",
        control_mode=control_mode,
        render_mode="rgb_array",
        sim_backend=sim_backend,
        max_episode_steps=1000000,
        human_render_camera_configs=dict(width=render_width, height=render_height),
    )
    env = CPUGymWrapper(env)

    try:
        # Reset with episode seed
        reset_kwargs = {}
        if ep_meta:
            reset_kwargs = ep_meta.get("reset_kwargs", {})
        obs, _ = env.reset(seed=reset_kwargs.get("seed", seed), options=reset_kwargs.get("options", {}))

        if has_env_states:
            # Set the initial state from the trajectory (step 0)
            # This is critical for cross-backend replay (physx_cuda -> physx_cpu)
            # as the random seed may produce different initial states on different backends.
            init_state = {}
            if "actors" in env_states_data:
                init_state["actors"] = {
                    name: data[0] for name, data in env_states_data["actors"].items()
                }
            if "articulations" in env_states_data:
                init_state["articulations"] = {
                    name: data[0] for name, data in env_states_data["articulations"].items()
                }
            env.unwrapped.set_state_dict(init_state)

            # State-based replay: set env state at each step and render
            # This is the ManiSkill-recommended approach for faithful replay
            for step_idx in range(total_steps):
                # Build state dict for this step (step_idx + 1 because index 0 is initial state)
                state_idx = step_idx + 1
                state_dict = {}
                if "actors" in env_states_data:
                    state_dict["actors"] = {
                        name: data[state_idx] for name, data in env_states_data["actors"].items()
                    }
                if "articulations" in env_states_data:
                    state_dict["articulations"] = {
                        name: data[state_idx] for name, data in env_states_data["articulations"].items()
                    }

                # Set the environment state directly
                env.unwrapped.set_state_dict(state_dict)

                frame = env.render()

                yield {
                    "frame": frame,
                    "step": step_idx + 1,
                    "total_steps": total_steps,
                    "reward": float(rewards[step_idx]) if rewards is not None and step_idx < len(rewards) else 0.0,
                    "success": bool(success[step_idx]) if success is not None and step_idx < len(success) else False,
                }
        else:
            # Action-based replay fallback (for replayed files without env_states)
            for step_idx in range(total_steps):
                action = actions[step_idx]
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, info = env.step(action)

                frame = env.render()

                yield {
                    "frame": frame,
                    "step": step_idx + 1,
                    "total_steps": total_steps,
                    "reward": float(rewards[step_idx]) if rewards is not None and step_idx < len(rewards) else float(reward),
                    "success": bool(success[step_idx]) if success is not None and step_idx < len(success) else info.get("success", False),
                }

                if terminated or truncated:
                    break
    finally:
        env.close()


def export_episode_to_dp3_zarr(
    traj_path,
    episode_id=0,
    control_mode=None,
    out_dir="data/datasets/teleop",
    n_points=512,
    sensor_width=128,
    sensor_height=128,
):
    """Replay one episode in current env and export a DP3 zarr dataset.

    Output format (same as DemoGen/DP3 expected structure):
      data/agent_pos:  (T, D)
      data/point_cloud:(T, n_points, 3)
      data/action:     (T, A)
      meta/episode_ends: (1,)
    """
    import h5py

    try:
        import zarr
    except Exception as e:
        raise RuntimeError("zarr 未安装，无法导出。请先安装 zarr==2.12.0") from e

    tmeta = _parse_trajectory_metadata(traj_path)
    env_id = tmeta["env_id"]
    if env_id is None:
        raise ValueError("Cannot determine env_id from trajectory metadata")
    if control_mode is None:
        control_mode = tmeta["control_mode"]

    ep_meta = None
    for em in tmeta["episodes"]:
        if em.get("episode_id") == episode_id:
            ep_meta = em
            break
    if ep_meta is None:
        raise ValueError(f"Episode {episode_id} not found in trajectory metadata")

    with h5py.File(traj_path, "r") as f:
        key = f"traj_{episode_id}"
        if key not in f:
            raise ValueError(f"Episode {key} not found in {traj_path}")
        g = f[key]
        if "env_states" not in g:
            raise ValueError("当前仅支持包含 env_states 的轨迹导出")

        env_states_group = g["env_states"]
        env_states_data = {}
        if "actors" in env_states_group:
            env_states_data["actors"] = {
                name: env_states_group["actors"][name][:] for name in env_states_group["actors"]
            }
        if "articulations" in env_states_group:
            env_states_data["articulations"] = {
                name: env_states_group["articulations"][name][:]
                for name in env_states_group["articulations"]
            }

        actions = g["actions"][:] if "actions" in g else None

        container = env_states_data.get("actors") or env_states_data.get("articulations")
        if not container:
            raise ValueError("env_states 缺少 actors/articulations 数据")
        first_key = list(container.keys())[0]
        total_steps = container[first_key].shape[0] - 1

    logger.info(
        f"Export DP3 zarr: env={env_id}, episode={episode_id}, "
        f"control_mode={control_mode}, steps={total_steps}"
    )

    import gymnasium as gym
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="rgbd",
        control_mode=control_mode,
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        max_episode_steps=1000000,
        sensor_configs=dict(width=sensor_width, height=sensor_height),
        human_render_camera_configs=dict(width=640, height=480),
    )
    env = CPUGymWrapper(env)

    agent_pos_list = []
    point_cloud_list = []
    action_list = []

    reset_kwargs = ep_meta.get("reset_kwargs", {})
    seed = ep_meta.get("episode_seed", reset_kwargs.get("seed", 0))

    try:
        env.reset(seed=reset_kwargs.get("seed", seed), options=reset_kwargs.get("options", {}))

        init_state = {}
        if "actors" in env_states_data:
            init_state["actors"] = {
                name: data[0] for name, data in env_states_data["actors"].items()
            }
        if "articulations" in env_states_data:
            init_state["articulations"] = {
                name: data[0] for name, data in env_states_data["articulations"].items()
            }
        env.unwrapped.set_state_dict(init_state)

        for step_idx in range(total_steps):
            state_idx = step_idx + 1
            state_dict = {}
            if "actors" in env_states_data:
                state_dict["actors"] = {
                    name: data[state_idx] for name, data in env_states_data["actors"].items()
                }
            if "articulations" in env_states_data:
                state_dict["articulations"] = {
                    name: data[state_idx] for name, data in env_states_data["articulations"].items()
                }
            env.unwrapped.set_state_dict(state_dict)

            obs = env.unwrapped.get_obs()
            flat_obs = _flatten_obs(obs)
            depth = _extract_depth_from_obs(obs)
            point_cloud = _depth_to_pointcloud_dp3(depth, n_points=n_points)

            if actions is not None and step_idx < len(actions):
                action = np.asarray(actions[step_idx], dtype=np.float32)
            else:
                action = np.zeros((env.action_space.shape[0],), dtype=np.float32)

            agent_pos_list.append(flat_obs)
            point_cloud_list.append(point_cloud)
            action_list.append(action)
    finally:
        env.close()

    if len(agent_pos_list) == 0:
        raise RuntimeError("未产生可导出的步数据")

    agent_pos = np.stack(agent_pos_list, axis=0).astype(np.float32)
    point_clouds = np.stack(point_cloud_list, axis=0).astype(np.float32)
    actions_arr = np.stack(action_list, axis=0).astype(np.float32)
    episode_ends = np.array([len(agent_pos_list)], dtype=np.int64)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    zarr_path = out_root / f"replay_{env_id}_ep{episode_id}_{timestamp}.zarr"

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    root = zarr.group(str(zarr_path))
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    data_group.create_dataset(
        "agent_pos",
        data=agent_pos,
        chunks=(min(100, len(agent_pos_list)), agent_pos.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    data_group.create_dataset(
        "point_cloud",
        data=point_clouds,
        chunks=(min(100, len(agent_pos_list)), point_clouds.shape[1], point_clouds.shape[2]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    data_group.create_dataset(
        "action",
        data=actions_arr,
        chunks=(min(100, len(agent_pos_list)), actions_arr.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    meta_group.create_dataset(
        "episode_ends",
        data=episode_ends,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

    return {
        "zarr_path": str(zarr_path),
        "env_id": env_id,
        "episode_id": episode_id,
        "steps": int(len(agent_pos_list)),
        "agent_pos_shape": list(agent_pos.shape),
        "point_cloud_shape": list(point_clouds.shape),
        "action_shape": list(actions_arr.shape),
    }


# -----------------------------------------------------------------------
#  CLI entry point
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ManiSkill 3 Trajectory Replay Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download demos for PickCube-v1
  python -m maniskill_teleop.replay download --env-id PickCube-v1

  # List all local demos
  python -m maniskill_teleop.replay list

  # Replay motion planning demos to pd_ee_delta_pos control
  python -m maniskill_teleop.replay replay \\
      --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \\
      --control-mode pd_ee_delta_pos --obs-mode state --count 10

  # Show info about a trajectory file
  python -m maniskill_teleop.replay info \\
      --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- download ---
    dl_parser = subparsers.add_parser("download", help="Download official demos")
    dl_parser.add_argument("--env-id", required=True, help="Environment ID (e.g., PickCube-v1)")
    dl_parser.add_argument("--source", default="all", choices=DEMO_SOURCES + ["all"])

    # --- list ---
    list_parser = subparsers.add_parser("list", help="List locally available demos")
    list_parser.add_argument("--env-id", default=None, help="Filter by environment ID")

    # --- replay ---
    rp_parser = subparsers.add_parser("replay", help="Replay trajectory to new control/obs mode")
    rp_parser.add_argument("--traj-path", required=True, help="Path to trajectory.h5")
    rp_parser.add_argument("--obs-mode", default="state", choices=["state", "rgbd", "pointcloud"])
    rp_parser.add_argument("--control-mode", default="pd_ee_delta_pos")
    rp_parser.add_argument("--count", type=int, default=None, help="Number of episodes to replay")
    rp_parser.add_argument("--save-video", action="store_true", help="Also save replay video")

    # --- info ---
    info_parser = subparsers.add_parser("info", help="Show info about a trajectory file")
    info_parser.add_argument("--traj-path", required=True, help="Path to trajectory.h5")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    if args.command == "download":
        download_demos(args.env_id, source=args.source)

    elif args.command == "list":
        demos = list_local_demos(args.env_id)
        if not demos:
            print("No local demos found.")
            print(f"Demo root: {DEMO_ROOT}")
            print("Use 'download' command to fetch demos.")
            return
        print(f"{'Env ID':<25} {'Source':<18} {'File':<55} {'Episodes':>8}  {'Control Mode'}")
        print("-" * 140)
        for d in demos:
            tag = " [replayed]" if d["is_replayed"] else ""
            print(
                f"{d['env_id']:<25} {d['source']:<18} {d['filename']:<55} "
                f"{d['num_episodes']:>8}  {d['control_mode']}{tag}"
            )

    elif args.command == "replay":
        out = replay_trajectory(
            traj_path=args.traj_path,
            obs_mode=args.obs_mode,
            control_mode=args.control_mode,
            count=args.count,
            save_traj=True,
            save_video=args.save_video,
        )
        if out:
            print(f"\nOutput: {out}")

    elif args.command == "info":
        try:
            tmeta = _parse_trajectory_metadata(args.traj_path)
            print(f"Trajectory:  {args.traj_path}")
            print(f"Env ID:      {tmeta['env_id']}")
            print(f"Control:     {tmeta['control_mode']}")
            print(f"Sim backend: {tmeta['sim_backend']}")
            episodes = tmeta["episodes"]
            print(f"Episodes:    {len(episodes)}")
            if episodes:
                steps = [e.get("elapsed_steps", 0) for e in episodes]
                successes = sum(1 for e in episodes if e.get("success", False))
                print(f"Steps:       min={min(steps)} max={max(steps)} mean={np.mean(steps):.1f}")
                print(f"Success:     {successes}/{len(episodes)} ({100*successes/len(episodes):.1f}%)")
        except Exception as e:
            print(f"Metadata error: {e}")
            meta = load_trajectory_metadata(args.traj_path)
            if meta is None:
                print(f"No JSON metadata found for {args.traj_path}")

        # Also show h5 structure
        try:
            import h5py
            with h5py.File(args.traj_path, "r") as f:
                traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
                print(f"\nH5 episodes: {len(traj_keys)}")
                if traj_keys:
                    g = f[traj_keys[0]]
                    print(f"Keys in {traj_keys[0]}: {list(g.keys())}")
                    if "actions" in g:
                        print(f"  actions: {g['actions'].shape} {g['actions'].dtype}")
                    if "env_states" in g:
                        print(f"  env_states: present (use_env_states replay supported)")
        except Exception as e:
            print(f"Could not read h5: {e}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
