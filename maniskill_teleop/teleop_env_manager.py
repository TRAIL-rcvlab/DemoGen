"""
ManiSkill 3 Teleoperation Environment Manager.

Manages the ManiSkill environment lifecycle, rendering, camera control,
and action computation for web-based teleoperation.

Key differences from Metaworld:
- Rendering uses Vulkan via SAPIEN (headless out of the box, no MUJOCO_GL needed)
- Depth is available directly from RGBD observations (no separate MuJoCo renderer)
- Supports both 4-DOF (pd_ee_delta_pos) and 7-DOF (pd_ee_delta_pose) control
- Camera manipulation via SAPIEN viewer API (not MuJoCo viewer.cam)
"""

import io
import logging
import time

import numpy as np
import base64
from PIL import Image

logger = logging.getLogger("teleop_server")


class TeleopState:
    """Shared mutable state for a ManiSkill teleoperation session.

    Drop-in replacement for the Metaworld TeleopState — exposes the same
    public interface so teleop_server.py can use either one interchangeably.
    """

    # Fixed camera params for data collection (not affected by user rotation)
    DEPTH_WIDTH = 128
    DEPTH_HEIGHT = 128

    def __init__(self):
        self.env = None
        self.task_name = "PickCube-v1"
        self.speed = 0.5
        self.seed = 42
        self.render_width = 640
        self.render_height = 480
        self.jpeg_quality = 80
        self.target_fps = 30
        self.debug = False

        # ManiSkill-specific settings
        self.control_mode = "pd_ee_delta_pos"  # or "pd_ee_delta_pose" for 7-DOF
        self.obs_mode = "state"  # "state" for lightweight teleop, "rgbd" for data

        # Action state (updated by WebSocket messages)
        self.keys_pressed = set()
        self.gripper_target = 0.0  # -1 open, +1 closed

        # Camera state (for user-controllable view)
        # ManiSkill camera pose is controlled differently from MuJoCo;
        # we store azimuth/elevation/distance for compatibility with the
        # teleop_server camera_rotate/zoom messages, but these are only
        # used for the streaming view, not for data collection cameras.
        self.camera_azimuth = 180.0
        self.camera_elevation = -25.0
        self.camera_distance = 1.5
        self.camera_lookat = [0.0, 0.0, 0.15]

        # Orientation deltas (for 7-DOF mode)
        self.orientation_delta = [0.0, 0.0, 0.0]  # [roll, pitch, yaw]

        # Episode tracking
        self.step_count = 0
        self.episode_count = 0
        self.last_reward = 0.0
        self.last_success = False
        self.obs = None

        # Data collection (reuse DemoCollector from metaworld_teleop)
        self.collector = None
        self.recording = False
        self.save_dir = "data/datasets/teleop"

        # Control
        self.running = False
        self.reset_flag = False
        self._pending_task = None  # Set by handle_client_message, consumed by main loop
        self._gamepad_action = None

    def create_env(self):
        """Create or recreate the ManiSkill environment."""
        if self.env is not None:
            self.env.close()

        from maniskill_teleop.utils import create_maniskill_env, get_env_info

        self.env = create_maniskill_env(
            task_name=self.task_name,
            render_mode="rgb_array",
            seed=self.seed,
            control_mode=self.control_mode,
            obs_mode=self.obs_mode,
            render_width=self.render_width,
            render_height=self.render_height,
            sensor_width=self.DEPTH_WIDTH,
            sensor_height=self.DEPTH_HEIGHT,
        )
        env_info = get_env_info(self.env)

        # Lazy import: DemoCollector lives in metaworld_teleop but is simulator-agnostic
        from metaworld_teleop.data_collector import DemoCollector
        self.collector = DemoCollector(
            save_dir=self.save_dir,
            obs_dim=env_info["obs_dim"],
            act_dim=env_info["action_dim"],
        )

        self.obs, _ = self.env.reset(seed=self.seed)
        self.step_count = 0
        self.episode_count = 1
        self.last_reward = 0.0
        self.last_success = False
        self.keys_pressed.clear()
        self.gripper_target = 0.0
        self.recording = False
        logger.info(
            f"ManiSkill env created: {self.task_name} "
            f"(control={self.control_mode}, obs={self.obs_mode})"
        )

    def compute_action(self):
        """Compute action from current input state (gamepad or keyboard).

        For pd_ee_delta_pos (4-DOF):  [dx, dy, dz, gripper]
        For pd_ee_delta_pose (7-DOF): [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        # Prefer gamepad input if available
        if self._gamepad_action is not None:
            action = self._gamepad_action
            self._gamepad_action = None
            # Pad or trim to match action space
            action = self._fit_action(action)
            return np.clip(action, self.env.action_space.low, self.env.action_space.high)

        dx, dy, dz = 0.0, 0.0, 0.0
        if "a" in self.keys_pressed:
            dx -= self.speed
        if "d" in self.keys_pressed:
            dx += self.speed
        if "w" in self.keys_pressed:
            dy += self.speed
        if "s" in self.keys_pressed:
            dy -= self.speed
        if "q" in self.keys_pressed:
            dz += self.speed
        if "e" in self.keys_pressed:
            dz -= self.speed

        action_dim = self.env.action_space.shape[0]

        if action_dim == 4:
            # pd_ee_delta_pos: [dx, dy, dz, gripper]
            action = np.array([dx, dy, dz, self.gripper_target], dtype=np.float32)
        elif action_dim == 7:
            # pd_ee_delta_pose: [dx, dy, dz, droll, dpitch, dyaw, gripper]
            dr, dp, dyw = self.orientation_delta
            action = np.array(
                [dx, dy, dz, dr, dp, dyw, self.gripper_target], dtype=np.float32
            )
            self.orientation_delta = [0.0, 0.0, 0.0]
        else:
            # Generic fallback: fill zeros, put gripper at the end
            action = np.zeros(action_dim, dtype=np.float32)
            action[0] = dx
            if action_dim > 1:
                action[1] = dy
            if action_dim > 2:
                action[2] = dz
            action[-1] = self.gripper_target

        return np.clip(action, self.env.action_space.low, self.env.action_space.high)

    def _fit_action(self, action):
        """Pad or trim a raw 4-DOF gamepad action to match the env action space."""
        action_dim = self.env.action_space.shape[0]
        if len(action) == action_dim:
            return action
        if action_dim == 4:
            return action[:4]
        if action_dim == 7 and len(action) == 4:
            # 4-DOF gamepad → 7-DOF env: insert zero orientation deltas
            return np.array(
                [action[0], action[1], action[2], 0.0, 0.0, 0.0, action[3]],
                dtype=np.float32,
            )
        # Generic: pad with zeros or trim
        fitted = np.zeros(action_dim, dtype=np.float32)
        n = min(len(action), action_dim)
        fitted[:n] = action[:n]
        return fitted

    def apply_orientation(self):
        """Apply end-effector orientation (no-op for ManiSkill).

        In ManiSkill, orientation is part of the action vector (pd_ee_delta_pose),
        not applied via mocap body manipulation. The orientation delta is consumed
        in compute_action() for 7-DOF mode, so this is intentionally empty.
        """
        pass

    def render_jpeg_base64(self):
        """Render current frame as base64-encoded JPEG (user view)."""
        frame = self.env.render()  # returns (H, W, 3) uint8 via CPUGymWrapper
        img = Image.fromarray(frame)
        if img.size != (self.render_width, self.render_height):
            img = img.resize((self.render_width, self.render_height), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def render_fixed_camera_frame(self):
        """Render frame from fixed data camera. Returns numpy array (H,W,3).

        ManiSkill's render() uses the human_render_camera which is fixed,
        so for now this returns the same as the user view.
        Camera orbit control for the user view will be added later
        if SAPIEN viewer camera API proves accessible.
        """
        return self.env.render()

    def render_depth(self):
        """Render depth image from observation cameras.

        In ManiSkill with obs_mode='rgbd', depth is available in the obs dict:
            obs['sensor_data']['base_camera']['depth']  -> (H, W, 1) int16

        For obs_mode='state', depth is not available (returns None).
        """
        if self.obs_mode != "rgbd" or self.obs is None:
            return None
        try:
            # Navigate the obs dict to find depth
            if isinstance(self.obs, dict):
                sensor_data = self.obs.get("sensor_data", {})
                # Try common camera names
                for cam_name in ("base_camera", "hand_camera", "sensor_camera"):
                    cam_data = sensor_data.get(cam_name, {})
                    if "depth" in cam_data:
                        depth = cam_data["depth"]
                        # depth shape: (H, W, 1) int16 — squeeze and convert
                        if depth.ndim == 3:
                            depth = depth[:, :, 0]
                        return depth.astype(np.float32)
            return None
        except Exception as e:
            if self.debug:
                logger.debug(f"Depth extraction error: {e}")
            return None

    def depth_to_pointcloud(self, depth, fov=45.0, n_points=512):
        """Convert depth image to DP3-style point cloud.

        Pipeline (same as Metaworld version):
        1. Depth → 3D points via camera intrinsics
        2. Crop workspace (remove table, background)
        3. DBSCAN outlier removal (if sklearn available)
        4. Farthest Point Sampling (FPS) downsample to n_points

        Args:
            depth: (H,W) float32 depth image
            fov: camera field of view in degrees
            n_points: target number of points (512 for sim, 1024 for real)
        Returns:
            (n_points, 3) float32 point cloud
        """
        h, w = depth.shape
        f = 0.5 * h / np.tan(np.radians(fov) / 2)
        cx, cy = w / 2, h / 2
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth
        x = (u - cx) * z / f
        y = (v - cy) * z / f
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # Step 1: Filter invalid depth
        valid = (z.reshape(-1) > 0.1) & (z.reshape(-1) < 5.0)
        points = points[valid]

        if len(points) == 0:
            return np.zeros((n_points, 3), dtype=np.float32)

        # Step 2: Workspace crop
        median = np.median(points, axis=0)
        std = np.std(points, axis=0)
        margin = np.maximum(std * 3, 0.1)
        ws_min = median - margin
        ws_max = median + margin
        in_ws = np.all((points >= ws_min) & (points <= ws_max), axis=1)
        points = points[in_ws]

        if len(points) == 0:
            return np.zeros((n_points, 3), dtype=np.float32)

        # Step 3: DBSCAN outlier removal (optional)
        try:
            from sklearn.cluster import DBSCAN

            if len(points) > 50:
                db = DBSCAN(eps=0.03, min_samples=5).fit(points)
                labels = db.labels_
                points = points[labels != -1]
        except ImportError:
            pass

        if len(points) == 0:
            return np.zeros((n_points, 3), dtype=np.float32)

        # Step 4: Farthest Point Sampling
        points = self._farthest_point_sampling(points, n_points)
        return points.astype(np.float32)

    @staticmethod
    def _farthest_point_sampling(points, n_points):
        """Farthest Point Sampling (FPS) for point cloud downsampling."""
        n = len(points)
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

    def _flatten_obs(self, obs):
        """Flatten a dict observation into a 1D numpy array for DemoCollector."""
        if isinstance(obs, np.ndarray) and obs.ndim == 1:
            return obs
        if isinstance(obs, dict):
            flat_parts = []
            self._flatten_dict(obs, flat_parts)
            return np.concatenate(flat_parts).astype(np.float32)
        return np.array(obs, dtype=np.float32).flatten()

    def _flatten_dict(self, d, parts):
        """Recursively flatten a nested dict of arrays."""
        for key in sorted(d.keys()):
            val = d[key]
            if isinstance(val, dict):
                self._flatten_dict(val, parts)
            elif isinstance(val, np.ndarray):
                parts.append(val.flatten())
            else:
                parts.append(np.array(val, dtype=np.float32).flatten())
