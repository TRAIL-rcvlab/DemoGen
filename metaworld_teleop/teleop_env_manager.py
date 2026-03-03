import io
import os
import time
import logging
import numpy as np
import base64
from PIL import Image

# --- numpy 2.x compatibility ---
# np.product was removed in numpy 2.0; some transitive deps still use it.
if not hasattr(np, "product"):
    np.product = np.prod

logger = logging.getLogger("teleop_server")


# Now that imports are done, configure offscreen rendering for headless servers.
# Try EGL first (GPU-accelerated), fall back to osmesa (software).
def _setup_offscreen_rendering():
    """Configure MuJoCo for offscreen rendering if no display is available."""
    if os.environ.get("DISPLAY"):
        return  # Display available, no need for offscreen
    if os.environ.get("MUJOCO_GL"):
        return  # Already configured by user

    for backend in ("egl", "osmesa"):
        os.environ["MUJOCO_GL"] = backend
        # Note: Do NOT attempt to import mujoco here to test; it locks the backend.
        # We just set the env var and trust gymnasium will use it later.
        if backend == "egl":
            os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
            os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"
        return


_setup_offscreen_rendering()

from metaworld_teleop.utils import (
    create_metaworld_env,
    list_available_tasks,
    get_env_info,
)
from metaworld_teleop.data_collector import DemoCollector

logger = logging.getLogger("teleop_server")


class TeleopState:
    """Shared mutable state for the teleoperation session."""

    # Fixed camera for data collection (not affected by user rotation)
    FIXED_CAM_AZIMUTH = 180.0
    FIXED_CAM_ELEVATION = -25.0
    FIXED_CAM_DISTANCE = 1.5
    FIXED_CAM_LOOKAT = [0.0, 0.6, 0.15]
    DEPTH_WIDTH = 128
    DEPTH_HEIGHT = 128
    DEPTH_CAMERA_NAME = "corner"

    def __init__(self):
        self.env = None
        self.task_name = "pick-place-v3"
        self.speed = 0.5
        self.seed = 42
        self.render_width = 640
        self.render_height = 480
        self.jpeg_quality = 60
        self.target_fps = 30
        self.debug = False

        # Action state (updated by WebSocket messages)
        self.keys_pressed = set()
        self.gripper_target = 0.0  # -1 open, +1 closed

        # Camera state (for user-controllable view, does NOT affect saved data)
        self.camera_azimuth = 180.0
        self.camera_elevation = -25.0
        self.camera_distance = 1.5
        self.camera_lookat = [0.0, 0.6, 0.15]

        # Control mode: 'camera' = IJKLUO rotate view, 'orientation' = rotate end-effector
        self.control_mode = "camera"
        # Orientation deltas (euler in radians) applied to mocap body each step
        self.orientation_delta = [0.0, 0.0, 0.0]  # [roll, pitch, yaw]

        # Episode tracking
        self.step_count = 0
        self.episode_count = 0
        self.last_reward = 0.0
        self.last_success = False
        self.obs = None

        # Data collection
        self.collector = None
        self.recording = False
        self.save_dir = "data/datasets/teleop"

        # Control
        self.running = False
        self.reset_flag = False
        self._pending_task = None  # Set by handle_client_message, consumed by main loop
        self._gamepad_action = None  # Set by joycon_input handler

    def create_env(self):
        """Create or recreate the Metaworld environment."""
        if self.env is not None:
            self.env.close()

        self.env = create_metaworld_env(
            task_name=self.task_name,
            render_mode="rgb_array",
            seed=self.seed,
        )
        env_info = get_env_info(self.env)
        self.collector = DemoCollector(
            save_dir=self.save_dir,
            obs_dim=env_info["obs_dim"],
            act_dim=env_info["action_dim"],
        )
        self.obs, _ = self.env.reset()
        self.step_count = 0
        self.episode_count = 1
        self.last_reward = 0.0
        self.last_success = False
        self.keys_pressed.clear()
        self.gripper_target = 0.0
        self.recording = False
        logger.info(f"Environment created: {self.task_name}")

    def compute_action(self):
        """Compute action from current input state (gamepad or keyboard)."""
        # Prefer gamepad input if available
        if self._gamepad_action is not None:
            action = self._gamepad_action
            self._gamepad_action = None  # Consume it
            return np.clip(
                action, self.env.action_space.low, self.env.action_space.high
            )

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

        action = np.array([dx, dy, dz, self.gripper_target], dtype=np.float32)
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)

    def apply_orientation(self):
        """Apply end-effector orientation via mocap body (experimental)."""
        if self.control_mode != "orientation":
            return
        dr, dp, dy = self.orientation_delta
        if abs(dr) < 1e-6 and abs(dp) < 1e-6 and abs(dy) < 1e-6:
            return
        try:
            inner_env = self.env.unwrapped
            if hasattr(inner_env, "data") and hasattr(inner_env.data, "mocap_quat"):
                from scipy.spatial.transform import Rotation

                # Get current quat and apply euler delta
                quat = inner_env.data.mocap_quat[0].copy()  # [w, x, y, z] (MuJoCo)
                # MuJoCo uses [w,x,y,z], scipy uses [x,y,z,w]
                r_cur = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
                r_delta = Rotation.from_euler("xyz", [dr, dp, dy])
                r_new = r_delta * r_cur
                q = r_new.as_quat()  # [x,y,z,w]
                inner_env.data.mocap_quat[0] = [
                    q[3],
                    q[0],
                    q[1],
                    q[2],
                ]  # back to [w,x,y,z]
                if self.debug:
                    logger.debug(
                        f"Orientation applied: delta=[{dr:.3f},{dp:.3f},{dy:.3f}]"
                    )
        except ImportError:
            logger.warning("scipy not installed; orientation control disabled")
        except Exception as e:
            if self.debug:
                logger.debug(f"Orientation error: {e}")
        finally:
            self.orientation_delta = [0.0, 0.0, 0.0]

    def _apply_camera_settings(self):
        """Apply user camera orientation before render (visual only)."""
        try:
            inner_env = self.env.unwrapped
            if hasattr(inner_env, "mujoco_renderer"):
                renderer = inner_env.mujoco_renderer
                if hasattr(renderer, "viewer") and renderer.viewer is not None:
                    viewer = renderer.viewer
                    if hasattr(viewer, "cam"):
                        viewer.cam.azimuth = self.camera_azimuth
                        viewer.cam.elevation = self.camera_elevation
                        viewer.cam.distance = self.camera_distance
                        viewer.cam.lookat[:] = self.camera_lookat
        except Exception:
            pass

    def _apply_fixed_camera(self):
        """Apply fixed camera for data collection."""
        try:
            inner_env = self.env.unwrapped
            if hasattr(inner_env, "mujoco_renderer"):
                renderer = inner_env.mujoco_renderer
                if hasattr(renderer, "viewer") and renderer.viewer is not None:
                    viewer = renderer.viewer
                    if hasattr(viewer, "cam"):
                        viewer.cam.azimuth = self.FIXED_CAM_AZIMUTH
                        viewer.cam.elevation = self.FIXED_CAM_ELEVATION
                        viewer.cam.distance = self.FIXED_CAM_DISTANCE
                        viewer.cam.lookat[:] = self.FIXED_CAM_LOOKAT
        except Exception:
            pass

    def render_jpeg_base64(self):
        """Render current frame as base64-encoded JPEG (user view)."""
        self._apply_camera_settings()
        frame = self.env.render()
        img = Image.fromarray(frame)
        if img.size != (self.render_width, self.render_height):
            img = img.resize((self.render_width, self.render_height), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def render_fixed_camera_frame(self):
        """Render frame from fixed data camera. Returns numpy array (H,W,3)."""
        try:
            import mujoco

            inner_env = self.env.unwrapped

            # Reuse gymnasium offscreen viewer and force fixed camera for this render.
            mujoco_renderer = inner_env.mujoco_renderer
            viewer = mujoco_renderer._get_viewer("rgb_array")

            prev_type = int(viewer.cam.type)
            prev_fixed_id = int(viewer.cam.fixedcamid)

            cam_id = inner_env.model.camera(self.DEPTH_CAMERA_NAME).id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cam_id

            frame = self.env.render()

            viewer.cam.type = prev_type
            viewer.cam.fixedcamid = prev_fixed_id
            return frame
        except Exception:
            # Fallback to legacy path if offscreen renderer is unavailable.
            self._apply_fixed_camera()
            frame = self.env.render()
            self._apply_camera_settings()  # restore user cam for next visual render
            return frame

    def _depth_camera_fovy(self):
        """Return FOV (degrees) for configured depth camera."""
        try:
            inner_env = self.env.unwrapped
            model = inner_env.model
            cam_id = model.camera(self.DEPTH_CAMERA_NAME).id
            return float(model.cam_fovy[cam_id])
        except Exception:
            return 45.0

    def render_depth(self):
        """Render depth image from fixed camera. Returns (H,W) float32 depth."""
        try:
            import mujoco

            inner_env = self.env.unwrapped
            model, data = inner_env.model, inner_env.data
            # Create offscreen context for depth
            renderer = mujoco.Renderer(model, self.DEPTH_HEIGHT, self.DEPTH_WIDTH)
            renderer.update_scene(data, camera=self.DEPTH_CAMERA_NAME)
            renderer.enable_depth_rendering()
            depth = renderer.render()
            renderer.disable_depth_rendering()
            renderer.close()
            return depth.astype(np.float32)
        except Exception as e:
            if self.debug:
                logger.debug(f"Depth render error: {e}")
            return None

    def depth_to_pointcloud(self, depth, fov=None, n_points=512):
        """Convert depth image to DP3-style point cloud.

        Pipeline (following the DemoGen/DP3 paper):
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
        if fov is None:
            fov = self._depth_camera_fovy()
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

        # Step 2: Workspace crop — auto-compute bounding box from median
        # In simulation, camera coords vary; use statistical crop
        median = np.median(points, axis=0)
        std = np.std(points, axis=0)
        margin = np.maximum(std * 3, 0.1)  # at least 10cm margin
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
                # Keep only points in valid clusters (label != -1)
                points = points[labels != -1]
        except ImportError:
            pass  # sklearn not available, skip clustering

        if len(points) == 0:
            return np.zeros((n_points, 3), dtype=np.float32)

        # Step 4: Farthest Point Sampling (FPS) to downsample
        points = self._farthest_point_sampling(points, n_points)
        return points.astype(np.float32)

    @staticmethod
    def _farthest_point_sampling(points, n_points):
        """Farthest Point Sampling (FPS) for point cloud downsampling.

        If len(points) <= n_points, pads with repeated last point.
        """
        n = len(points)
        if n <= n_points:
            # Pad by repeating points
            idx = np.arange(n)
            pad = np.random.choice(n, n_points - n, replace=True)
            idx = np.concatenate([idx, pad])
            return points[idx]

        # FPS algorithm
        selected = np.zeros(n_points, dtype=np.int64)
        selected[0] = np.random.randint(n)
        distances = np.full(n, np.inf)

        for i in range(1, n_points):
            last = points[selected[i - 1]]
            d = np.sum((points - last) ** 2, axis=1)
            distances = np.minimum(distances, d)
            selected[i] = np.argmax(distances)

        return points[selected]
