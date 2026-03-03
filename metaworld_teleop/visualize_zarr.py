#!/usr/bin/env python3
"""
Visualize zarr teleoperation data using Rerun.

Usage:
    python -m metaworld_teleop.visualize_zarr <path_to_zarr>
    # or via API: POST /api/visualize/{name}
"""

import argparse
import sys
from urllib.parse import quote
import numpy as np

try:
    import zarr
except ImportError:
    print("ERROR: zarr not installed. Run: pip install zarr")
    sys.exit(1)

try:
    import rerun as rr
except ImportError:
    rr = None


def visualize_zarr(
    zarr_path: str,
    web_port: int = 9090,
    show_scene: bool = False,
    show_robot: bool = False,
):
    """Load zarr dataset and visualize with Rerun web viewer."""
    if rr is None:
        print("ERROR: rerun-sdk not installed. Run: pip install rerun-sdk")
        return

    z = zarr.open(zarr_path, "r")

    obs = z["data/observations"][:]
    actions = z["data/actions"][:]
    rewards = z["data/rewards"][:]
    dones = z["data/dones"][:]
    episode_ends = (
        z["meta/episode_ends"][:] if "meta/episode_ends" in z else np.array([len(obs)])
    )

    n_steps = len(obs)
    n_episodes = len(episode_ends)

    has_scene_pc = "data/point_cloud" in z
    has_robot_pc = "data/point_cloud_robot" in z
    has_object_pc = "data/point_cloud_objects" in z

    image_key = None
    image_candidates = [
        "data/images",
        "data/image",
        "data/rgb",
        "data/rgb_images",
    ]
    for key in image_candidates:
        if key in z:
            image_key = key
            break

    object_names = []
    if "meta/object_names" in z:
        for name in z["meta/object_names"][:]:
            if isinstance(name, bytes):
                object_names.append(name.decode("utf-8"))
            else:
                object_names.append(str(name))
    elif has_object_pc:
        object_names = list(z["data/point_cloud_objects"].keys())

    # Keep only task-relevant object clouds by default.
    # Many datasets include robot and scene bodies in point_cloud_objects.
    robot_like_tokens = {
        "leftclaw",
        "rightclaw",
        "leftpad",
        "rightpad",
        "torso",
        "head",
        "screen",
        "controller_box",
        "robot",
        "gripper",
        "wrist",
        "arm",
        "hand",
    }
    scene_like_tokens = {
        "table",
        "pedestal",
        "pedestal_feet",
        "floor",
        "wall",
        "background",
    }

    all_object_names = list(object_names)
    filtered_object_names = []
    for name in object_names:
        low = name.lower()
        if any(tok in low for tok in robot_like_tokens):
            continue
        if any(tok in low for tok in scene_like_tokens):
            continue
        filtered_object_names.append(name)

    object_names = filtered_object_names

    def color_from_name(name):
        seed = sum(ord(ch) for ch in name)
        return [
            80 + (seed * 29) % 175,
            80 + (seed * 53) % 175,
            80 + (seed * 71) % 175,
        ]

    print(f"Dataset: {zarr_path}")
    print(f"  Steps:    {n_steps}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Obs dim:  {obs.shape[1]}")
    print(f"  Act dim:  {actions.shape[1]}")
    if has_object_pc:
        print(f"  Object clouds (raw): {len(all_object_names)}")
        print(f"  Object clouds (kept): {len(object_names)}")
    if image_key is None:
        print(
            "  [INFO] No RGB image stream found in zarr (expected one of: data/images, data/image, data/rgb, data/rgb_images)"
        )
    else:
        print(f"  RGB stream: {image_key}")

    # Diagnose potential planar collapse in saved point clouds.
    # This does not alter data; it only prints a warning for debugging.
    def _z_span(pts):
        if len(pts) == 0:
            return 0.0
        z_col = np.asarray(pts)[:, 2]
        return float(np.max(z_col) - np.min(z_col))

    if has_object_pc and len(object_names) > 0:
        sample_idx = 0
        planar_count = 0
        for obj_name in object_names:
            if obj_name not in z["data/point_cloud_objects"]:
                continue
            pts = z[f"data/point_cloud_objects/{obj_name}"][sample_idx]
            if _z_span(pts) < 1e-3:
                planar_count += 1
        if planar_count > 0:
            print(
                f"  [WARN] {planar_count}/{len(object_names)} object clouds have tiny z-span at step 0; check depth->pointcloud pipeline if this looks wrong."
            )

    # Initialize Rerun for headless web viewing (Rerun 0.30+ API)
    rr.init("metaworld_teleop_viz", spawn=False)

    # 1. Serve gRPC FIRST - this buffers the logged data
    grpc_port = web_port + 1
    grpc_url = rr.serve_grpc(grpc_port=grpc_port)
    print(f"  gRPC server: {grpc_url}")

    # 2. Serve Web Viewer and connect it to the gRPC server
    print(f"  Starting Rerun web viewer on port {web_port}...")
    rr.serve_web_viewer(open_browser=False, web_port=web_port, connect_to=grpc_url)
    viewer_url = f"http://127.0.0.1:{web_port}/?url={quote(grpc_url, safe='')}"
    print(f"  --> View at: {viewer_url} <--")
    rr.set_time("step", sequence=0)

    # Log metadata
    rr.log("metadata/dataset", rr.TextLog(f"Dataset: {zarr_path}"))
    rr.log(
        "metadata/stats",
        rr.TextLog(
            f"Steps: {n_steps} | Episodes: {n_episodes} | Obs: {obs.shape} | Act: {actions.shape}"
        ),
    )
    if image_key is None:
        rr.log(
            "camera/rgb_status",
            rr.TextLog(
                "No RGB images in this dataset. Recordings must save data/images (or data/image/data/rgb/data/rgb_images) to visualize camera frames."
            ),
        )

    # Log time-series data
    for step_idx in range(n_steps):
        rr.set_time("step", sequence=step_idx)

        # Find which episode this step belongs to
        episode_idx = np.searchsorted(episode_ends, step_idx, side="right")

        # Reward
        rr.log("metrics/reward", rr.Scalars(float(rewards[step_idx])))

        # Cumulative reward within episode
        ep_start = int(episode_ends[episode_idx - 1]) if episode_idx > 0 else 0
        cum_reward = float(np.sum(rewards[ep_start : step_idx + 1]))
        rr.log("metrics/cumulative_reward", rr.Scalars(cum_reward))

        # Actions (4D: dx, dy, dz, gripper)
        act = actions[step_idx]
        rr.log("actions/dx", rr.Scalars(float(act[0])))
        rr.log("actions/dy", rr.Scalars(float(act[1])))
        rr.log("actions/dz", rr.Scalars(float(act[2])))
        if len(act) > 3:
            rr.log("actions/gripper", rr.Scalars(float(act[3])))

        # End-effector position from observations
        # Metaworld obs: first 3 values are gripper xyz
        ee_pos = obs[step_idx][:3]
        rr.log(
            "robot/end_effector",
            rr.Points3D([ee_pos], radii=[0.02], colors=[[0, 200, 255]]),
        )

        # Object position (obs indices 18:21 typically for Metaworld)
        if obs.shape[1] >= 21:
            obj_pos = obs[step_idx][18:21]
            rr.log(
                "objects/target",
                rr.Points3D([obj_pos], radii=[0.015], colors=[[255, 100, 0]]),
            )

        # Episode boundary markers
        if step_idx in episode_ends:
            rr.log("events/episode_end", rr.TextLog(f"Episode {episode_idx} ended"))

        # Log images and point clouds if they exist in the zarr
        if image_key is not None and step_idx < z[image_key].shape[0]:
            img = z[image_key][step_idx]
            rr.log("camera/rgb", rr.Image(img))

        if show_scene and has_scene_pc:
            scene_pc = z["data/point_cloud"][step_idx]
            rr.log(
                "point_cloud/scene",
                rr.Points3D(scene_pc, radii=[0.002], colors=[[180, 180, 180]]),
            )

        if show_robot and has_robot_pc:
            robot_pc = z["data/point_cloud_robot"][step_idx]
            rr.log(
                "point_cloud/robot",
                rr.Points3D(robot_pc, radii=[0.0025], colors=[[0, 200, 255]]),
            )

        if has_object_pc:
            for obj_name in object_names:
                if obj_name not in z["data/point_cloud_objects"]:
                    continue
                obj_pc = z[f"data/point_cloud_objects/{obj_name}"][step_idx]
                rr.log(
                    f"point_cloud/objects/{obj_name}",
                    rr.Points3D(
                        obj_pc, radii=[0.0025], colors=[color_from_name(obj_name)]
                    ),
                )

        if "data/point_clouds" in z:
            pc = z["data/point_clouds"][step_idx]
            if len(pc) > 0:
                rr.log("camera/point_cloud", rr.Points3D(pc, radii=[0.002]))

    print(f"✓ Visualization complete. {n_steps} steps logged.")
    print(f"  View at: {viewer_url}")
    # Keep alive so web viewer stays accessible
    print("  Press Ctrl+C to stop.")
    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Visualize zarr teleop data with Rerun"
    )
    parser.add_argument("zarr_path", help="Path to zarr dataset")
    parser.add_argument(
        "--port", type=int, default=9090, help="Rerun web viewer port (default: 9090)"
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Save to .rrd file instead"
    )
    parser.add_argument(
        "--show-scene",
        action="store_true",
        help="Show scene/background point cloud (disabled by default)",
    )
    parser.add_argument(
        "--show-robot",
        action="store_true",
        help="Show robot point cloud (disabled by default)",
    )

    args = parser.parse_args()

    if args.save:
        if rr is None:
            print("ERROR: rerun-sdk not installed. Run: pip install rerun-sdk")
            return
        rr.init("metaworld_teleop_viz")
        rr.save(args.save)
        # Just log data without web viewer
        visualize_zarr(
            args.zarr_path,
            web_port=0,
            show_scene=args.show_scene,
            show_robot=args.show_robot,
        )
    else:
        visualize_zarr(
            args.zarr_path,
            web_port=args.port,
            show_scene=args.show_scene,
            show_robot=args.show_robot,
        )


if __name__ == "__main__":
    main()
