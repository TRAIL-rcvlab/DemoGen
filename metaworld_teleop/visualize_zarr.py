#!/usr/bin/env python3
"""
Visualize zarr teleoperation data using Rerun.

Usage:
    python -m metaworld_teleop.visualize_zarr <path_to_zarr>
    # or via API: POST /api/visualize/{name}
"""

import argparse
import sys
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


def visualize_zarr(zarr_path: str, web_port: int = 9090):
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

    object_names = []
    if "meta/object_names" in z:
        for name in z["meta/object_names"][:]:
            if isinstance(name, bytes):
                object_names.append(name.decode("utf-8"))
            else:
                object_names.append(str(name))
    elif has_object_pc:
        object_names = list(z["data/point_cloud_objects"].keys())

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

    # Initialize Rerun for headless web viewing (Rerun 0.30 API)
    rr.init("metaworld_teleop_viz", spawn=False)

    # Start web viewer on port 9090
    print(f"  Starting Rerun web viewer on port {web_port}...")
    rr.serve_web_viewer(open_browser=False, web_port=web_port)

    # Serve data via gRPC
    grpc_port = web_port + 1
    server_uri = rr.serve_grpc(grpc_port=grpc_port)
    print(f"  Serving data at {server_uri}")
    print(f"  --> View at: http://0.0.0.0:{web_port} <--")

    rr.set_time("step", sequence=0)

    # Log metadata
    rr.log("metadata/dataset", rr.TextLog(f"Dataset: {zarr_path}"))
    rr.log(
        "metadata/stats",
        rr.TextLog(
            f"Steps: {n_steps} | Episodes: {n_episodes} | Obs: {obs.shape} | Act: {actions.shape}"
        ),
    )

    # Log time-series data
    for step_idx in range(n_steps):
        rr.set_time("step", sequence=step_idx)

        # Find which episode this step belongs to
        episode_idx = np.searchsorted(episode_ends, step_idx, side="right")

        # Reward
        rr.log("metrics/reward", rr.Scalar(float(rewards[step_idx])))

        # Cumulative reward within episode
        ep_start = int(episode_ends[episode_idx - 1]) if episode_idx > 0 else 0
        cum_reward = float(np.sum(rewards[ep_start : step_idx + 1]))
        rr.log("metrics/cumulative_reward", rr.Scalar(cum_reward))

        # Actions (4D: dx, dy, dz, gripper)
        act = actions[step_idx]
        rr.log("actions/dx", rr.Scalar(float(act[0])))
        rr.log("actions/dy", rr.Scalar(float(act[1])))
        rr.log("actions/dz", rr.Scalar(float(act[2])))
        if len(act) > 3:
            rr.log("actions/gripper", rr.Scalar(float(act[3])))

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
        if "data/images" in z and step_idx < z["data/images"].shape[0]:
            img = z["data/images"][step_idx]
            rr.log("camera/rgb", rr.Image(img))

        if has_scene_pc:
            scene_pc = z["data/point_cloud"][step_idx]
            rr.log(
                "point_cloud/scene",
                rr.Points3D(scene_pc, radii=[0.002], colors=[[180, 180, 180]]),
            )

        if has_robot_pc:
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
    print(f"  View at: http://0.0.0.0:{web_port}")
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

    args = parser.parse_args()

    if args.save:
        if rr is None:
            print("ERROR: rerun-sdk not installed. Run: pip install rerun-sdk")
            return
        rr.init("metaworld_teleop_viz")
        rr.save(args.save)
        # Just log data without web viewer
        visualize_zarr(args.zarr_path, web_port=0)
    else:
        visualize_zarr(args.zarr_path, web_port=args.port)


if __name__ == "__main__":
    main()
