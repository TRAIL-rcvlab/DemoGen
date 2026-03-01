#!/usr/bin/env python3
"""
Render a Metaworld teleop zarr dataset into an MP4 video.

This replays recorded actions episode-by-episode in the corresponding task env,
renders RGB frames, and writes a video file.

Usage:
    python -m metaworld_teleop.render_zarr_video \
        data/datasets/teleop/teleop_assembly-v3_20260302_032249.zarr \
        --out data/videos/assembly_replay.mp4
"""

import argparse
import re
import os
from pathlib import Path

import numpy as np


def _setup_offscreen_rendering():
    """Configure MuJoCo offscreen backend for headless machines."""
    if os.environ.get("DISPLAY"):
        return
    if os.environ.get("MUJOCO_GL"):
        return

    # Prefer EGL on GPU servers; MuJoCo/Gym will initialize backend later.
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
    os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"


_setup_offscreen_rendering()


def infer_task_name(zarr_path: str) -> str:
    """Infer task from filename like teleop_<task>_YYYYMMDD_HHMMSS.zarr."""
    name = Path(zarr_path).name
    m = re.match(r"^teleop_(.+)_\d{8}_\d{6}\.zarr$", name)
    if not m:
        raise ValueError(
            f"Cannot infer task from filename: {name}. "
            "Expected format teleop_<task>_YYYYMMDD_HHMMSS.zarr"
        )
    return m.group(1)


def render_zarr_to_video(zarr_path: str, out_path: str, fps: int = 20, width: int = 640, height: int = 480):
    from metaworld_teleop.utils import create_metaworld_env

    try:
        import zarr
    except Exception as e:
        raise RuntimeError("zarr is required: pip install zarr==2.12.0") from e

    try:
        import imageio.v2 as imageio
    except Exception as e:
        raise RuntimeError("imageio is required: pip install imageio imageio-ffmpeg") from e

    from PIL import Image

    task_name = infer_task_name(zarr_path)
    z = zarr.open(zarr_path, "r")

    if "data/actions" not in z:
        raise ValueError(f"Missing data/actions in {zarr_path}")

    actions = z["data/actions"][:].astype(np.float32)
    episode_ends = z["meta/episode_ends"][:] if "meta/episode_ends" in z else np.array([len(actions)], dtype=np.int64)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    env = create_metaworld_env(task_name=task_name, render_mode="rgb_array", seed=42)
    writer = imageio.get_writer(str(out_file), fps=fps, codec="libx264")

    try:
        start = 0
        total_frames = 0
        for ep_idx, end in enumerate(episode_ends.tolist()):
            if end <= start:
                continue

            env.reset()

            # Optional initial frame per episode
            frame0 = env.render()
            if frame0 is not None:
                if frame0.shape[:2] != (height, width):
                    frame0 = np.array(Image.fromarray(frame0).resize((width, height), Image.LANCZOS))
                writer.append_data(frame0)
                total_frames += 1

            for i in range(start, end):
                env.step(actions[i])
                frame = env.render()
                if frame is None:
                    continue
                if frame.shape[:2] != (height, width):
                    frame = np.array(Image.fromarray(frame).resize((width, height), Image.LANCZOS))
                writer.append_data(frame)
                total_frames += 1

            print(f"Episode {ep_idx + 1}: steps={end - start}")
            start = end

        print(f"Saved video: {out_file}")
        print(f"Frames: {total_frames}, FPS: {fps}")
    finally:
        writer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Render Metaworld teleop zarr to MP4")
    parser.add_argument("zarr_path", help="Path to teleop zarr dataset")
    parser.add_argument("--out", required=True, help="Output mp4 path")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS (default: 20)")
    parser.add_argument("--width", type=int, default=640, help="Output width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Output height (default: 480)")
    args = parser.parse_args()

    render_zarr_to_video(
        zarr_path=args.zarr_path,
        out_path=args.out,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
