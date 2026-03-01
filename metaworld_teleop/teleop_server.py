#!/usr/bin/env python3
"""
Web Teleoperation Server (supports Metaworld and ManiSkill 3).

FastAPI + WebSocket backend for browser-based teleoperation.
Renders offscreen (MuJoCo/EGL for Metaworld, Vulkan/SAPIEN for ManiSkill)
and streams JPEG frames to the browser.

Usage:
    # Metaworld (default):
    python -m metaworld_teleop.teleop_server --port 9527 --task pick-place-v3

    # ManiSkill:
    python -m metaworld_teleop.teleop_server --simulator maniskill --port 9527 --task PickCube-v1

    # Then open http://server_ip:9527 in your browser
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# --- numpy 2.x compatibility ---
# np.product was removed in numpy 2.0; some transitive deps still use it.
if not hasattr(np, 'product'):
    np.product = np.prod

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    raise ImportError(
        "Web teleoperation requires: pip install fastapi uvicorn[standard] websockets"
    )

from metaworld_teleop.joycon_decoder import (
    JoyConState, parse_report, JOYCON_R_PRODUCT_ID, JOYCON_L_PRODUCT_ID,
)

# ---------------------------------------------------------------------------
# Simulator selection (set by CLI --simulator flag before app starts)
# ---------------------------------------------------------------------------
SIMULATOR = "metaworld"  # default; overwritten by main()


def _get_task_list():
    """Return the task list for the active simulator."""
    if SIMULATOR == "maniskill":
        from maniskill_teleop.utils import list_available_tasks
    else:
        from metaworld_teleop.utils import list_available_tasks
    return list_available_tasks()


def _create_teleop_state():
    """Create the correct TeleopState for the active simulator."""
    if SIMULATOR == "maniskill":
        from maniskill_teleop.teleop_env_manager import TeleopState
    else:
        from metaworld_teleop.teleop_env_manager import TeleopState
    return TeleopState()

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
        try:
            import mujoco
            # Quick test: if mujoco can be used, this backend works
            logger.info(f"Offscreen rendering backend: {backend}")
            return
        except Exception:
            pass

    # If neither works, unset and let it fail later with a clear error
    os.environ.pop("MUJOCO_GL", None)
    logger.warning("No offscreen rendering backend available (tried egl, osmesa)")

_setup_offscreen_rendering()

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="Web Teleoperation")

# Serve static files (frontend)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Shared state (single-user teleoperation)
# ---------------------------------------------------------------------------
# State is lazily initialized in main() via _create_teleop_state().
# At module level we create a placeholder that gets replaced.
# This avoids importing Metaworld/ManiSkill at module load time, which
# would fail if the wrong simulator's dependencies are installed.

state = None  # Replaced by main() → _create_teleop_state()
joycon_state = JoyConState()  # Stateful decoder for raw HID reports


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.head("/")
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main frontend page."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/debug", response_class=HTMLResponse)
async def joycon_debug():
    """Serve the JoyCon debug / button mapping page."""
    html_path = STATIC_DIR / "joycon_debug.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.head("/replay")
@app.get("/replay", response_class=HTMLResponse)
async def replay_page():
    """Serve the trajectory replay viewer page."""
    html_path = STATIC_DIR / "replay.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/tasks")
async def api_tasks():
    """List available tasks for the active simulator."""
    tasks = _get_task_list()
    task_labels = {}
    try:
        if SIMULATOR == "maniskill":
            from maniskill_teleop.utils import get_task_label
        else:
            from metaworld_teleop.utils import get_task_label
        task_labels = {t: get_task_label(t) for t in tasks}
    except Exception:
        task_labels = {}
    return JSONResponse(
        content={
            "tasks": tasks,
            "task_labels": task_labels,
            "current_task": state.task_name,
            "simulator": SIMULATOR,
        }
    )


@app.get("/api/status")
async def api_status():
    """Get current teleoperation status."""
    return JSONResponse(content={
        "simulator": SIMULATOR,
        "task": state.task_name,
        "episode": state.episode_count,
        "step": state.step_count,
        "recording": state.recording,
        "episodes_recorded": state.collector.num_episodes if state.collector else 0,
        "total_steps_recorded": state.collector.total_steps if state.collector else 0,
    })


# ---------------------------------------------------------------------------
# Demos API (ManiSkill trajectory replay)
# ---------------------------------------------------------------------------

@app.get("/api/available_tasks")
async def api_available_tasks():
    """List all ManiSkill tasks with local demo availability.

    Returns each task with a flag indicating whether demos are available locally.
    Used by the replay UI to show downloadable tasks.
    """
    if SIMULATOR != "maniskill":
        return JSONResponse(content={"tasks": [], "error": "Only available for ManiSkill simulator"})
    try:
        from maniskill_teleop.replay import list_local_demos
        from maniskill_teleop.utils import get_task_label, list_official_downloadable_tasks
        all_tasks = _get_task_list()
        demos = list_local_demos()
        downloadable_tasks = set(list_official_downloadable_tasks())
        # Build set of env_ids that have local demos
        local_envs = {}
        for d in demos:
            eid = d["env_id"]
            if eid not in local_envs:
                local_envs[eid] = {"num_files": 0, "total_episodes": 0}
            local_envs[eid]["num_files"] += 1
            local_envs[eid]["total_episodes"] += d.get("num_episodes", 0)

        result = []
        for task in all_tasks:
            entry = {
                "env_id": task,
                "label": get_task_label(task),
                "has_demos": task in local_envs,
                "downloadable": task in downloadable_tasks,
                "num_files": local_envs[task]["num_files"] if task in local_envs else 0,
                "total_episodes": local_envs[task]["total_episodes"] if task in local_envs else 0,
            }
            result.append(entry)
        return JSONResponse(content={"tasks": result})
    except Exception as e:
        return JSONResponse(content={"tasks": [], "error": str(e)}, status_code=500)


@app.get("/api/demos")
async def api_demos():
    """List locally available ManiSkill demo datasets."""
    if SIMULATOR != "maniskill":
        return JSONResponse(content={"demos": [], "error": "Only available for ManiSkill simulator"})
    try:
        from maniskill_teleop.replay import list_local_demos
        demos = list_local_demos()
        return JSONResponse(content={"demos": demos})
    except Exception as e:
        return JSONResponse(content={"demos": [], "error": str(e)}, status_code=500)


@app.get("/api/demos/{env_id}")
async def api_demos_for_env(env_id: str):
    """List demos for a specific environment."""
    if SIMULATOR != "maniskill":
        return JSONResponse(content={"demos": [], "error": "Only available for ManiSkill simulator"})
    try:
        from maniskill_teleop.replay import list_local_demos
        demos = list_local_demos(env_id=env_id)
        return JSONResponse(content={"demos": demos, "env_id": env_id})
    except Exception as e:
        return JSONResponse(content={"demos": [], "error": str(e)}, status_code=500)


@app.post("/api/demos/download/{env_id}")
async def api_download_demos(env_id: str):
    """Download official ManiSkill demos for an environment (blocking)."""
    if SIMULATOR != "maniskill":
        return JSONResponse(content={"error": "Only available for ManiSkill simulator"}, status_code=400)
    try:
        from maniskill_teleop.replay import download_demos
        from maniskill_teleop.utils import list_official_downloadable_tasks

        downloadable_tasks = set(list_official_downloadable_tasks())
        if env_id not in downloadable_tasks:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": f"{env_id} 不在 ManiSkill 官方可下载演示列表中",
                    "env_id": env_id,
                },
                status_code=400,
            )

        success = download_demos(env_id)
        if success:
            return JSONResponse(content={"status": "ok", "env_id": env_id})
        else:
            return JSONResponse(content={"status": "error", "message": "Download failed"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/demos/replay_hint")
async def api_replay_hint():
    """Return CLI command hint for heavy trajectory replay (conversion).

    Full replay (control mode conversion) is CPU-intensive and best done via CLI.
    For visual replay (streaming frames), use the /ws/replay WebSocket.
    """
    return JSONResponse(content={
        "hint": "Use CLI for trajectory conversion: python -m maniskill_teleop.replay replay --help",
        "visual_replay": "Connect to /ws/replay WebSocket for frame-by-frame visual replay",
    })


@app.get("/api/demos/trajectory_info")
async def api_trajectory_info(traj_path: str):
    """Get info about a trajectory file."""
    if SIMULATOR != "maniskill":
        return JSONResponse(content={"error": "Only available for ManiSkill simulator"}, status_code=400)
    try:
        from maniskill_teleop.replay import load_trajectory_metadata
        import h5py
        meta = load_trajectory_metadata(traj_path)
        info = {"traj_path": traj_path}
        if meta:
            episodes = meta.get("episodes", [])
            env_info = meta.get("env_info", {})
            env_kwargs = env_info.get("env_kwargs", {})
            info["env_id"] = env_info.get("env_id") or meta.get("env_id", "unknown")
            info["num_episodes"] = len(episodes)
            info["control_mode"] = env_kwargs.get("control_mode", "unknown")
            info["sim_backend"] = env_kwargs.get("sim_backend", "unknown")
            if episodes:
                # Per-episode control_mode overrides env-level
                ep0_cm = episodes[0].get("control_mode")
                if ep0_cm:
                    info["control_mode"] = ep0_cm
                steps = [e.get("elapsed_steps", 0) for e in episodes]
                info["step_stats"] = {
                    "min": int(min(steps)),
                    "max": int(max(steps)),
                    "mean": float(np.mean(steps)),
                }
                info["success_rate"] = sum(1 for e in episodes if e.get("success", False)) / len(episodes)

        # H5 structure
        with h5py.File(traj_path, "r") as f:
            traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
            info["h5_episodes"] = len(traj_keys)
            if traj_keys:
                g = f[traj_keys[0]]
                if "actions" in g:
                    info["action_shape"] = list(g["actions"].shape)

        return JSONResponse(content=info)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/demos/export_dp3")
async def api_export_dp3(payload: dict):
    """Replay one episode in current env and export DP3-format zarr."""
    if SIMULATOR != "maniskill":
        return JSONResponse(content={"error": "Only available for ManiSkill simulator"}, status_code=400)
    try:
        from maniskill_teleop.replay import export_episode_to_dp3_zarr

        traj_path = payload.get("traj_path")
        if not traj_path:
            return JSONResponse(content={"error": "traj_path is required"}, status_code=400)

        episode_id = int(payload.get("episode_id", 0))
        control_mode = payload.get("control_mode")
        n_points = int(payload.get("n_points", 512))
        out_dir = payload.get("out_dir", state.save_dir)

        result = export_episode_to_dp3_zarr(
            traj_path=traj_path,
            episode_id=episode_id,
            control_mode=control_mode,
            out_dir=out_dir,
            n_points=n_points,
        )
        return JSONResponse(content={"status": "ok", **result})
    except Exception as e:
        return JSONResponse(content={"status": "error", "error": str(e)}, status_code=500)


@app.get("/api/datasets")
async def api_datasets():
    """List saved zarr datasets."""
    import glob
    datasets = []
    pattern = os.path.join(state.save_dir, "*.zarr")
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        try:
            import zarr
            z = zarr.open(path, "r")
            n_steps = z["data/observations"].shape[0] if "data/observations" in z else 0
            n_episodes = len(z["meta/episode_ends"]) if "meta/episode_ends" in z else 0
            datasets.append({
                "name": name,
                "steps": n_steps,
                "episodes": n_episodes,
                "path": path,
            })
        except Exception:
            datasets.append({"name": name, "steps": 0, "episodes": 0, "path": path})
    return JSONResponse(content={"datasets": datasets})


@app.get("/api/datasets/{name}")
async def api_dataset_detail(name: str):
    """Get details of a specific zarr dataset."""
    import zarr
    path = os.path.join(state.save_dir, name)
    if not os.path.exists(path):
        return JSONResponse(content={"error": "Not found"}, status_code=404)
    try:
        z = zarr.open(path, "r")
        obs = z["data/observations"]
        actions = z["data/actions"]
        rewards = z["data/rewards"]
        episode_ends = z["meta/episode_ends"][:].tolist() if "meta/episode_ends" in z else []
        reward_list = z["data/rewards"][:].tolist()
        return JSONResponse(content={
            "name": name,
            "obs_shape": list(obs.shape),
            "action_shape": list(actions.shape),
            "total_steps": obs.shape[0],
            "episodes": len(episode_ends),
            "episode_ends": episode_ends,
            "rewards": reward_list,
            "reward_sum": float(rewards[:].sum()),
            "reward_mean": float(rewards[:].mean()),
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)





# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Main WebSocket for bidirectional communication.

    Receives: key_down, key_up, mouse, reset, set_task, toggle_record, save, set_speed
    Sends: frame (base64 JPEG + state info)
    """
    await ws.accept()
    logger.info("WebSocket client connected")

    # Create environment if not exists, or reset if stale
    if state.env is None:
        state.create_env()
    else:
        # Always reset on new connection to avoid truncated-state errors
        state.obs, _ = state.env.reset()
        state.step_count = 0
        state.keys_pressed.clear()
        state.gripper_target = 0.0

    state.running = True

    # Task for reading client messages
    async def read_messages():
        try:
            while state.running:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                handle_client_message(msg)
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
            state.running = False
        except Exception as e:
            logger.error(f"WebSocket read error: {e}")
            state.running = False

    # Start message reader in background
    reader_task = asyncio.create_task(read_messages())

    frame_interval = 1.0 / state.target_fps
    fps_counter = 0
    fps_time = time.time()
    current_fps = 0.0

    try:
        while state.running:
            loop_start = time.time()

            # Handle pending task switch (set by handle_client_message)
            if state._pending_task is not None:
                new_task = state._pending_task
                state._pending_task = None
                try:
                    if state.recording and state.collector:
                        state.collector.end_episode()
                    state.task_name = new_task
                    state.create_env()
                    logger.info(f"Task switched to: {new_task}")
                except Exception as e:
                    logger.error(f"Failed to switch task to {new_task}: {e}")

            # Handle manual reset
            if state.reset_flag:
                state.reset_flag = False
                if state.recording and state.collector:
                    state.collector.end_episode()
                state.obs, _ = state.env.reset()
                state.step_count = 0
                state.episode_count += 1
                state.keys_pressed.clear()
                state.gripper_target = 0.0
                joycon_state.reset()
                logger.info(f"Manual reset → Episode {state.episode_count}")

            # Apply end-effector orientation (experimental)
            state.apply_orientation()

            # Compute and apply action
            action = state.compute_action()
            next_obs, reward, terminated, truncated, info = state.env.step(action)

            # Record if active (use fixed camera for data)
            if state.recording and state.collector:
                state.collector.step(state.obs, action, reward, terminated)

            state.obs = next_obs
            state.step_count += 1
            state.last_reward = reward
            state.last_success = info.get("success", False)

            # Render frame (user-controllable camera)
            frame_b64 = state.render_jpeg_base64()

            # FPS calculation
            fps_counter += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_time = time.time()

            # Send frame + state to client
            await ws.send_text(json.dumps({
                "type": "frame",
                "image": frame_b64,
                "step": state.step_count,
                "reward": float(round(reward, 4)),
                "total_reward": float(round(state.last_reward, 4)),
                "episode": state.episode_count,
                "recording": state.recording,
                "success": state.last_success,
                "fps": round(current_fps, 1),
                "task": state.task_name,
                "episodes_recorded": state.collector.num_episodes if state.collector else 0,
                "gripper": "closed" if state.gripper_target > 0 else "open",
                "control_mode": state.control_mode,
            }))

            # Auto-reset on TERMINATED only (task success/failure)
            # TimeLimit wrapper is stripped, so truncated should not occur
            if terminated:
                reason = "success" if state.last_success else "terminated"
                logger.info(f"Episode {state.episode_count} ended: {reason} at step {state.step_count}")
                if state.recording and state.collector:
                    state.collector.end_episode()
                state.obs, _ = state.env.reset()
                state.step_count = 0
                state.episode_count += 1
            elif truncated:
                # Safety fallback: if truncation somehow occurs, reset silently
                state.obs, _ = state.env.reset()
                logger.warning(f"Unexpected truncation at step {state.step_count}, auto-reset")
                state.step_count = 0
                state.episode_count += 1

            # Frame rate control
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_interval - elapsed)
            await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        logger.info("Client disconnected during stream")
    except Exception as e:
        logger.error(f"Stream error: {e}")
    finally:
        state.running = False
        reader_task.cancel()
        logger.info("WebSocket session ended")


# ---------------------------------------------------------------------------
# WebSocket Replay Endpoint (ManiSkill only)
# ---------------------------------------------------------------------------

@app.websocket("/ws/replay")
async def websocket_replay_endpoint(ws: WebSocket):
    """
    WebSocket endpoint for streaming trajectory replay.

    Client sends: { "traj_path": "...", "episode_id": 0, "speed": 1.0, "control_mode": "pd_ee_delta_pos" }
    Server streams: { "type": "replay_frame", "image": base64, "step": N, "total_steps": M, ... }
    Also supports: { "type": "pause" }, { "type": "resume" }, { "type": "seek", "step": N }
    """
    await ws.accept()
    logger.info("Replay WebSocket client connected")

    if SIMULATOR != "maniskill":
        await ws.send_text(json.dumps({"type": "error", "message": "Replay only available for ManiSkill"}))
        await ws.close()
        return

    paused = False
    replay_speed = 1.0
    replay_active = False

    try:
        while True:
            # Wait for a replay request
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "start_replay":
                traj_path = msg.get("traj_path", "")
                episode_id = int(msg.get("episode_id", 0))
                replay_speed = float(msg.get("speed", 1.0))
                control_mode = msg.get("control_mode")  # None = auto-detect from metadata
                render_width = int(msg.get("width", 640))
                render_height = int(msg.get("height", 480))

                if not traj_path or not os.path.exists(traj_path):
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": f"Trajectory file not found: {traj_path}",
                    }))
                    continue

                replay_active = True
                paused = False

                try:
                    from maniskill_teleop.replay import visual_replay_generator

                    await ws.send_text(json.dumps({"type": "replay_started", "traj_path": traj_path, "episode_id": episode_id}))

                    generator = visual_replay_generator(
                        traj_path=traj_path,
                        episode_id=episode_id,
                        control_mode=control_mode,
                        render_width=render_width,
                        render_height=render_height,
                    )

                    frame_interval = (1.0 / 30) / max(0.1, replay_speed)

                    for frame_data in generator:
                        # Check for control messages (non-blocking)
                        try:
                            ctrl_raw = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                            ctrl_msg = json.loads(ctrl_raw)
                            ctrl_type = ctrl_msg.get("type", "")
                            if ctrl_type == "pause":
                                paused = True
                            elif ctrl_type == "resume":
                                paused = False
                            elif ctrl_type == "stop":
                                replay_active = False
                                break
                            elif ctrl_type == "set_speed":
                                replay_speed = float(ctrl_msg.get("speed", 1.0))
                                frame_interval = (1.0 / 30) / max(0.1, replay_speed)
                        except asyncio.TimeoutError:
                            pass

                        # Wait while paused
                        while paused and replay_active:
                            try:
                                ctrl_raw = await asyncio.wait_for(ws.receive_text(), timeout=0.1)
                                ctrl_msg = json.loads(ctrl_raw)
                                if ctrl_msg.get("type") == "resume":
                                    paused = False
                                elif ctrl_msg.get("type") == "stop":
                                    replay_active = False
                            except asyncio.TimeoutError:
                                pass

                        if not replay_active:
                            break

                        # Encode and send frame
                        frame = frame_data["frame"]
                        img = Image.fromarray(frame)
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=80)
                        frame_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

                        await ws.send_text(json.dumps({
                            "type": "replay_frame",
                            "image": frame_b64,
                            "step": frame_data["step"],
                            "total_steps": frame_data["total_steps"],
                            "reward": frame_data["reward"],
                            "success": frame_data["success"],
                        }))

                        await asyncio.sleep(frame_interval)

                    await ws.send_text(json.dumps({"type": "replay_ended"}))

                except Exception as e:
                    logger.error(f"Replay error: {e}")
                    await ws.send_text(json.dumps({"type": "error", "message": str(e)}))

                replay_active = False

            elif msg_type == "pause":
                paused = True
            elif msg_type == "resume":
                paused = False
            elif msg_type == "set_speed":
                replay_speed = float(msg.get("speed", 1.0))

    except WebSocketDisconnect:
        logger.info("Replay WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Replay WebSocket error: {e}")
    finally:
        logger.info("Replay WebSocket session ended")


def _switch_task(direction: int):
    """Switch to next (+1) or previous (-1) task in the task list.

    Wraps around at the boundaries. Resets the environment after switching.
    """
    tasks = _get_task_list()
    if not tasks:
        return
    try:
        current_idx = tasks.index(state.task_name)
    except ValueError:
        current_idx = 0
    new_idx = (current_idx + direction) % len(tasks)
    new_task = tasks[new_idx]
    if new_task != state.task_name:
        state._pending_task = new_task
        logger.info(f"Task switch requested via JoyCon: → {new_task} (index {new_idx}/{len(tasks)})")


def handle_client_message(msg: dict):
    """Process a message from the frontend client."""
    msg_type = msg.get("type", "")

    if msg_type == "key_down":
        key = msg.get("key", "").lower()
        if key in ("w", "a", "s", "d", "q", "e"):
            state.keys_pressed.add(key)
        elif key in ("space", " "):
            # Keyboard gripper control: hold SPACE to close
            state.gripper_target = 1.0

    elif msg_type == "key_up":
        key = msg.get("key", "").lower()
        if key in ("w", "a", "s", "d", "q", "e"):
            state.keys_pressed.discard(key)
        elif key in ("space", " "):
            # Release SPACE to open
            state.gripper_target = -1.0

    elif msg_type == "mouse":
        button = msg.get("button", "")
        pressed = msg.get("pressed", False)
        if pressed:
            if button == "left":
                state.gripper_target = 1.0
            elif button == "right":
                state.gripper_target = -1.0

    elif msg_type == "reset":
        state.reset_flag = True

    elif msg_type == "set_task":
        new_task = msg.get("task", state.task_name)
        if new_task != state.task_name:
            # Don't call create_env() here — it races with the main loop.
            # Set a pending task flag; the main loop will handle the switch safely.
            state._pending_task = new_task
            logger.info(f"Task switch requested: {state.task_name} → {new_task}")

    elif msg_type == "camera_rotate":
        state.camera_azimuth += float(msg.get("dx", 0)) * 0.5
        state.camera_elevation = max(-90, min(0, state.camera_elevation + float(msg.get("dy", 0)) * 0.5))

    elif msg_type == "camera_zoom":
        delta = float(msg.get("delta", 0))
        state.camera_distance = max(0.3, min(5.0, state.camera_distance + delta * 0.1))

    elif msg_type == "toggle_record":
        state.recording = not state.recording
        logger.info(f"Recording: {'ON' if state.recording else 'OFF'}")

    elif msg_type == "save":
        if state.collector:
            if state.collector.current_episode_steps > 0:
                state.collector.end_episode()
            path = state.collector.save(task_name=state.task_name)
            logger.info(f"Data saved to: {path}")

    elif msg_type == "set_speed":
        state.speed = float(msg.get("speed", 0.1))
        logger.info(f"Speed set to: {state.speed}")

    elif msg_type == "teleop_cmd":
        # Gamepad/JoyCon explicit mapping input sent from client
        dx = float(msg.get("dx", 0)) * state.speed
        dy = float(msg.get("dy", 0)) * state.speed
        dz = float(msg.get("dz", 0)) * state.speed
        
        # Deadzone filter
        if abs(dx) < 0.08 * state.speed: dx = 0.0
        if abs(dy) < 0.08 * state.speed: dy = 0.0
        if abs(dz) < 0.08 * state.speed: dz = 0.0

        cam = float(msg.get("cam", 0))
        if abs(cam) > 0.1:
            state.camera_azimuth += cam * 2.0

        droll = float(msg.get("droll", 0)) * state.speed
        dpitch = float(msg.get("dpitch", 0)) * state.speed
        dyaw = float(msg.get("dyaw", 0)) * state.speed

        if abs(droll) > 0.005 or abs(dpitch) > 0.005 or abs(dyaw) > 0.005:
            # Add to orientation delta directly (applied to mocap in apply_orientation)
            state.orientation_delta[0] += droll * 0.5
            state.orientation_delta[1] += dpitch * 0.5
            state.orientation_delta[2] += dyaw * 0.5

        buttons = msg.get("buttons", {})
        if buttons.get("gripper_close"):
            state.gripper_target = 1.0
        elif buttons.get("gripper_open"):
            state.gripper_target = -1.0
        if buttons.get("reset"):
            state.reset_flag = True
        if buttons.get("record_start"):
            state.recording = True
        elif buttons.get("record_stop"):
            state.recording = False
            
        state._gamepad_action = np.array([dx, dy, dz, state.gripper_target], dtype=np.float32)
        if state.debug and (abs(dx) > 0.001 or abs(dy) > 0.001 or abs(dz) > 0.001):
            logger.info(f"teleop_cmd action: dx={dx:.4f} dy={dy:.4f} dz={dz:.4f} grip={state.gripper_target}")

    elif msg_type == "joycon_input":
        # Legacy fallback
        pass

    elif msg_type == "joycon_connected":
        # Browser notifies which JoyCon was paired via WebHID
        pid = msg.get("product_id", JOYCON_R_PRODUCT_ID)
        joycon_state.is_right = (pid == JOYCON_R_PRODUCT_ID)
        pname = msg.get("product_name", "unknown")
        side = "Right" if joycon_state.is_right else "Left"
        logger.info(f"JoyCon connected via WebHID: {pname} (pid=0x{pid:04X}, {side})")

    elif msg_type == "joycon_raw":
        # Raw 0x30 HID report forwarded from browser WebHID
        raw_b64 = msg.get("data", "")
        try:
            raw_bytes = base64.b64decode(raw_b64)
        except Exception:
            logger.warning("joycon_raw: invalid base64 data")
            return

        report = parse_report(raw_bytes)
        result = joycon_state.process_report(report)

        # Map decoded result → TeleopState action
        dx = result["dx"] * state.speed
        dy = result["dy"] * state.speed
        dz = result["dz"] * state.speed

        # Deadzone
        if abs(dx) < 0.02 * state.speed:
            dx = 0.0
        if abs(dy) < 0.02 * state.speed:
            dy = 0.0
        if abs(dz) < 0.02 * state.speed:
            dz = 0.0

        state.gripper_target = result["gripper"]
        state._gamepad_action = np.array([dx, dy, dz, state.gripper_target], dtype=np.float32)

        # Orientation delta from IMU → accumulate for next apply_orientation() call
        ori = result["orientation_delta"]
        if abs(ori[0]) > 0.001 or abs(ori[1]) > 0.001 or abs(ori[2]) > 0.001:
            state.orientation_delta[0] += ori[0]
            state.orientation_delta[1] += ori[1]
            state.orientation_delta[2] += ori[2]

        # Button actions
        if result["reset"]:
            state.reset_flag = True

        # Recording control: A = start, B = stop
        if result["button_control"] == 1:
            # A → start recording
            if not state.recording:
                state.recording = True
                logger.info("Recording started (JoyCon A)")
        elif result["button_control"] == -1:
            # B → stop recording
            if state.recording:
                state.recording = False
                logger.info("Recording stopped (JoyCon B)")

        # Scene/task switching: X = next, Y = previous
        if result["scene_control"] != 0:
            _switch_task(result["scene_control"])

        if state.debug and (abs(dx) > 0.001 or abs(dy) > 0.001 or abs(dz) > 0.001):
            logger.info(f"joycon_raw → dx={dx:.4f} dy={dy:.4f} dz={dz:.4f} grip={state.gripper_target}")

    elif msg_type == "set_control_mode":
        mode = msg.get("mode", "camera")
        if mode in ("camera", "orientation"):
            state.control_mode = mode
            logger.info(f"Control mode: {mode}")

    elif msg_type == "orientation_input":
        # IJKLUO in orientation mode → euler deltas
        rate = 0.05  # radians per message
        dr = float(msg.get("roll", 0)) * rate
        dp = float(msg.get("pitch", 0)) * rate
        dy = float(msg.get("yaw", 0)) * rate
        state.orientation_delta = [dr, dp, dy]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    global state, SIMULATOR

    parser = argparse.ArgumentParser(description="Web Teleoperation Server (Metaworld / ManiSkill)")
    parser.add_argument("--simulator", type=str, default="metaworld",
                        choices=["metaworld", "maniskill"],
                        help="Simulation backend (default: metaworld)")
    parser.add_argument("--port", type=int, default=9527, help="Server port (default: 9527)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--task", type=str, default=None,
                        help="Initial task (default: pick-place-v3 for metaworld, PickCube-v1 for maniskill)")
    parser.add_argument("--speed", type=float, default=0.1, help="Movement speed (default: 0.1)")
    parser.add_argument("--save-dir", type=str, default="data/datasets/teleop", help="Save directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--width", type=int, default=640, help="Render width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Render height (default: 480)")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality 1-100 (default: 80)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (gamepad, orientation)")
    # ManiSkill-specific options
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pos",
                        choices=["pd_ee_delta_pos", "pd_ee_delta_pose"],
                        help="ManiSkill control mode: 4-DOF (pos) or 7-DOF (pose)")
    parser.add_argument("--obs-mode", type=str, default="state",
                        choices=["state", "rgbd", "pointcloud"],
                        help="ManiSkill observation mode")

    args = parser.parse_args()

    # Set simulator globally before creating state
    SIMULATOR = args.simulator
    state = _create_teleop_state()

    # Default task per simulator
    default_task = "pick-place-v3" if SIMULATOR == "metaworld" else "PickCube-v1"

    # Configure state
    state.task_name = args.task if args.task else default_task
    state.speed = args.speed
    state.save_dir = args.save_dir
    state.seed = args.seed
    state.target_fps = args.fps
    state.render_width = args.width
    state.render_height = args.height
    state.jpeg_quality = args.jpeg_quality
    state.debug = args.debug

    # ManiSkill-specific config
    if SIMULATOR == "maniskill":
        state.control_mode = args.control_mode
        state.obs_mode = args.obs_mode

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(name)s] %(message)s")

    logger.info(f"Simulator: {SIMULATOR}")
    logger.info(f"Starting web teleoperation server on {args.host}:{args.port}")
    logger.info(f"Task: {state.task_name} | Speed: {args.speed} | FPS: {args.fps} | Debug: {args.debug}")
    if SIMULATOR == "maniskill":
        logger.info(f"Control mode: {args.control_mode} | Obs mode: {args.obs_mode}")
    logger.info(f"Open http://<server_ip>:{args.port} in your browser to start teleoperation")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
