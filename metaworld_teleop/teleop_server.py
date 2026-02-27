#!/usr/bin/env python3
"""
Metaworld Web Teleoperation Server.

FastAPI + WebSocket backend for browser-based teleoperation of Metaworld environments.
Renders MuJoCo offscreen and streams JPEG frames to the browser.

Usage:
    python -m metaworld_teleop.teleop_server --port 9527 --task pick-place-v3
    # Then open http://server_ip:9527 in your browser
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    raise ImportError(
        "Web teleoperation requires: pip install fastapi uvicorn[standard] websockets"
    )

from metaworld_teleop.utils import create_metaworld_env, list_available_tasks, get_env_info
from metaworld_teleop.data_collector import DemoCollector

logger = logging.getLogger("teleop_server")

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="Metaworld Web Teleoperation")

# Serve static files (frontend)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Shared state (single-user teleoperation)
# ---------------------------------------------------------------------------

class TeleopState:
    """Shared mutable state for the teleoperation session."""

    def __init__(self):
        self.env = None
        self.task_name = "pick-place-v3"
        self.speed = 0.1
        self.seed = 42
        self.render_width = 640
        self.render_height = 480
        self.jpeg_quality = 80
        self.target_fps = 30

        # Action state (updated by WebSocket messages)
        self.keys_pressed = set()
        self.gripper_target = 0.0  # -1 open, +1 closed

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
        """Compute action from current key state."""
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

    def render_jpeg_base64(self):
        """Render current frame as base64-encoded JPEG."""
        frame = self.env.render()
        img = Image.fromarray(frame)
        # Resize if needed
        if img.size != (self.render_width, self.render_height):
            img = img.resize((self.render_width, self.render_height), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")


state = TeleopState()


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main frontend page."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/tasks")
async def api_tasks():
    """List available Metaworld tasks."""
    return JSONResponse(content={"tasks": list_available_tasks()})


@app.get("/api/status")
async def api_status():
    """Get current teleoperation status."""
    return JSONResponse(content={
        "task": state.task_name,
        "episode": state.episode_count,
        "step": state.step_count,
        "recording": state.recording,
        "episodes_recorded": state.collector.num_episodes if state.collector else 0,
        "total_steps_recorded": state.collector.total_steps if state.collector else 0,
    })


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

    # Create environment if not exists
    if state.env is None:
        state.create_env()

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

            # Handle reset
            if state.reset_flag:
                state.reset_flag = False
                if state.recording and state.collector:
                    state.collector.end_episode()
                state.obs, _ = state.env.reset()
                state.step_count = 0
                state.episode_count += 1
                state.keys_pressed.clear()
                state.gripper_target = 0.0

            # Compute and apply action
            action = state.compute_action()
            next_obs, reward, terminated, truncated, info = state.env.step(action)
            done = terminated or truncated

            # Record if active
            if state.recording and state.collector:
                state.collector.step(state.obs, action, reward, done)

            state.obs = next_obs
            state.step_count += 1
            state.last_reward = reward
            state.last_success = info.get("success", False)

            # Render frame
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
                "reward": round(reward, 4),
                "total_reward": round(state.last_reward, 4),
                "episode": state.episode_count,
                "recording": state.recording,
                "success": state.last_success,
                "fps": round(current_fps, 1),
                "task": state.task_name,
                "episodes_recorded": state.collector.num_episodes if state.collector else 0,
                "gripper": "closed" if state.gripper_target > 0 else "open",
            }))

            # Auto-reset on termination
            if done:
                reason = "success" if state.last_success else "terminated" if terminated else "truncated"
                logger.info(f"Episode {state.episode_count} ended: {reason} at step {state.step_count}")
                if state.recording and state.collector:
                    state.collector.end_episode()
                state.obs, _ = state.env.reset()
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


def handle_client_message(msg: dict):
    """Process a message from the frontend client."""
    msg_type = msg.get("type", "")

    if msg_type == "key_down":
        key = msg.get("key", "").lower()
        if key in ("w", "a", "s", "d", "q", "e"):
            state.keys_pressed.add(key)

    elif msg_type == "key_up":
        key = msg.get("key", "").lower()
        state.keys_pressed.discard(key)

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
            state.task_name = new_task
            state.create_env()
            logger.info(f"Task changed to: {new_task}")

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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Metaworld Web Teleoperation Server")
    parser.add_argument("--port", type=int, default=9527, help="Server port (default: 9527)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--task", type=str, default="pick-place-v3", help="Initial task")
    parser.add_argument("--speed", type=float, default=0.1, help="Movement speed (default: 0.1)")
    parser.add_argument("--save-dir", type=str, default="data/datasets/teleop", help="Save directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--width", type=int, default=640, help="Render width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Render height (default: 480)")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality 1-100 (default: 80)")

    args = parser.parse_args()

    # Configure state
    state.task_name = args.task
    state.speed = args.speed
    state.save_dir = args.save_dir
    state.seed = args.seed
    state.target_fps = args.fps
    state.render_width = args.width
    state.render_height = args.height
    state.jpeg_quality = args.jpeg_quality

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    logger.info(f"Starting web teleoperation server on {args.host}:{args.port}")
    logger.info(f"Task: {args.task} | Speed: {args.speed} | FPS: {args.fps}")
    logger.info(f"Open http://<server_ip>:{args.port} in your browser to start teleoperation")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
