#!/usr/bin/env python3
"""
Metaworld Keyboard/Mouse Teleoperation Script.

Control a Metaworld robot arm using keyboard and mouse to collect demonstrations.

Controls:
    W/S     - End-effector Y forward/backward
    A/D     - End-effector X left/right
    Q/E     - End-effector Z up/down
    Mouse L - Close gripper (grasp)
    Mouse R - Open gripper (release)
    R       - Reset environment (start new episode)
    ESC     - Quit and save data

Usage:
    python metaworld_teleop/teleop_keyboard_mouse.py --task pick-place-v3
    python metaworld_teleop/teleop_keyboard_mouse.py --task reach-v3 --save-path data/datasets/teleop
"""

import argparse
import sys
import threading
import time
import numpy as np

from pynput import keyboard, mouse

from metaworld_teleop.utils import create_metaworld_env, list_available_tasks, get_env_info
from metaworld_teleop.data_collector import DemoCollector


class TeleopController:
    """
    Keyboard/mouse teleoperation controller for Metaworld environments.

    Translates keyboard/mouse inputs into continuous 4-DOF actions [dx, dy, dz, gripper].
    """

    def __init__(self, speed=0.1):
        """
        Args:
            speed: Movement speed multiplier for keyboard-controlled axes.
        """
        self.speed = speed

        # Movement state (controlled by keyboard)
        self._keys_pressed = set()

        # Gripper state (controlled by mouse)
        self._gripper_target = 0.0  # -1.0 = open, +1.0 = closed

        # Control flags
        self._reset_requested = False
        self._quit_requested = False

        # Input listeners
        self._keyboard_listener = None
        self._mouse_listener = None

    def start(self):
        """Start listening for keyboard and mouse events."""
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._mouse_listener = mouse.Listener(
            on_click=self._on_mouse_click,
        )
        self._keyboard_listener.start()
        self._mouse_listener.start()

    def stop(self):
        """Stop listening for input events."""
        if self._keyboard_listener:
            self._keyboard_listener.stop()
        if self._mouse_listener:
            self._mouse_listener.stop()

    def get_action(self):
        """
        Compute the current action based on pressed keys and mouse state.

        Returns:
            np.ndarray: 4-DOF action [dx, dy, dz, gripper]
        """
        dx, dy, dz = 0.0, 0.0, 0.0

        # Keyboard -> translation
        if "a" in self._keys_pressed:
            dx -= self.speed
        if "d" in self._keys_pressed:
            dx += self.speed
        if "w" in self._keys_pressed:
            dy += self.speed
        if "s" in self._keys_pressed:
            dy -= self.speed
        if "q" in self._keys_pressed:
            dz += self.speed
        if "e" in self._keys_pressed:
            dz -= self.speed

        action = np.array([dx, dy, dz, self._gripper_target], dtype=np.float32)
        return action

    @property
    def reset_requested(self):
        flag = self._reset_requested
        self._reset_requested = False
        return flag

    @property
    def quit_requested(self):
        return self._quit_requested

    # ---- Input callbacks ----

    def _on_key_press(self, key):
        try:
            k = key.char.lower()
            if k in ("w", "a", "s", "d", "q", "e"):
                self._keys_pressed.add(k)
            elif k == "r":
                self._reset_requested = True
        except AttributeError:
            # Special keys
            if key == keyboard.Key.esc:
                self._quit_requested = True

    def _on_key_release(self, key):
        try:
            k = key.char.lower()
            self._keys_pressed.discard(k)
        except AttributeError:
            pass

    def _on_mouse_click(self, x, y, button, pressed):
        if pressed:
            if button == mouse.Button.left:
                self._gripper_target = 1.0  # Close gripper
            elif button == mouse.Button.right:
                self._gripper_target = -1.0  # Open gripper


def print_banner():
    """Print the teleoperation control banner."""
    print("\n" + "=" * 60)
    print("  Metaworld Keyboard/Mouse Teleoperation")
    print("=" * 60)
    print()
    print("  Controls:")
    print("    W/S       Y forward / backward")
    print("    A/D       X left / right")
    print("    Q/E       Z up / down")
    print("    Mouse L   Close gripper (grasp)")
    print("    Mouse R   Open gripper (release)")
    print("    R         Reset (new episode)")
    print("    ESC       Quit & save")
    print()
    print("=" * 60)


def run_teleop(task_name, save_path=None, speed=0.1,
               render_mode="human", seed=42, max_steps_per_episode=500):
    """
    Main teleoperation loop.

    Args:
        task_name: Metaworld task name (e.g. 'pick-place-v3')
        save_path: Directory to save collected data (None = don't save)
        speed: Movement speed multiplier
        render_mode: 'human' for GUI, 'rgb_array' for offscreen
        seed: Random seed
        max_steps_per_episode: Max steps before auto-reset
    """
    print_banner()
    print(f"  Task: {task_name}")
    print(f"  Speed: {speed}")
    print(f"  Save path: {save_path or '(disabled)'}")
    print(f"  Max steps/episode: {max_steps_per_episode}")
    print()

    # Create environment
    print("Creating environment...")
    env = create_metaworld_env(task_name, render_mode=render_mode, seed=seed)
    env_info = get_env_info(env)
    print(f"  Observation space: {env_info['observation_space']}")
    print(f"  Action space: {env_info['action_space']}")
    print()

    # Create data collector
    collector = DemoCollector(
        save_dir=save_path,
        obs_dim=env_info["obs_dim"],
        act_dim=env_info["action_dim"],
    )

    # Create teleop controller
    controller = TeleopController(speed=speed)
    controller.start()

    try:
        episode = 0
        step_count = 0

        # Initial reset
        obs, info = env.reset()
        episode += 1
        step_count = 0
        print(f"\n--- Episode {episode} started ---")

        while not controller.quit_requested:
            # Check for reset request
            if controller.reset_requested:
                collector.end_episode()
                obs, info = env.reset()
                episode += 1
                step_count = 0
                print(f"\n--- Episode {episode} started ---")
                continue

            # Get action from keyboard/mouse
            action = controller.get_action()

            # Clip action to valid range
            action = np.clip(action, env.action_space.low, env.action_space.high)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record data
            collector.step(obs, action, reward, done)

            obs = next_obs
            step_count += 1

            # Print status periodically
            if step_count % 50 == 0:
                success = info.get("success", False)
                print(
                    f"  Step {step_count:4d} | "
                    f"Reward: {reward:+6.2f} | "
                    f"Gripper: {'CLOSED' if action[3] > 0 else 'OPEN  '} | "
                    f"Success: {success}"
                )

            # Auto-reset on termination or max steps
            if done or step_count >= max_steps_per_episode:
                reason = "terminated" if terminated else "truncated" if truncated else "max steps"
                print(f"  Episode ended ({reason}) at step {step_count}")
                collector.end_episode()
                obs, info = env.reset()
                episode += 1
                step_count = 0
                print(f"\n--- Episode {episode} started ---")

            # Control loop rate (~30 Hz)
            time.sleep(1.0 / 30.0)

    except KeyboardInterrupt:
        print("\n\nInterrupted by Ctrl+C")
    finally:
        # Finalize current episode if any data exists
        if collector.current_episode_steps > 0:
            collector.end_episode()

        # Stop input listeners
        controller.stop()

        # Save data
        if save_path and collector.num_episodes > 0:
            zarr_path = collector.save(task_name=task_name)
            if zarr_path:
                print(f"Data saved to: {zarr_path}")
        else:
            print(f"\nSession summary: {collector.num_episodes} episodes, {collector.total_steps} total steps")
            if save_path is None:
                print("(Data not saved — use --save-path to enable saving)")

        # Cleanup
        env.close()
        print("Environment closed. Goodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="Metaworld Keyboard/Mouse Teleoperation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --task reach-v3
  %(prog)s --task pick-place-v3 --save-path data/datasets/teleop
  %(prog)s --task door-open-v3 --speed 0.15 --max-steps 1000
  %(prog)s --list-tasks
        """,
    )
    parser.add_argument(
        "--task", type=str, default="pick-place-v3",
        help="Metaworld task name (default: pick-place-v3)"
    )
    parser.add_argument(
        "--save-path", type=str, default=None,
        help="Directory to save collected demos as zarr (default: disabled)"
    )
    parser.add_argument(
        "--speed", type=float, default=0.1,
        help="Keyboard movement speed multiplier (default: 0.1)"
    )
    parser.add_argument(
        "--render-mode", type=str, default="human",
        choices=["human", "rgb_array"],
        help="Render mode (default: human)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Max steps per episode before auto-reset (default: 500)"
    )
    parser.add_argument(
        "--list-tasks", action="store_true",
        help="List all available tasks and exit"
    )

    args = parser.parse_args()

    if args.list_tasks:
        tasks = list_available_tasks()
        print(f"Available Metaworld tasks ({len(tasks)}):")
        for t in tasks:
            print(f"  {t}")
        sys.exit(0)

    run_teleop(
        task_name=args.task,
        save_path=args.save_path,
        speed=args.speed,
        render_mode=args.render_mode,
        seed=args.seed,
        max_steps_per_episode=args.max_steps,
    )


if __name__ == "__main__":
    main()
