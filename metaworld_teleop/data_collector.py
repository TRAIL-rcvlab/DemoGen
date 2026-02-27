"""
Data collector for Metaworld teleoperation.
Saves demonstration trajectories in zarr format compatible with DemoGen pipeline.
"""

import os
import time
import numpy as np

try:
    import zarr
except ImportError:
    zarr = None
    print("[WARNING] zarr not installed. Data saving will be disabled.")


class DemoCollector:
    """
    Collects and saves demonstration trajectories from teleoperation sessions.

    Data is saved in zarr format with the following structure:
        dataset.zarr/
        ├── data/
        │   ├── observations    (N, obs_dim) float32
        │   ├── actions         (N, act_dim) float32
        │   ├── rewards         (N,)         float32
        │   └── dones           (N,)         bool
        └── meta/
            └── episode_ends    (num_episodes,) int64
    """

    def __init__(self, save_dir=None, obs_dim=39, act_dim=4):
        """
        Args:
            save_dir: Directory to save zarr datasets. None = don't save.
            obs_dim: Observation dimension (Metaworld default: 39)
            act_dim: Action dimension (Metaworld default: 4)
        """
        self.save_dir = save_dir
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Buffer for current episode
        self._current_episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

        # All completed episodes
        self._episodes = []
        self._total_steps = 0

    def step(self, obs, action, reward, done):
        """Record a single transition."""
        self._current_episode["observations"].append(np.array(obs, dtype=np.float32))
        self._current_episode["actions"].append(np.array(action, dtype=np.float32))
        self._current_episode["rewards"].append(float(reward))
        self._current_episode["dones"].append(bool(done))
        self._total_steps += 1

    def end_episode(self):
        """Finalize the current episode and start a new one."""
        if len(self._current_episode["observations"]) == 0:
            return

        episode = {
            k: np.array(v) for k, v in self._current_episode.items()
        }
        self._episodes.append(episode)

        n_steps = len(episode["observations"])
        print(
            f"  Episode {len(self._episodes)} completed: "
            f"{n_steps} steps, "
            f"total reward: {episode['rewards'].sum():.2f}"
        )

        # Reset buffer
        self._current_episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

    def save(self, task_name="unknown"):
        """
        Save all collected episodes to a zarr dataset.

        Args:
            task_name: Name of the task for the filename
        """
        if zarr is None:
            print("[ERROR] zarr not installed, cannot save data.")
            return None

        if len(self._episodes) == 0:
            print("[WARNING] No episodes to save.")
            return None

        if self.save_dir is None:
            print("[WARNING] No save directory specified.")
            return None

        os.makedirs(self.save_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        zarr_path = os.path.join(self.save_dir, f"teleop_{task_name}_{timestamp}.zarr")

        # Concatenate all episodes
        all_obs = np.concatenate([ep["observations"] for ep in self._episodes])
        all_actions = np.concatenate([ep["actions"] for ep in self._episodes])
        all_rewards = np.concatenate([ep["rewards"] for ep in self._episodes])
        all_dones = np.concatenate([ep["dones"] for ep in self._episodes])

        # Compute episode end indices
        episode_ends = np.cumsum([len(ep["observations"]) for ep in self._episodes])

        # Save to zarr
        root = zarr.open(zarr_path, mode="w")
        data_group = root.create_group("data")
        data_group.create_dataset("observations", data=all_obs, chunks=(1000, all_obs.shape[1]))
        data_group.create_dataset("actions", data=all_actions, chunks=(1000, all_actions.shape[1]))
        data_group.create_dataset("rewards", data=all_rewards, chunks=(1000,))
        data_group.create_dataset("dones", data=all_dones, chunks=(1000,))

        meta_group = root.create_group("meta")
        meta_group.create_dataset("episode_ends", data=episode_ends)

        print(f"\n{'='*60}")
        print(f"  Data saved to: {zarr_path}")
        print(f"  Episodes: {len(self._episodes)}")
        print(f"  Total steps: {self._total_steps}")
        print(f"  Observations shape: {all_obs.shape}")
        print(f"  Actions shape: {all_actions.shape}")
        print(f"{'='*60}\n")

        return zarr_path

    @property
    def num_episodes(self):
        return len(self._episodes)

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def current_episode_steps(self):
        return len(self._current_episode["observations"])
