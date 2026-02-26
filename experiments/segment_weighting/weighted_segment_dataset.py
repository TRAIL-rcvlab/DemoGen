from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policies.common.pytorch_util import dict_apply
from diffusion_policies.common.replay_buffer import ReplayBuffer
from diffusion_policies.common.sampler import get_val_mask, downsample_mask
from diffusion_policies.model_dp3.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policies.dataset.base_dataset import BasePointcloudDataset

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from segment_weighted_sampler import SegmentWeightedSampler


class WeightedSegmentDataset(BasePointcloudDataset):
    """
    A dataset that applies segment-aware weighted sampling for training.

    Extends the standard PandaDataset by using SegmentWeightedSampler,
    which oversamples motion or skill segments based on configurable weights.
    This enables experiments to test whether different data segments
    contribute differently to the learned policy.
    """

    def __init__(self,
                 zarr_path,
                 parsing_frames,
                 segment_weights,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 task_name=None,
                 ):
        """
        Args:
            zarr_path: Path to zarr dataset
            parsing_frames: Dict with segment boundaries, e.g.:
                {"motion-1": 0, "skill-1": 6, "motion-2": 68, "skill-2": 83}
            segment_weights: Dict with sampling weights, e.g.:
                {"motion": 2.0, "skill": 1.0}
            horizon: Sequence length for sampling
            pad_before: Padding before sequence
            pad_after: Padding after sequence
            seed: Random seed
            val_ratio: Validation set ratio
            max_train_episodes: Maximum number of training episodes
            task_name: Name of the task
        """
        super().__init__()
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['agent_pos', 'action', 'point_cloud'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SegmentWeightedSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            parsing_frames=parsing_frames,
            segment_weights=segment_weights,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            seed=seed,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.parsing_frames = parsing_frames
        self.segment_weights = segment_weights
        self.seed = seed

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        # Validation uses uniform sampling (no weighting)
        val_set.sampler = SegmentWeightedSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            parsing_frames=self.parsing_frames,
            segment_weights={"motion": 1.0, "skill": 1.0},
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            seed=self.seed,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['agent_pos'][..., :],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['agent_pos'][:, ].astype(np.float32)
        point_cloud = sample['point_cloud'][:, ].astype(np.float32)

        data = {
            'obs': {
                'point_cloud': point_cloud,
                'agent_pos': agent_pos,
            },
            'action': sample['action'].astype(np.float32)
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
