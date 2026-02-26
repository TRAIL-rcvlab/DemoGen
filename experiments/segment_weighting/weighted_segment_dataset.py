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
    支持按片段加权采样的数据集类。

    基于 PandaDataset 扩展，使用 SegmentWeightedSampler 对 motion（运动）或
    skill（技能/规划）片段进行过采样。通过改变不同片段的采样权重，可以验证
    数据的不同片段对模型性能的贡献是否不同。
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
        参数:
            zarr_path: zarr 数据集路径
            parsing_frames: 片段边界字典，例如：
                {"motion-1": 0, "skill-1": 6, "motion-2": 68, "skill-2": 83}
            segment_weights: 采样权重字典，例如：
                {"motion": 2.0, "skill": 1.0}
            horizon: 采样序列长度
            pad_before: 序列前填充步数
            pad_after: 序列后填充步数
            seed: 随机种子
            val_ratio: 验证集比例
            max_train_episodes: 最大训练 episode 数量
            task_name: 任务名称
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
        # 验证集使用均匀采样（不加权）
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
