from typing import Optional, Dict
import numpy as np
from diffusion_policies.common.replay_buffer import ReplayBuffer
from diffusion_policies.common.sampler import create_indices, SequenceSampler


class SegmentWeightedSampler(SequenceSampler):
    """
    对不同轨迹片段施加不同采样权重的采样器。

    轨迹根据 parsing_frames 被划分为 motion（运动）和 skill（技能/规划）片段：
        - motion-1: 帧 [0, skill_1)          — 接近运动
        - skill-1:  帧 [skill_1, motion_2)    — 第一阶段操作技能
        - motion-2: 帧 [motion_2, skill_2)    — 过渡运动
        - skill-2:  帧 [skill_2, end)         — 第二阶段操作技能

    通过按权重（取整为整数倍）重复对应片段的索引来实现过采样。
    所有权重必须 >= 1.0，以保证至少采样一倍原始数据。
    """

    def __init__(self,
                 replay_buffer: ReplayBuffer,
                 sequence_length: int,
                 parsing_frames: Dict[str, int],
                 segment_weights: Dict[str, float],
                 pad_before: int = 0,
                 pad_after: int = 0,
                 keys=None,
                 key_first_k=dict(),
                 episode_mask: Optional[np.ndarray] = None,
                 seed: int = 42,
                 ):
        """
        参数:
            replay_buffer: 包含回放数据的 ReplayBuffer
            sequence_length: 每条采样序列的长度
            parsing_frames: 片段边界字典，例如：
                {"motion-1": 0, "skill-1": 6, "motion-2": 68, "skill-2": 83}
            segment_weights: 采样权重字典，例如：
                {"motion": 2.0, "skill": 1.0}
                权重必须 >= 1.0
            pad_before: 序列前填充步数
            pad_after: 序列后填充步数
            keys: 从 replay buffer 中采样的数据键
            key_first_k: 仅取这些键的前 k 条数据（提升性能）
            episode_mask: 布尔掩码，指定包含哪些 episode
            seed: 随机种子，用于可复现性
        """
        assert segment_weights.get("motion", 1.0) >= 1.0, "motion 权重必须 >= 1.0"
        assert segment_weights.get("skill", 1.0) >= 1.0, "skill 权重必须 >= 1.0"

        # 初始化父类，创建标准索引
        super().__init__(
            replay_buffer=replay_buffer,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=keys,
            key_first_k=key_first_k,
            episode_mask=episode_mask,
        )

        self.parsing_frames = parsing_frames
        self.segment_weights = segment_weights
        self.seed = seed

        # 根据片段类型对索引施加采样权重
        self.indices = self._apply_segment_weights(
            replay_buffer, episode_mask, sequence_length
        )

    def _classify_segment(self, center_frame_in_episode, episode_length):
        """
        将 episode 内的帧位置分类为对应的片段类型。

        参数:
            center_frame_in_episode: 相对于 episode 起始的帧索引
            episode_length: episode 总长度

        返回:
            "motion" 或 "skill"
        """
        skill_1 = self.parsing_frames.get("skill-1", 0)
        motion_2 = self.parsing_frames.get("motion-2")
        skill_2 = self.parsing_frames.get("skill-2")

        # 单阶段任务（motion-2 和 skill-2 为 None）
        if motion_2 is None or skill_2 is None:
            if center_frame_in_episode < skill_1:
                return "motion"
            else:
                return "skill"

        # 双阶段任务
        if center_frame_in_episode < skill_1:
            return "motion"   # motion-1（接近运动）
        elif center_frame_in_episode < motion_2:
            return "skill"    # skill-1（操作技能）
        elif center_frame_in_episode < skill_2:
            return "motion"   # motion-2（过渡运动）
        else:
            return "skill"    # skill-2（操作技能）

    def _apply_segment_weights(self, replay_buffer, episode_mask, sequence_length):
        """
        对每个索引进行片段分类，并按权重重复对应索引。

        返回:
            np.ndarray: 加权后的索引数组
        """
        if len(self.indices) == 0:
            return self.indices

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        motion_weight = self.segment_weights.get("motion", 1.0)
        skill_weight = self.segment_weights.get("skill", 1.0)

        # 将浮点权重转换为整数重复次数
        motion_repeat = max(1, int(round(motion_weight)))
        skill_repeat = max(1, int(round(skill_weight)))

        weighted_indices = []

        for idx_row in self.indices:
            buffer_start_idx = idx_row[0]

            # 确定该索引属于哪个 episode
            episode_idx = np.searchsorted(episode_ends, buffer_start_idx, side='right')
            episode_start = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]

            # 计算序列中心帧在 episode 内的位置
            center_buffer_idx = buffer_start_idx + sequence_length // 2
            center_frame_in_episode = center_buffer_idx - episode_start

            # 分类片段并确定重复次数
            segment_type = self._classify_segment(
                center_frame_in_episode,
                episode_ends[episode_idx] - episode_start
            )

            repeat_count = motion_repeat if segment_type == "motion" else skill_repeat
            for _ in range(repeat_count):
                weighted_indices.append(idx_row)

        weighted_indices = np.array(weighted_indices)

        # 打乱顺序，混合重复的索引
        rng = np.random.default_rng(seed=self.seed)
        rng.shuffle(weighted_indices)

        return weighted_indices
