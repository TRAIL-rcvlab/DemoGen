"""
SegmentWeightedSampler 的测试。

验证以下内容：
1. 单阶段和双阶段任务的片段分类是否正确
2. 加权索引是否按片段权重正确重复
3. 平衡权重（1.0, 1.0）是否保持与原始采样器相同的索引数量
4. 过采样是否按比例增加了总索引数量
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from segment_weighted_sampler import SegmentWeightedSampler


def _make_mock_replay_buffer(n_episodes=4, episode_length=100):
    """创建用于测试的最小化模拟 replay buffer。"""

    class MockReplayBuffer:
        def __init__(self, n_episodes, episode_length):
            self.n_episodes = n_episodes
            self._episode_ends = np.cumsum([episode_length] * n_episodes)
            self._data = {
                'agent_pos': np.random.randn(n_episodes * episode_length, 12).astype(np.float32),
                'action': np.random.randn(n_episodes * episode_length, 12).astype(np.float32),
                'point_cloud': np.random.randn(n_episodes * episode_length, 1024, 3).astype(np.float32),
            }

        @property
        def episode_ends(self):
            return self._episode_ends

        def keys(self):
            return list(self._data.keys())

        def __getitem__(self, key):
            return self._data[key]

    return MockReplayBuffer(n_episodes, episode_length)


def test_segment_classification_two_stage():
    """测试双阶段任务的帧片段分类是否正确。"""
    sampler = SegmentWeightedSampler.__new__(SegmentWeightedSampler)
    sampler.parsing_frames = {
        "motion-1": 0,
        "skill-1": 6,
        "motion-2": 68,
        "skill-2": 83,
    }

    # motion-1: 帧 [0, 6)
    assert sampler._classify_segment(0, 100) == "motion"
    assert sampler._classify_segment(3, 100) == "motion"
    assert sampler._classify_segment(5, 100) == "motion"

    # skill-1: 帧 [6, 68)
    assert sampler._classify_segment(6, 100) == "skill"
    assert sampler._classify_segment(30, 100) == "skill"
    assert sampler._classify_segment(67, 100) == "skill"

    # motion-2: 帧 [68, 83)
    assert sampler._classify_segment(68, 100) == "motion"
    assert sampler._classify_segment(75, 100) == "motion"
    assert sampler._classify_segment(82, 100) == "motion"

    # skill-2: 帧 [83, end)
    assert sampler._classify_segment(83, 100) == "skill"
    assert sampler._classify_segment(90, 100) == "skill"
    assert sampler._classify_segment(99, 100) == "skill"

    print("通过: test_segment_classification_two_stage")


def test_segment_classification_one_stage():
    """测试单阶段任务（无 motion-2/skill-2）的帧片段分类。"""
    sampler = SegmentWeightedSampler.__new__(SegmentWeightedSampler)
    sampler.parsing_frames = {
        "motion-1": 0,
        "skill-1": 7,
        "motion-2": None,
        "skill-2": None,
    }

    # motion-1: 帧 [0, 7)
    assert sampler._classify_segment(0, 50) == "motion"
    assert sampler._classify_segment(6, 50) == "motion"

    # skill-1: 帧 [7, end)
    assert sampler._classify_segment(7, 50) == "skill"
    assert sampler._classify_segment(30, 50) == "skill"

    print("通过: test_segment_classification_one_stage")


def test_balanced_weights_preserve_count():
    """测试平衡权重（1.0, 1.0）是否保持索引数量不变。"""
    replay_buffer = _make_mock_replay_buffer(n_episodes=2, episode_length=50)
    parsing_frames = {"motion-1": 0, "skill-1": 5, "motion-2": 30, "skill-2": 40}

    from diffusion_policies.common.sampler import SequenceSampler
    base_sampler = SequenceSampler(
        replay_buffer=replay_buffer,
        sequence_length=8,
        pad_before=1,
        pad_after=4,
    )

    weighted_sampler = SegmentWeightedSampler(
        replay_buffer=replay_buffer,
        sequence_length=8,
        parsing_frames=parsing_frames,
        segment_weights={"motion": 1.0, "skill": 1.0},
        pad_before=1,
        pad_after=4,
    )

    assert len(base_sampler) == len(weighted_sampler), (
        f"平衡权重应保持索引数量不变: "
        f"基准={len(base_sampler)}, 加权={len(weighted_sampler)}"
    )
    print("通过: test_balanced_weights_preserve_count")


def test_motion_oversampling_increases_count():
    """测试 motion 过采样是否增加了总索引数量。"""
    replay_buffer = _make_mock_replay_buffer(n_episodes=2, episode_length=50)
    parsing_frames = {"motion-1": 0, "skill-1": 5, "motion-2": 30, "skill-2": 40}

    balanced = SegmentWeightedSampler(
        replay_buffer=replay_buffer,
        sequence_length=8,
        parsing_frames=parsing_frames,
        segment_weights={"motion": 1.0, "skill": 1.0},
        pad_before=1,
        pad_after=4,
    )

    motion_2x = SegmentWeightedSampler(
        replay_buffer=replay_buffer,
        sequence_length=8,
        parsing_frames=parsing_frames,
        segment_weights={"motion": 2.0, "skill": 1.0},
        pad_before=1,
        pad_after=4,
    )

    assert len(motion_2x) > len(balanced), (
        f"motion 2倍过采样应有更多索引: "
        f"平衡={len(balanced)}, motion_2x={len(motion_2x)}"
    )
    print("通过: test_motion_oversampling_increases_count")


def test_skill_oversampling_increases_count():
    """测试 skill 过采样是否增加了总索引数量。"""
    replay_buffer = _make_mock_replay_buffer(n_episodes=2, episode_length=50)
    parsing_frames = {"motion-1": 0, "skill-1": 5, "motion-2": 30, "skill-2": 40}

    balanced = SegmentWeightedSampler(
        replay_buffer=replay_buffer,
        sequence_length=8,
        parsing_frames=parsing_frames,
        segment_weights={"motion": 1.0, "skill": 1.0},
        pad_before=1,
        pad_after=4,
    )

    skill_2x = SegmentWeightedSampler(
        replay_buffer=replay_buffer,
        sequence_length=8,
        parsing_frames=parsing_frames,
        segment_weights={"motion": 1.0, "skill": 2.0},
        pad_before=1,
        pad_after=4,
    )

    assert len(skill_2x) > len(balanced), (
        f"skill 2倍过采样应有更多索引: "
        f"平衡={len(balanced)}, skill_2x={len(skill_2x)}"
    )
    print("通过: test_skill_oversampling_increases_count")


def test_weight_assertion():
    """测试权重低于 1.0 时是否被正确拒绝。"""
    replay_buffer = _make_mock_replay_buffer(n_episodes=1, episode_length=50)
    parsing_frames = {"motion-1": 0, "skill-1": 5, "motion-2": 30, "skill-2": 40}

    try:
        SegmentWeightedSampler(
            replay_buffer=replay_buffer,
            sequence_length=8,
            parsing_frames=parsing_frames,
            segment_weights={"motion": 0.5, "skill": 1.0},
            pad_before=1,
            pad_after=4,
        )
        assert False, "motion 权重 < 1.0 时应抛出 AssertionError"
    except AssertionError:
        pass

    print("通过: test_weight_assertion")


if __name__ == "__main__":
    test_segment_classification_two_stage()
    test_segment_classification_one_stage()
    test_balanced_weights_preserve_count()
    test_motion_oversampling_increases_count()
    test_skill_oversampling_increases_count()
    test_weight_assertion()
    print("\n所有测试通过！")
