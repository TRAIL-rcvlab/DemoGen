"""
Tests for the SegmentWeightedSampler.

These tests validate that:
1. Segment classification works correctly for both one-stage and two-stage tasks
2. Weighted indices are properly duplicated based on segment weights
3. Balanced weights produce the same number of indices as the original sampler
4. Oversampling increases the total number of indices proportionally
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from segment_weighted_sampler import SegmentWeightedSampler


def _make_mock_replay_buffer(n_episodes=4, episode_length=100):
    """Create a minimal mock replay buffer for testing."""

    class MockReplayBuffer:
        def __init__(self, n_eps, ep_len):
            self.n_episodes = n_eps
            self._episode_ends = np.cumsum([ep_len] * n_eps)
            self._data = {
                'agent_pos': np.random.randn(n_eps * ep_len, 12).astype(np.float32),
                'action': np.random.randn(n_eps * ep_len, 12).astype(np.float32),
                'point_cloud': np.random.randn(n_eps * ep_len, 1024, 3).astype(np.float32),
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
    """Test that frames are classified into correct segments for two-stage tasks."""
    sampler = SegmentWeightedSampler.__new__(SegmentWeightedSampler)
    sampler.parsing_frames = {
        "motion-1": 0,
        "skill-1": 6,
        "motion-2": 68,
        "skill-2": 83,
    }

    # motion-1: frames [0, 6)
    assert sampler._classify_segment(0, 100) == "motion"
    assert sampler._classify_segment(3, 100) == "motion"
    assert sampler._classify_segment(5, 100) == "motion"

    # skill-1: frames [6, 68)
    assert sampler._classify_segment(6, 100) == "skill"
    assert sampler._classify_segment(30, 100) == "skill"
    assert sampler._classify_segment(67, 100) == "skill"

    # motion-2: frames [68, 83)
    assert sampler._classify_segment(68, 100) == "motion"
    assert sampler._classify_segment(75, 100) == "motion"
    assert sampler._classify_segment(82, 100) == "motion"

    # skill-2: frames [83, end)
    assert sampler._classify_segment(83, 100) == "skill"
    assert sampler._classify_segment(90, 100) == "skill"
    assert sampler._classify_segment(99, 100) == "skill"

    print("PASSED: test_segment_classification_two_stage")


def test_segment_classification_one_stage():
    """Test that frames are classified for one-stage tasks (no motion-2/skill-2)."""
    sampler = SegmentWeightedSampler.__new__(SegmentWeightedSampler)
    sampler.parsing_frames = {
        "motion-1": 0,
        "skill-1": 7,
        "motion-2": None,
        "skill-2": None,
    }

    # motion-1: frames [0, 7)
    assert sampler._classify_segment(0, 50) == "motion"
    assert sampler._classify_segment(6, 50) == "motion"

    # skill-1: frames [7, end)
    assert sampler._classify_segment(7, 50) == "skill"
    assert sampler._classify_segment(30, 50) == "skill"

    print("PASSED: test_segment_classification_one_stage")


def test_balanced_weights_preserve_count():
    """Test that balanced weights (1.0, 1.0) produce the same number of indices."""
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
        f"Balanced weights should preserve index count: "
        f"base={len(base_sampler)}, weighted={len(weighted_sampler)}"
    )
    print("PASSED: test_balanced_weights_preserve_count")


def test_motion_oversampling_increases_count():
    """Test that motion oversampling increases total indices."""
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
        f"Motion 2x should have more indices: "
        f"balanced={len(balanced)}, motion_2x={len(motion_2x)}"
    )
    print("PASSED: test_motion_oversampling_increases_count")


def test_skill_oversampling_increases_count():
    """Test that skill oversampling increases total indices."""
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
        f"Skill 2x should have more indices: "
        f"balanced={len(balanced)}, skill_2x={len(skill_2x)}"
    )
    print("PASSED: test_skill_oversampling_increases_count")


def test_weight_assertion():
    """Test that weights below 1.0 are rejected."""
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
        assert False, "Should have raised AssertionError for motion weight < 1.0"
    except AssertionError:
        pass

    print("PASSED: test_weight_assertion")


if __name__ == "__main__":
    test_segment_classification_two_stage()
    test_segment_classification_one_stage()
    test_balanced_weights_preserve_count()
    test_motion_oversampling_increases_count()
    test_skill_oversampling_increases_count()
    test_weight_assertion()
    print("\nAll tests passed!")
