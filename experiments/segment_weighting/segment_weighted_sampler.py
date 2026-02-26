from typing import Optional, Dict
import numpy as np
from diffusion_policies.common.replay_buffer import ReplayBuffer
from diffusion_policies.common.sampler import create_indices, SequenceSampler


class SegmentWeightedSampler(SequenceSampler):
    """
    A sampler that applies different sampling weights to different trajectory segments.

    Trajectories are decomposed into motion and skill segments based on parsing_frames:
        - motion-1: frames [0, skill_1)
        - skill-1:  frames [skill_1, motion_2)
        - motion-2: frames [motion_2, skill_2)
        - skill-2:  frames [skill_2, end)

    Each segment can be oversampled by repeating its indices according to the
    specified weight (integer multiplier derived from the float weight).
    All weights must be >= 1.0 to ensure at least 1x the original data is sampled.
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
        Args:
            replay_buffer: ReplayBuffer containing episode data
            sequence_length: Length of each sampled sequence
            parsing_frames: Dict with segment boundaries, e.g.:
                {"motion-1": 0, "skill-1": 6, "motion-2": 68, "skill-2": 83}
            segment_weights: Dict with sampling weights, e.g.:
                {"motion": 2.0, "skill": 1.0}
                Weights must be >= 1.0
            pad_before: Number of steps to pad before sequence
            pad_after: Number of steps to pad after sequence
            keys: Keys to sample from replay buffer
            key_first_k: Only take first k data from these keys
            episode_mask: Boolean mask for which episodes to include
            seed: Random seed for reproducibility
        """
        assert segment_weights.get("motion", 1.0) >= 1.0, "Motion weight must be >= 1.0"
        assert segment_weights.get("skill", 1.0) >= 1.0, "Skill weight must be >= 1.0"

        # Initialize base class to create standard indices
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

        # Apply segment-based weighting to indices
        self.indices = self._apply_segment_weights(
            replay_buffer, episode_mask, sequence_length
        )

    def _classify_segment(self, center_frame_in_episode, episode_length):
        """
        Classify a frame position within an episode into its segment type.

        Args:
            center_frame_in_episode: Frame index relative to episode start
            episode_length: Total length of the episode

        Returns:
            "motion" or "skill"
        """
        skill_1 = self.parsing_frames.get("skill-1", 0)
        motion_2 = self.parsing_frames.get("motion-2")
        skill_2 = self.parsing_frames.get("skill-2")

        # For one-stage tasks (motion-2 and skill-2 are None)
        if motion_2 is None or skill_2 is None:
            if center_frame_in_episode < skill_1:
                return "motion"
            else:
                return "skill"

        # For two-stage tasks
        if center_frame_in_episode < skill_1:
            return "motion"   # motion-1
        elif center_frame_in_episode < motion_2:
            return "skill"    # skill-1
        elif center_frame_in_episode < skill_2:
            return "motion"   # motion-2
        else:
            return "skill"    # skill-2

    def _apply_segment_weights(self, replay_buffer, episode_mask, sequence_length):
        """
        Classify each index into a segment and repeat according to weights.

        Returns:
            np.ndarray: Weighted indices array
        """
        if len(self.indices) == 0:
            return self.indices

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        motion_weight = self.segment_weights.get("motion", 1.0)
        skill_weight = self.segment_weights.get("skill", 1.0)

        # Convert float weights to integer repeat counts
        motion_repeat = max(1, int(round(motion_weight)))
        skill_repeat = max(1, int(round(skill_weight)))

        weighted_indices = []

        for idx_row in self.indices:
            buffer_start_idx = idx_row[0]

            # Determine which episode this index belongs to
            episode_idx = np.searchsorted(episode_ends, buffer_start_idx, side='right')
            episode_start = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]

            # Calculate the center frame position within the episode
            center_buffer_idx = buffer_start_idx + sequence_length // 2
            center_frame_in_episode = center_buffer_idx - episode_start

            # Classify and determine repeat count
            segment_type = self._classify_segment(
                center_frame_in_episode,
                episode_ends[episode_idx] - episode_start
            )

            repeat_count = motion_repeat if segment_type == "motion" else skill_repeat
            for _ in range(repeat_count):
                weighted_indices.append(idx_row)

        weighted_indices = np.array(weighted_indices)

        # Shuffle to mix the repeated indices
        rng = np.random.default_rng(seed=self.seed)
        rng.shuffle(weighted_indices)

        return weighted_indices
