"""
We are interested in sampling frames from a given episode.
We have a thesis that frames towards the end of an episode are more interesting than those in the beginning.
Below, we implement a few strategies to present this strategy with a focus on an upper bound of F total frames.

Note: individual sampling strategies are not guaranteed to return F frames. Use the `sampling_bias` function to ensure F frames are returned.
The `sampling_bias` function will sample extra frames if necessary to reach the target number of frames or uniformly subsample if more frames are sampled than necessary.
"""

import numpy as np
import typing as t


# linear sampling bias
def linear_sampling_bias(
    input_n_frames: int, total_desired_frames: int, **kwargs
) -> t.Sequence[int]:
    """
    Sample frames with a probability linearly proportional to their index.
    """
    return [i for i in range(input_n_frames) if np.random.random() < i / input_n_frames]


def exponential_sampling_bias(
    input_n_frames: int, total_desired_frames: int, rate: float = 5.0, **kwargs
) -> t.Sequence[int]:
    """
    Sample frames with exponentially increasing probability.

    The probability of selecting a frame increases exponentially with its index,
    controlled by the rate parameter. Higher rates lead to stronger bias towards
    later frames.
    """
    return [
        i
        for i in range(input_n_frames)
        if np.random.rand() < (1 - np.exp(-rate * i / total_desired_frames))
    ]


def threshold_sampling_bias(
    input_n_frames: int,
    total_desired_frames: int,
    frame_threshold: float = 0.75,
    bias_rate: float = 0.5,
):
    """
    Uniformly sample frames with a proportion coming from before the frame_threshold and the rest coming from after.
    Control the bias rate to control the relative number of frames sampled from before and after.
    """
    desired_n_frames_before = int(bias_rate * total_desired_frames)
    desired_n_frames_after = total_desired_frames - desired_n_frames_before

    threshold_n = int(input_n_frames * frame_threshold)
    frames_before = np.random.choice(
        range(threshold_n), desired_n_frames_before, replace=False
    )
    frames_after = np.random.choice(
        range(threshold_n, input_n_frames), desired_n_frames_after, replace=False
    )
    return sorted(list(frames_before) + list(frames_after))


def sampling_bias(
    input_n_frames: int, total_desired_frames: int, strategy: str = "linear", **kwargs
) -> t.Sequence[int]:
    if input_n_frames < total_desired_frames:
        raise ValueError(
            f"Input number of frames ({input_n_frames}) is less than the desired number of frames ({total_desired_frames})."
        )

    if strategy == "linear":
        sampled = linear_sampling_bias(input_n_frames, total_desired_frames)
    elif strategy == "exponential":
        sampled = exponential_sampling_bias(
            input_n_frames, total_desired_frames, **kwargs
        )
    elif strategy == "threshold":
        sampled = threshold_sampling_bias(
            input_n_frames, total_desired_frames, **kwargs
        )
    if len(sampled) < total_desired_frames:
        extra_samples = np.random.choice(
            set(range(input_n_frames)) - set(sampled),
            size=total_desired_frames - len(sampled),
            replace=False,
        )
        sampled = sorted(sampled + extra_samples)
    elif len(sampled) > total_desired_frames:
        sampled = np.random.choice(sampled, total_desired_frames, replace=False)
    return sorted(sampled)
