"""
Simple training script for training on rollouts and annotations. This is a mock to show how to construct training dataset and dataloader based on the ARES platform.
See ares/train/README.md for more details and ares/train/preprocess.py for how to preprocess rollouts and annotations into the train format.

We include a time-mask and a feature-mask to allow for variable length sequences in both the time dimension and the feature dimension.
"""

import json
import typing as t
from dataclasses import dataclass
from functools import partial

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ares.configs.base import Rollout
from ares.configs.pydantic_sql_helpers import recreate_model
from ares.constants import ANNOTATION_GROUNDING_FPS
from ares.databases.structured_database import RolloutSQLModel
from ares.utils.image_utils import load_video_frames


@dataclass
class TrainingConfig:
    parquet_path: str
    extra_info_cols: list[str]
    target_fps: int = ANNOTATION_GROUNDING_FPS
    include_last_frame: bool = True
    batch_size: int = 2
    num_workers: int = 2
    max_seq_len: int = 100
    image_resize: tuple[int, int] | None = (128, 128)


class RolloutDataset(Dataset):
    """
    Simple dataset for training on rollouts and annotations.
    __getitem__ returns a dictionary with the rollout, images, and extra information according to the extra_info_cols.
    Extra info cols may be grounding annotations, embodied chain of thought, etc.
    """

    def __init__(self, train_config: TrainingConfig):
        """Initialize dataset from preprocessed parquet file."""
        self.train_config = train_config
        self.df = pd.read_parquet(train_config.parquet_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> dict[str, Rollout | dict[str, t.Any] | list[np.ndarray]]:
        """Get a rollout and its annotations."""
        row = self.df.iloc[idx]

        # Construct Rollout object via RolloutSQLModel
        rollout_dict = {
            k: v
            for k, v in row.items()
            if (k not in self.train_config.extra_info_cols and pd.notna(v))
        }
        rollout_sql_model = RolloutSQLModel(**rollout_dict)
        rollout = recreate_model(rollout_sql_model, Rollout)

        # construct extra information like grounding annotations, chain of thought, etc.
        extra_info = {
            k: (json.loads(v) if isinstance(v, str) else v)
            for k, v in row.items()
            if k in self.train_config.extra_info_cols
        }
        # load images
        images, _ = load_video_frames(
            rollout.dataset_filename,
            rollout.filename,
            target_fps=self.train_config.target_fps,
            include_last_frame=self.train_config.include_last_frame,
            resize=self.train_config.image_resize,
        )
        return dict(rollout=rollout, extra_info=extra_info, images=images)


def pad_sequence(
    sequences: list[torch.Tensor],
    max_len: int | None = None,
    pad_feature_dim: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Pad sequences to the same length, optionally padding feature dimension too.
    Returns (padded_tensor, sequence_mask, feature_mask); feature_mask is None if pad_feature_dim=False
    """
    if not sequences:
        return torch.tensor([]), torch.tensor([]), None

    # Get max length if not specified
    max_len = max_len or max(seq.shape[0] for seq in sequences)
    batch_size = len(sequences)

    if pad_feature_dim:
        # Get maximum feature dimension
        max_feat_dim = max(
            seq.shape[1] if len(seq.shape) > 1 else 1 for seq in sequences
        )

        # Initialize padded tensor and masks
        padded = torch.zeros(
            (batch_size, max_len, max_feat_dim),
            dtype=sequences[0].dtype,
            device=sequences[0].device,
        )
        seq_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.bool, device=sequences[0].device
        )
        feat_mask = torch.zeros(
            (batch_size, max_len, max_feat_dim),
            dtype=torch.bool,
            device=sequences[0].device,
        )

        # Fill in values and masks
        for i, seq in enumerate(sequences):
            seq_len = seq.shape[0]
            feat_dim = seq.shape[1] if len(seq.shape) > 1 else 1

            # Reshape if needed
            if len(seq.shape) == 1:
                seq = seq.unsqueeze(1)

            padded[i, :seq_len, :feat_dim] = seq
            seq_mask[i, :seq_len] = True
            feat_mask[i, :seq_len, :feat_dim] = True

        return padded, seq_mask, feat_mask

    else:
        # For image-like tensors, preserve all dimensions after the sequence dimension
        other_dims = sequences[0].shape[1:]

        # Initialize padded tensor and mask
        padded = torch.zeros(
            (batch_size, max_len, *other_dims),
            dtype=sequences[0].dtype,
            device=sequences[0].device,
        )
        mask = torch.zeros(
            (batch_size, max_len), dtype=torch.bool, device=sequences[0].device
        )

        # Fill in values and mask
        for i, seq in enumerate(sequences):
            seq_len = seq.shape[0]
            padded[i, :seq_len] = seq
            mask[i, :seq_len] = True

        return padded, mask, None


def collate_fn(
    batch: list[dict[str, Rollout | dict[str, t.Any] | list[np.ndarray]]],
    max_seq_len: int,
) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    # Get max sequence length in this batch, capped by max_seq_len
    max_frames = min(max(len(item["images"]) for item in batch), max_seq_len)

    # Convert images to tensors and pad
    image_tensors = [torch.tensor(item["images"][:max_frames]) for item in batch]
    padded_images, image_mask, _ = pad_sequence(image_tensors, max_frames)

    # Convert and pad trajectory data with None handling
    states_tensors = [
        (
            torch.tensor(item["rollout"].trajectory.states_array)
            if item["rollout"].trajectory.states_array is not None
            else torch.zeros((0,))
        )
        for item in batch
    ]
    actions_tensors = [
        (
            torch.tensor(item["rollout"].trajectory.actions_array)
            if item["rollout"].trajectory.actions_array is not None
            else torch.zeros((0,))
        )
        for item in batch
    ]
    rewards_tensors = [
        (
            torch.tensor(item["rollout"].trajectory.rewards_array)
            if item["rollout"].trajectory.rewards_array is not None
            else torch.zeros((0,))
        )
        for item in batch
    ]

    padded_states, states_seq_mask, states_feat_mask = pad_sequence(
        states_tensors, pad_feature_dim=True
    )
    padded_actions, actions_seq_mask, actions_feat_mask = pad_sequence(
        actions_tensors, pad_feature_dim=True
    )
    padded_rewards, rewards_mask, _ = pad_sequence(rewards_tensors)

    inputs = {
        "image_batch": padded_images,
        "image_mask": image_mask,
        "states_batch": padded_states,
        "states_seq_mask": states_seq_mask,
        "states_feat_mask": states_feat_mask,
        "extra_info_batch": [item["extra_info"] for item in batch],
    }

    outputs = {
        "actions_batch": padded_actions,
        "actions_seq_mask": actions_seq_mask,
        "actions_feat_mask": actions_feat_mask,
        "rewards_batch": padded_rewards,
        "rewards_mask": rewards_mask,
    }
    return inputs, outputs


def get_dataloader(
    dataset: RolloutDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader with custom collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, max_seq_len=dataset.train_config.max_seq_len),
    )


MOCK_MODEL = lambda x: x


@click.command()
@click.option("--preprocessed-path", type=str)
@click.option(
    "--extra-info-cols",
    type=str,
    multiple=True,
    help="Extra info columns to collect annotations for. Can be specified multiple times for different columns",
    required=False,
    default=None,
)
def main(preprocessed_path: str, extra_info_cols: list[str]) -> None:
    """Train a model on rollouts and annotations."""
    # Load config
    config = TrainingConfig(
        parquet_path=preprocessed_path,
        extra_info_cols=extra_info_cols,
    )

    # Create dataset and dataloader
    dataset = RolloutDataset(config)
    dataloader = get_dataloader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )

    # example loop
    for i, (inputs, outputs) in enumerate(dataloader):
        if i == 0:
            print(inputs.keys())
            print(outputs.keys())

        breakpoint()
        preds = MOCK_MODEL(inputs)

        if i > 10:
            break


if __name__ == "__main__":
    main()
