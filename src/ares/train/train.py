"""
Simple training script for training on rollouts and annotations. This is a mock to show how to construct training dataset and dataloader based on the ARES platform.
See ares/train/README.md for more details and ares/train/preprocess.py for how to preprocess rollouts and annotations into the train format.
"""

from typing import Any, Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ares.configs.base import Rollout
from ares.utils.image_utils import load_video_frames


@dataclass
class TrainingConfig:
    parquet_path: str
    extra_info_cols: list[str]
    target_fps: int
    include_last_frame: bool


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
    ) -> Dict[str, Rollout | Dict[str, Any] | list[np.ndarray]]:
        """Get a rollout and its annotations."""
        row = self.df.iloc[idx]

        # Construct Rollout object
        rollout = Rollout(
            **{
                k: v
                for k, v in row.items()
                if (k not in self.train_config.extra_info_cols and pd.notna(v))
            }
        )
        # construct extra information like grounding annotations, chain of thought, etc.
        extra_info = {
            k: json.loads(v)
            for k, v in row.items()
            if k in self.train_config.extra_info_cols
        }
        # load images
        images = load_video_frames(
            rollout.dataset_filename,
            rollout.fname,
            target_fps=self.train_config.target_fps,
            include_last_frame=self.train_config.include_last_frame,
        )
        return dict(rollout=rollout, extra_info=extra_info, images=images)


def get_dataloader(
    dataset: RolloutDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader with custom collate function."""

    def collate_fn(batch: list[Dict[str, Rollout | Dict[str, Any] | list[np.ndarray]]]):
        # construct your training data here! example below
        inputs = {
            "image_batch": torch.stack(
                [torch.tensor(item["images"]) for item in batch]
            ),
            "states_batch": torch.stack(
                [
                    torch.tensor(item["rollout"].trajectory.states_array)
                    for item in batch
                ]
            ),
            "extra_info_batch": [item["extra_info"] for item in batch],
        }
        outputs = {
            "actions_batch": torch.stack(
                [
                    torch.tensor(item["rollout"].trajectory.actions_array)
                    for item in batch
                ]
            ),
            "rewards_batch": torch.stack(
                [
                    torch.tensor(item["rollout"].trajectory.rewards_array)
                    for item in batch
                ]
            ),
        }
        return inputs, outputs

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


@click.command()
@click.argument("config_path", type=str)
def main(config_path: str):
    """Train a model on rollouts and annotations."""
    # Load config
    with open(config_path) as f:
        config = TrainingConfig(**yaml.safe_load(f))

    # Create dataset and dataloader
    dataset = RolloutDataset(config)
    dataloader = get_dataloader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )

    mock_model = lambda x: x

    # one example loop
    for inputs, outputs in dataloader:
        preds = mock_model(inputs)
        break


if __name__ == "__main__":
    main()
