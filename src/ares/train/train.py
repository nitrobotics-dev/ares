# TODO
from typing import Any, Dict, Tuple

import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader, Dataset

from ares.configs.base import Rollout


class RolloutDataset(Dataset):
    def __init__(self, parquet_path: str):
        """Initialize dataset from preprocessed parquet file."""
        self.df = pd.read_parquet(parquet_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Rollout, Dict[str, Any]]:
        """Get a rollout and its annotations."""
        row = self.df.iloc[idx]

        # Construct Rollout object
        rollout = Rollout(
            **{k: v for k, v in row.items() if k != "annotations" and pd.notna(v)}
        )

        # Get annotations
        annotations = row["annotations"]

        return rollout, annotations


def get_dataloader(
    dataset: RolloutDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader with custom collate function."""

    def collate_fn(batch):
        rollouts, annotations = zip(*batch)
        return list(rollouts), list(annotations)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def main(
    data_path: str = typer.Option(..., help="Path to preprocessed parquet file"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
    num_workers: int = typer.Option(4, help="Number of worker processes"),
):
    # Create dataset and dataloader
    dataset = RolloutDataset(data_path)
    dataloader = get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Training loop would go here
    for rollouts, annotations in dataloader:
        # Process batch
        pass


if __name__ == "__main__":
    typer.run(main)
