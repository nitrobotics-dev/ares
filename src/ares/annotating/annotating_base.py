"""
Base classes and functions for annotating rollouts, either by API, Modal, or local.
"""

import pickle
from dataclasses import dataclass, field

from sqlalchemy import Engine

from ares.configs.base import Rollout
from ares.constants import get_dataset_info_by_key
from ares.databases.structured_database import get_rollouts_by_ids, setup_rollouts


@dataclass
class ErrorResult:
    rollout_id: str
    error_pattern: str
    error: str
    exception: str | None = None


@dataclass
class ResultTracker:
    videos: int = 0
    frames: int = 0
    annotations: int = 0
    video_ids: list[str] = field(default_factory=list)

    def update_via_batch(
        self, n_videos: int, n_frames: int, n_annotations: int, video_ids: list[str]
    ) -> None:
        self.videos += n_videos
        self.frames += n_frames
        self.annotations += n_annotations
        self.video_ids.extend(video_ids)

    def update_tracker(self, tracker: "ResultTracker") -> None:
        self.videos += tracker.videos
        self.frames += tracker.frames
        self.annotations += tracker.annotations
        self.video_ids.extend(tracker.video_ids)

    def print_stats(self) -> None:
        print(
            f"Processed {self.videos} videos, {self.frames} frames, {len(self.video_ids)} videos with annotations"
        )


def setup_rollouts_from_sources(
    engine: Engine,
    rollout_ids: list[str] | None = None,
    ids_path: str | None = None,
    dataset_filename: str | None = None,
    split: str | None = None,
) -> list[Rollout]:
    """
    Helper function to setup rollouts from a variety of sources, e.g. a list of rollout IDs, a dataset filename, or a file path to a list of failed rollout IDs.
    The file path can be a pickle or a txt file; the pickle file should contain a list of dictionaries with a `rollout_id` key whereas the txt file should contain a list of rollout IDs.
    """
    assert (
        ids_path or rollout_ids or dataset_filename
    ), f"Must provide either ids_path, rollout_ids, or dataset_filename. Received: ids_path={ids_path}, rollout_ids={rollout_ids}, dataset_filename={dataset_filename}"

    if ids_path:
        # load rollouts from a file path
        # - a pickle implies a list of failed rollout dictionaries
        # - a txt implies a list of rollout IDs directly
        if ids_path.endswith(".pkl"):
            with open(ids_path, "rb") as f:
                failures = pickle.load(f)
            failed_ids = [str(f["rollout_id"]) for f in failures]
            return get_rollouts_by_ids(engine, failed_ids)
        elif ids_path.endswith(".txt"):
            with open(ids_path, "r") as f:
                failed_ids = [line.strip() for line in f.readlines()]
            return get_rollouts_by_ids(engine, failed_ids)
        else:
            raise ValueError(f"Unknown file type: {ids_path}")
    elif rollout_ids:
        # load rollouts from a list of IDs
        return get_rollouts_by_ids(engine, rollout_ids)
    else:
        # load rollouts from a dataset filename
        dataset_info = get_dataset_info_by_key("dataset_filename", dataset_filename)
        if not dataset_info:
            raise ValueError(f"Dataset filename {dataset_filename} not found.")
        dataset_formalname = dataset_info["dataset_formalname"]
        rollouts = setup_rollouts(engine, dataset_formalname=dataset_formalname)
        if split:
            rollouts = [r for r in rollouts if r.split == split]
    return rollouts
