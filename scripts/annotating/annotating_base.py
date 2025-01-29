import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Callable, List, Protocol, TypeAlias

from sqlalchemy import Engine
from typing_extensions import TypeAlias  # for Python <3.10

from ares.configs.base import Rollout
from ares.constants import (
    ANNOTATION_OUTER_BATCH_SIZE,
    ARES_DATA_DIR,
    DATASET_NAMES,
    get_dataset_info_by_key,
)
from ares.databases.annotation_database import AnnotationDatabase
from ares.databases.structured_database import (
    RolloutSQLModel,
    get_rollouts_by_ids,
    setup_database,
    setup_rollouts,
)


@dataclass
class ErrorResult:
    rollout_id: str
    error_pattern: str
    error: str


@dataclass
class ResultTracker:
    videos: int = 0
    frames: int = 0
    annotations: int = 0
    video_ids: List[str] = field(default_factory=list)

    def update_via_batch(
        self, n_videos: int, n_frames: int, n_annotations: int, video_ids: List[str]
    ):
        self.videos += n_videos
        self.frames += n_frames
        self.annotations += n_annotations
        self.video_ids.extend(video_ids)

    def update_tracker(self, tracker: "ResultTracker"):
        self.videos += tracker.videos
        self.frames += tracker.frames
        self.annotations += tracker.annotations
        self.video_ids.extend(tracker.video_ids)

    def print_stats(self):
        print(
            f"Processed {self.videos} videos, {self.frames} frames, {len(self.video_ids)} videos with annotations"
        )


def setup_rollouts_from_sources(
    engine: Engine,
    rollout_ids: List[str] | None = None,
    ids_path: str | None = None,
    dataset_filename: str | None = None,
    split: str | None = None,
) -> List[Rollout]:
    """
    Helper function to setup rollouts from a variety of sources.
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


class AnnotatingFn(Protocol):
    """
    Compute primitive to annotate rollouts. AnnotatingFns should conduct their annotation in batches,
    whether via API, Modal, local, etc.
    """

    def __call__(
        self,
        rollouts: List[Rollout],
        ann_db: AnnotationDatabase,
        outer_batch_size: int,
        **kwargs,
    ) -> tuple[ResultTracker, List[ErrorResult]]: ...


def orchestrate_annotating(
    engine_path: str,
    ann_db_path: str,
    annotating_fn: AnnotatingFn,
    dataset_filename: str | None = None,
    split: str | None = None,
    rollout_ids: List[str] | None = None,
    outer_batch_size: int = ANNOTATION_OUTER_BATCH_SIZE,  # RAM limits number of concurrent rollouts formatted into requests
    ids_path: str = None,  # Path to ids to load; may be failed IDs from previous run
    annotating_kwargs: dict | None = None,
    failures_path: str | None = None,
) -> tuple[ResultTracker, List[ErrorResult]]:
    """
    Main function to run grounding annotation using Modal.

    Args:
        engine_path (str): Path to the database engine.
        ann_db_path (str): Path to the annotation database.
        annotating_fn (Callable): Function to run annotation.
        dataset_filename (str, optional): Dataset filename.
        split (str, optional): Data split.
        rollout_ids (List[str], optional): Specific rollout IDs to process.
        outer_batch_size (int, optional): Batch size for processing rollouts.
        ids_path (str, optional): Path to ids to load.
        annotating_kwargs (dict, optional): Additional keyword arguments for annotating_fn.
    """
    assert (
        dataset_filename is not None or ids_path is not None or rollout_ids is not None
    ), f"Must provide either dataset_filename, ids_path, or rollout_ids. Received: dataset_filename={dataset_filename}, ids_path={ids_path}, rollout_ids={rollout_ids}"

    annotating_kwargs = annotating_kwargs or {}
    # Initialize databases
    ann_db = AnnotationDatabase(connection_string=ann_db_path)
    engine = setup_database(RolloutSQLModel, path=engine_path)
    rollouts = setup_rollouts_from_sources(
        engine, rollout_ids, ids_path, dataset_filename, split
    )
    print(f"\n\nFound {len(rollouts)} total rollouts\n\n")
    if not rollouts:
        print(
            f"No rollouts found for dataset filename {dataset_filename}, retry failed path {ids_path}"
        )
        return

    tic = time.time()
    overall_tracker, overall_failures = annotating_fn(
        rollouts, ann_db, outer_batch_size, **annotating_kwargs
    )
    print(f"\n\nFailures: {overall_failures}\n\n")

    # Write failures to file to retry
    if failures_path and len(overall_failures) > 0:
        os.makedirs(os.path.dirname(failures_path), exist_ok=True)
        print(f"Writing failures to {failures_path}")
        with open(failures_path, "wb") as f:
            pickle.dump(overall_failures, f)

    print("Time taken:", time.time() - tic)
    print(f"\n\n")
    overall_tracker.print_stats()
    print(f"\nNumber of failures: {len(overall_failures)}")
    return overall_tracker, overall_failures
