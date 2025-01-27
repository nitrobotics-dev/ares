"""
Orchestration script to run grounding annotation using Modal.
"""

import asyncio
import os
import pickle
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from ares.configs.base import Rollout
from ares.constants import (
    ARES_DATA_DIR,
    DATASET_NAMES,
    OUTER_BATCH_SIZE,
    get_dataset_info_by_key,
)
from ares.databases.annotation_database import ANNOTATION_DB_PATH, AnnotationDatabase
from ares.databases.structured_database import (
    ROBOT_DB_PATH,
    RolloutSQLModel,
    get_rollouts_by_ids,
    setup_database,
    setup_rollouts,
)
from ares.models.base import VLM
from ares.models.grounding import ANNOTATION_GROUNDING_FPS
from ares.models.grounding_utils import get_grounding_nouns_async
from ares.models.shortcuts import get_gpt_4o
from ares.utils.image_utils import load_video_frames

from .modal_grounding import GroundingModalWrapper

FAILURES_PATH = os.path.join(ARES_DATA_DIR, "failures.pkl")


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


async def run_annotate_and_ingest(
    annotator: GroundingModalWrapper,
    rollout_ids: List[str],
    annotation_input_futures: List[Any],
    db: AnnotationDatabase,
    rollouts: List[Any],
) -> Tuple[ResultTracker, List[ErrorResult]]:
    """
    Run annotation tasks in parallel using Modal.

    Args:
        annotator (GroundingModalWrapper): Modal wrapper for grounding tasks.
        rollout_ids (List[str]): List of rollout IDs.
        annotation_input_futures (List[Any]): List of asyncio Tasks for preparing annotation inputs.
        db (AnnotationDatabase): Database instance for storing annotations.
        rollouts (List[Any]): List of rollout objects.

    Returns:
        Tuple[ResultTracker, List[ErrorResult]]: Tracker and list of failures.
    """
    id_to_rollout = {r.id: r for r in rollouts}
    id_to_annotation_inputs = {}
    tracker = ResultTracker()
    failures = []

    # Await all preparation tasks
    results = await asyncio.gather(*annotation_input_futures, return_exceptions=True)

    tasks = []

    for res in results:
        if isinstance(res, ErrorResult):
            failures.append(res)
            continue
        elif isinstance(res, Exception):
            breakpoint()
        else:
            rollout_id, frames, frame_indices, label_str = res
        print(
            f"Received grounding output for {rollout_id}: {len(frames)} frames, "
            f"{len(frame_indices)} frame indices, label str: {label_str}"
        )
        id_to_annotation_inputs[rollout_id] = (
            rollout_id,
            frames,
            frame_indices,
            label_str,
        )
        # Prepare annotation task
        tasks.append((rollout_id, frames, label_str))

    # Submit annotation tasks to Modal
    with annotator.app.run():
        annotation_results = await annotator.annotate_videos(tasks)

    breakpoint()

    for rollout_id, all_frame_annotation_dicts in annotation_results:
        try:
            rollout = id_to_rollout[rollout_id]
            _, frames, frame_indices, label_str = id_to_annotation_inputs[rollout_id]
            video_id = db.add_video_with_annotations(
                dataset_filename=rollout.dataset_filename,
                video_path=rollout.filename + ".mp4",
                frames=frames,
                frame_indices=frame_indices,
                annotations=all_frame_annotation_dicts,
                label_str=label_str,
            )
            tracker.update_via_batch(
                n_videos=1,
                n_frames=len(all_frame_annotation_dicts),
                n_annotations=sum(
                    len(frame_annotations)
                    for frame_annotations in all_frame_annotation_dicts
                ),
                video_ids=[video_id],
            )
        except Exception as e:
            print(f"Error processing task: {e}")
            failures.append(
                ErrorResult(
                    rollout_id=rollout_id,
                    error_pattern="grounding_failure",
                    error=traceback.format_exc(),
                )
            )

    return tracker, failures


async def setup_query(
    rollout: Any,
    vlm: VLM,
    target_fps: int = 5,
    refusal_phrases: List[str] | None = None,
) -> Tuple[str, List[np.ndarray], List[int], str] | Dict[str, Any]:
    """
    Prepare annotation inputs for a rollout.

    Args:
        rollout (Any): Rollout object.
        vlm (VLM): Vision-Language Model instance.
        target_fps (int, optional): Target FPS for frame extraction.
        refusal_phrases (List[str], optional): Phrases indicating refusal.

    Returns:
        Tuple[str, List[np.ndarray], List[int], str] | Dict[str, Any]: Prepared data or error dict.
    """
    try:
        frames, frame_indices = load_video_frames(
            rollout.dataset_filename,
            rollout.filename,
            target_fps,
        )
    except Exception as e:
        return ErrorResult(
            rollout_id=rollout.id,
            error_pattern="grounding_failure",
            error=traceback.format_exc(),
        )

    try:
        label_str = await get_grounding_nouns_async(
            vlm,
            frames[0],
            rollout.task.language_instruction,
        )
    except Exception as e:
        return ErrorResult(
            rollout_id=rollout.id,
            error_pattern="grounding_request_failure",
            error=str(e),
        )

    refusal_phrases = refusal_phrases or ["I'm"]
    if any(phrase in label_str for phrase in refusal_phrases):
        return ErrorResult(
            rollout_id=rollout.id,
            error_pattern="grounding_request_failure",
            error=str(e),
        )
    return rollout.id, frames, frame_indices, label_str


async def run_ground_and_annotate(
    rollouts: List[Any],
    vlm: VLM,
    ann_db: AnnotationDatabase,
    target_fps: int = 5,
) -> Tuple[ResultTracker, List[ErrorResult]]:
    """
    Process, ground, and annotate list of rollouts.

    Args:
        rollouts (List[Any]): List of rollout objects.
        vlm (VLM): Vision-Language Model instance.
        ann_db (AnnotationDatabase): Annotation database instance.
        target_fps (int, optional): Target FPS for annotation.

    Returns:
        Tuple[Dict[str, int], List[Dict]]: Tracker and list of failures.
    """
    rollout_ids = [r.id for r in rollouts]

    # Create and gather the futures properly
    annotation_input_futures = [
        asyncio.create_task(setup_query(rollout, vlm, target_fps))
        for rollout in rollouts
    ]

    annotator = GroundingModalWrapper()
    tracker, failures = await run_annotate_and_ingest(
        annotator,
        rollout_ids,
        annotation_input_futures,
        ann_db,
        rollouts,
    )
    return tracker, failures


def setup_rollouts(
    engine, rollout_ids, retry_failed_path, dataset_filename, split
) -> List[Rollout]:
    if retry_failed_path:
        if retry_failed_path.endswith(".pkl"):
            with open(retry_failed_path, "rb") as f:
                failures = pickle.load(f)
            failed_ids = [str(f["rollout_id"]) for f in failures]
            return get_rollouts_by_ids(engine, failed_ids)
        elif retry_failed_path.endswith(".txt"):
            with open(retry_failed_path, "r") as f:
                failed_ids = [line.strip() for line in f.readlines()]
            return get_rollouts_by_ids(engine, failed_ids)
        else:
            raise ValueError(f"Unknown file type: {retry_failed_path}")
    elif rollout_ids:
        return get_rollouts_by_ids(engine, rollout_ids)
    else:
        dataset_info = get_dataset_info_by_key("dataset_filename", dataset_filename)
        if not dataset_info:
            print(f"Dataset filename {dataset_filename} not found.")
            return
        dataset_formalname = dataset_info["dataset_formalname"]
        rollouts = setup_rollouts(engine, dataset_formalname)
        if split:
            rollouts = [r for r in rollouts if r.split == split]
    return rollouts


def orchestrate_grounding_batch(
    rollouts: List[Rollout],
    ann_db: AnnotationDatabase,
    outer_batch_size: int,
    annotation_fps: int,
):
    # initialize objects for batches
    overall_tracker = ResultTracker()
    overall_failures = []

    # Limited by CPU RAM (can't create all requests at once)
    for i in range(0, len(rollouts), outer_batch_size):
        print(
            f"Processing batch {i // outer_batch_size + 1} of {len(rollouts) // outer_batch_size}"
        )
        vlm = get_gpt_4o()
        rollouts_batch = rollouts[i : i + outer_batch_size]
        tracker, failures = asyncio.run(
            run_ground_and_annotate(
                rollouts_batch,
                vlm,
                ann_db,
                annotation_fps,
            )
        )
        print(
            f"Completed batch {i // outer_batch_size + 1} of {max(1, len(rollouts) // outer_batch_size)}"
        )
        overall_tracker.update_tracker(tracker)
        overall_failures.extend(failures)
    return overall_tracker, overall_failures


def orchestrate_grounding(
    engine_path: str,
    ann_db_path: str,
    dataset_filename: str | None = None,
    split: str | None = None,
    rollout_ids: List[str] | None = None,
    outer_batch_size: int = OUTER_BATCH_SIZE,  # RAM limits number of concurrent rollouts formatted into requests
    retry_failed_path: str = None,  # Path to pickle file with failures to retry
    annotation_fps: int = ANNOTATION_GROUNDING_FPS,
) -> None:
    """
    Main function to run grounding annotation using Modal.

    Args:
        engine_path (str): Path to the database engine.
        ann_db_path (str): Path to the annotation database.
        dataset_filename (str, optional): Dataset filename.
        split (str, optional): Data split.
        rollout_ids (List[str], optional): Specific rollout IDs to process.
        outer_batch_size (int, optional): Batch size for processing rollouts.
        retry_failed_path (str, optional): Path to retry failed annotations.
        annotation_fps (int, optional): Frames per second for annotation.
    """
    assert (
        dataset_filename is not None
        or retry_failed_path is not None
        or rollout_ids is not None
    ), f"Must provide either dataset_filename, retry_failed_path, or rollout_ids. Received: dataset_filename={dataset_filename}, retry_failed_path={retry_failed_path}, rollout_ids={rollout_ids}"

    # Initialize databases
    ann_db = AnnotationDatabase(connection_string=ann_db_path)
    engine = setup_database(RolloutSQLModel, path=engine_path)
    rollouts = setup_rollouts(
        engine, rollout_ids, retry_failed_path, dataset_filename, split
    )[
        :1
    ]  # HACK # TODO
    print(f"\n\nFound {len(rollouts)} total rollouts\n\n")
    if not rollouts:
        print(
            f"No rollouts found for dataset filename {dataset_filename}, retry failed path {retry_failed_path}"
        )
        return

    tic = time.time()
    overall_tracker, overall_failures = orchestrate_grounding_batch(
        rollouts, ann_db, outer_batch_size, annotation_fps
    )
    print(f"\n\nFailures: {overall_failures}\n\n")

    # Write failures to file to retry
    with open(FAILURES_PATH, "wb") as f:
        pickle.dump(overall_failures, f)

    print("Time taken:", time.time() - tic)
    print(f"\n\n")
    overall_tracker.print_stats()
    print(f"\nNumber of failures: {len(overall_failures)}")


if __name__ == "__main__":
    fails_path = (
        "/workspaces/ares/data/heal_info/2025-01-27_18-51-45/update_grounding_ids.txt"
    )
    orchestrate_grounding(
        engine_path=ROBOT_DB_PATH,
        ann_db_path=ANNOTATION_DB_PATH,
        retry_failed_path=fails_path,
    )
