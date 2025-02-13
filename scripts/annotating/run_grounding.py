"""
Orchestration script to run grounding annotations using Modal.
"""

import asyncio
import os
import traceback
import typing as t

import numpy as np
from tqdm import tqdm

from ares.annotating.annotating_base import ErrorResult, ResultTracker
from ares.annotating.annotating_fn import AnnotatingFn
from ares.annotating.modal_grounding import GroundingModalWrapper
from ares.annotating.orchestration import orchestrate_annotating
from ares.configs.annotations import Annotation
from ares.configs.base import Rollout
from ares.constants import (
    ANNOTATION_GROUNDING_FPS,
    ANNOTATION_OUTER_BATCH_SIZE,
    ARES_DATA_DIR,
)
from ares.databases.annotation_database import ANNOTATION_DB_PATH, AnnotationDatabase
from ares.databases.structured_database import ROBOT_DB_PATH
from ares.models.base import VLM
from ares.models.grounding_utils import get_grounding_nouns_async
from ares.models.refusal import check_refusal
from ares.models.shortcuts import get_gpt_4o
from ares.utils.image_utils import load_video_frames


async def run_annotate_and_ingest(
    annotator: GroundingModalWrapper,
    rollout_ids: list[str],
    annotation_input_futures: list[t.Any],
    db: AnnotationDatabase,
    rollouts: list[t.Any],
) -> tuple[ResultTracker, list[ErrorResult]]:
    """
    Run annotation tasks in parallel using Modal.

    Args:
        annotator (GroundingModalWrapper): Modal wrapper for grounding tasks.
        rollout_ids (list[str]): List of rollout IDs.
        annotation_input_futures (list[t.Any]): List of asyncio Tasks for preparing annotation inputs.
        db (AnnotationDatabase): Database instance for storing annotations.
        rollouts (list[t.Any]): List of rollout objects.

    Returns:
        tuple[ResultTracker, list[ErrorResult]]: Tracker and list of failures.
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
    annotation_results = await annotator.annotate_videos(tasks)
    for rollout_id, all_frame_annotations in annotation_results:
        try:
            rollout = id_to_rollout[rollout_id]
            _, frames, frame_indices, label_str = id_to_annotation_inputs[rollout_id]
            all_frame_annotation_objs = [
                [Annotation(**ann) for ann in frame_annotations]
                for frame_annotations in all_frame_annotations
            ]
            video_id = db.add_video_with_annotations(
                dataset_filename=rollout.dataset_filename,
                video_path=rollout.filename + ".mp4",
                frames=frames,
                frame_indices=frame_indices,
                annotations=all_frame_annotation_objs,
                label_str=label_str,
            )
            tracker.update_via_batch(
                n_videos=1,
                n_frames=len(all_frame_annotations),
                n_annotations=sum(
                    len(frame_annotations)
                    for frame_annotations in all_frame_annotations
                ),
                video_ids=[video_id],
            )
        except Exception as e:
            failures.append(
                ErrorResult(
                    rollout_id=rollout_id,
                    error_pattern="grounding_failure",
                    error=traceback.format_exc(),
                    exception=str(e),
                )
            )

    return tracker, failures


async def setup_query(
    rollout: t.Any,
    vlm: VLM,
    target_fps: int = 5,
) -> tuple[str, list[np.ndarray], list[int], str] | ErrorResult:
    """
    Prepare annotation inputs for a rollout.

    Args:
        rollout (t.Any): Rollout object.
        vlm (VLM): Vision-Language Model instance.
        target_fps (int, optional): Target FPS for frame extraction.

    Returns:
        tuple[str, list[np.ndarray], list[int], str] | dict[str, t.Any]: Prepared data or error dict.
    """
    try:
        frames, frame_indices = load_video_frames(
            rollout.dataset_filename,
            rollout.filename,
            target_fps,
        )
    except Exception as e:
        return ErrorResult(
            rollout_id=str(rollout.id),
            error_pattern="grounding_failure",
            error=traceback.format_exc(),
            exception=str(e),
        )

    try:
        label_str = await get_grounding_nouns_async(
            vlm,
            frames[0],
            rollout.task.language_instruction,
        )
    except Exception as e:
        return ErrorResult(
            rollout_id=str(rollout.id),
            error_pattern="grounding_request_failure",
            error=traceback.format_exc(),
            exception=str(e),
        )

    if check_refusal(label_str):
        return ErrorResult(
            rollout_id=str(rollout.id),
            error_pattern="grounding_request_failure",
            error=f"Refusal phrase triggered: '{label_str}'",
            exception=None,
        )
    return rollout.id, frames, frame_indices, label_str


async def run_ground_and_annotate(
    rollouts: list[t.Any],
    vlm: VLM,
    ann_db: AnnotationDatabase,
    annotator: GroundingModalWrapper,
    target_fps: int = ANNOTATION_GROUNDING_FPS,
) -> tuple[ResultTracker, list[ErrorResult]]:
    """
    Process, ground, and annotate list of rollouts.

    Args:
        rollouts (list[t.Any]): List of rollout objects.
        vlm (VLM): Vision-Language Model instance.
        ann_db (AnnotationDatabase): Annotation database instance.
        target_fps (int, optional): Target FPS for annotation.

    Returns:
        tuple[dict[str, int], list[dict]]: Tracker and list of failures.
    """
    rollout_ids = [r.id for r in rollouts]

    # Create and gather the futures properly
    annotation_input_futures = [
        asyncio.create_task(setup_query(rollout, vlm, target_fps))
        for rollout in rollouts
    ]

    tracker, failures = await run_annotate_and_ingest(
        annotator,
        rollout_ids,
        annotation_input_futures,
        ann_db,
        rollouts,
    )
    return tracker, failures


class GroundingModalAnnotatingFn(AnnotatingFn):
    def __call__(
        self,
        rollouts: list[Rollout],
        ann_db: AnnotationDatabase,
        outer_batch_size: int,
        annotation_fps: int = ANNOTATION_GROUNDING_FPS,
    ) -> tuple[ResultTracker, list[ErrorResult]]:
        """
        Main function to run grounding annotation using Modal.
        """
        # initialize objects for batches
        overall_tracker = ResultTracker()
        overall_failures = []

        annotator = GroundingModalWrapper()
        with annotator.app.run():
            # Limited by CPU RAM (can't create all requests at once)
            for i in tqdm(
                range(0, len(rollouts), outer_batch_size),
                desc="Processing outer batches",
            ):
                print(
                    f"Processing batch {i // outer_batch_size + 1} of {len(rollouts) // outer_batch_size}"
                )
                # create VLM outside async as semaphore gets "bound" to async context
                vlm = get_gpt_4o()
                rollouts_batch = rollouts[i : i + outer_batch_size]
                tracker, failures = asyncio.run(
                    run_ground_and_annotate(
                        rollouts_batch,
                        vlm,
                        ann_db,
                        annotator,
                        annotation_fps,
                    )
                )
                print(
                    f"Completed batch {i // outer_batch_size + 1} of {max(1, len(rollouts) // outer_batch_size)}"
                )
                overall_tracker.update_tracker(tracker)
                overall_failures.extend(failures)
        return overall_tracker, overall_failures


if __name__ == "__main__":
    ids_path = (
        "/workspaces/ares/data/heal_info/2025-01-27_22-04-01/update_grounding_ids.txt"
    )
    orchestrate_annotating(
        engine_path=ROBOT_DB_PATH,
        ann_db_path=ANNOTATION_DB_PATH,
        annotating_fn=GroundingModalAnnotatingFn(),
        ids_path=ids_path,
        outer_batch_size=ANNOTATION_OUTER_BATCH_SIZE,
        annotating_kwargs=dict(
            annotation_fps=ANNOTATION_GROUNDING_FPS,
        ),
        failures_path=os.path.join(
            ARES_DATA_DIR, "annotating_failures", f"grounding_failures.pkl"
        ),
    )
