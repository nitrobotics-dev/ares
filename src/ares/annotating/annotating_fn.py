import asyncio
import traceback
import typing as t

from tqdm import tqdm

from ares.annotating.annotating_base import ErrorResult, ResultTracker
from ares.configs.annotations import Annotation
from ares.configs.base import Rollout
from ares.databases.annotation_database import AnnotationDatabase, get_video_id
from ares.models.base import VLM
from ares.models.shortcuts import get_vlm


class AnnotatingFn:
    """
    Compute primitive to annotate rollouts. AnnotatingFns should conduct their annotation in batches,
    whether via API, Modal, local, etc.
    """

    def __call__(
        self,
        rollouts: list[Rollout],
        ann_db: AnnotationDatabase,
        outer_batch_size: int,
        **kwargs: t.Any,
    ) -> tuple[ResultTracker, list[ErrorResult]]:
        raise NotImplementedError


class APIAnnotatingFn(AnnotatingFn):
    """
    Base class to create annotating functions that use an API. E.g success criteria, grounding phrases, etc.
    """

    def __init__(self, annotation_key: str, annotation_type: str):
        self.annotation_key = annotation_key
        self.annotation_type = annotation_type

    async def run_query(
        self,
        vlm: VLM,
        rollout: Rollout,
        ann_db: AnnotationDatabase,
    ) -> t.Any:
        raise NotImplementedError

    async def run_batch(
        self,
        vlm: VLM,
        rollouts_batch: list[Rollout],
        ann_db: AnnotationDatabase,
    ) -> tuple[ResultTracker, list[ErrorResult]]:
        """
        Default function to annotate a batch of rollouts and store annotations in the database with an API-driven annotating function.
        """

        # Create futures with their corresponding rollouts
        futures = []
        for rollout in rollouts_batch:
            future = asyncio.create_task(self.run_query(vlm, rollout, ann_db))
            futures.append((future, rollout))

        tracker = ResultTracker()
        failures = []

        for future, rollout in futures:
            try:
                result = await future
                if isinstance(result, ErrorResult):
                    # check for error result and record if so
                    failures.append(result)
                else:
                    # otherwise, get the video id and add the annotaiton to the database
                    video_id = get_video_id(rollout.dataset_filename, rollout.filename)
                    ann_db.add_annotation(
                        video_id=video_id,
                        key=self.annotation_key,
                        value=Annotation(
                            description=result, annotation_type=self.annotation_type
                        ),
                        annotation_type=self.annotation_type,
                        frame=None,
                    )
                    tracker.update_via_batch(
                        n_videos=1, n_frames=1, n_annotations=1, video_ids=[video_id]
                    )
            except Exception as e:
                failures.append(
                    ErrorResult(
                        rollout_id=str(rollout.id),
                        error_pattern="batch_processing_failure",
                        error=traceback.format_exc(),
                        exception=str(e),
                    )
                )

        return tracker, failures

    def __call__(
        self,
        rollouts: list[Rollout],
        ann_db: AnnotationDatabase,
        outer_batch_size: int,
        vlm_name: str = "gpt-4o-mini",
    ) -> tuple[ResultTracker, list[ErrorResult]]:
        """
        Orchestrating function for this annotating function. The __call__ function instantiates the objects and
        create the "outer loop" for annotating batches of rollouts.
        """
        overall_tracker = ResultTracker()
        overall_failures = []

        for i in tqdm(
            range(0, len(rollouts), outer_batch_size),
            desc="Processing outer batches",
        ):
            print(
                f"Processing batch {i // outer_batch_size + 1} of {max(1, len(rollouts) // outer_batch_size)}"
            )
            # create VLM outside async as the semaphore gets "bound" to async context
            vlm = get_vlm(vlm_name)

            # get batch results
            rollouts_batch = rollouts[i : i + outer_batch_size]
            tracker, failures = asyncio.run(
                self.run_batch(
                    vlm=vlm,
                    rollouts_batch=rollouts_batch,
                    ann_db=ann_db,
                )
            )

            print(
                f"Completed batch {i // outer_batch_size + 1} of {max(1, len(rollouts) // outer_batch_size)}"
            )
            overall_tracker.update_tracker(tracker)
            overall_failures.extend(failures)
        return overall_tracker, overall_failures
