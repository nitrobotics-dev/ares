import asyncio
import os
import traceback
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from ares.configs.base import Rollout
from ares.constants import ANNOTATION_OUTER_BATCH_SIZE, ARES_DATA_DIR
from ares.databases.annotation_database import ANNOTATION_DB_PATH, AnnotationDatabase
from ares.databases.structured_database import ROBOT_DB_PATH, RolloutSQLModel
from ares.models.base import VLM, parse_response
from ares.models.grounding import ANNOTATION_GROUNDING_FPS
from ares.models.shortcuts import get_vlm
from ares.utils.image_utils import load_video_frames

from .annotation_base import (
    AnnotatingFn,
    ErrorResult,
    ResultTracker,
    orchestrate_annotating,
)


class SuccessCriteriaAnnotatingFn(AnnotatingFn):
    async def run_query(self, vlm: VLM, rollout: Rollout):
        try:
            frames, frame_indices = load_video_frames(
                rollout.dataset_filename,
                rollout.filename,
                target_fps=0,
            )
        except Exception as e:
            return ErrorResult(
                rollout_id=rollout.id,
                error_pattern="loading_video_failure",
                error=traceback.format_exc(),
            )
        try:
            messages, res = await vlm.ask_async(
                info=dict(task=rollout.task.language_instruction),
                prompt_filename="success_constraint_generation.jinja2",
                images=[frames[0]],
            )
        except Exception as e:
            return ErrorResult(
                rollout_id=rollout.id,
                error_pattern="success_constraint_generation_failure",
                error=traceback.format_exc(),
            )
        try:
            # just a string, so use the default
            success_criteria = parse_response(res.choices[0], load_json=False)
        except Exception as e:
            return ErrorResult(
                rollout_id=rollout.id,
                error_pattern="success_constraint_generation_failure",
                error=traceback.format_exc(),
            )
        return success_criteria

    async def run_batch(
        self, vlm: VLM, rollouts_batch: List[Rollout], ann_db: AnnotationDatabase
    ) -> Tuple[ResultTracker, List[ErrorResult]]:
        # Create futures with their corresponding rollouts
        futures = []
        for rollout in rollouts_batch:
            future = asyncio.create_task(self.run_query(vlm, rollout))
            futures.append((future, rollout))

        tracker = ResultTracker()
        failures = []

        for future, rollout in futures:
            try:
                result = await future
                if isinstance(result, ErrorResult):
                    failures.append(result)
                else:
                    video_id = f"{rollout.dataset_filename}/{Path(rollout.filename).with_suffix('.mp4')}"
                    # Add success criteria annotation to database
                    ann_db.add_annotation(
                        video_id=video_id,
                        key="string",
                        value=result,
                        annotation_type="success_criteria",
                        frame=None,
                    )
                    tracker.update_via_batch(
                        n_videos=1, n_frames=1, n_annotations=1, video_ids=[video_id]
                    )
            except Exception as e:
                failures.append(
                    ErrorResult(
                        rollout_id=rollout.id,
                        error_pattern="batch_processing_failure",
                        error=traceback.format_exc(),
                    )
                )

        return tracker, failures

    def __call__(
        self,
        rollouts: List[Rollout],
        ann_db: AnnotationDatabase,
        outer_batch_size: int,
        vlm_name: str = "gpt-4o-mini",
    ) -> Tuple[ResultTracker, List[ErrorResult]]:
        overall_tracker = ResultTracker()
        overall_failures = []

        # HACK
        rollouts = rollouts[10:]

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
                self.run_batch(vlm=vlm, rollouts_batch=rollouts_batch, ann_db=ann_db)
            )

            print(
                f"Completed batch {i // outer_batch_size + 1} of {max(1, len(rollouts) // outer_batch_size)}"
            )
            overall_tracker.update_tracker(tracker)
            overall_failures.extend(failures)
        return overall_tracker, overall_failures


if __name__ == "__main__":
    from ares.constants import DATASET_NAMES

    overall_tracker = ResultTracker()
    overall_failures = []

    for dataset_info in DATASET_NAMES:
        print(f"Processing {dataset_info['dataset_formalname']}")
        dataset_filename = dataset_info["dataset_filename"]
        tracker, failures = orchestrate_annotating(
            engine_path=ROBOT_DB_PATH,
            ann_db_path=ANNOTATION_DB_PATH,
            annotating_fn=SuccessCriteriaAnnotatingFn(),
            dataset_filename=dataset_filename,
            outer_batch_size=ANNOTATION_OUTER_BATCH_SIZE,
            failures_path=os.path.join(
                ARES_DATA_DIR,
                "annotating_failures",
                f"success_criteria_failures_{dataset_filename}.pkl",
            ),
        )
        overall_tracker.update_tracker(tracker)
        overall_failures.extend(failures)

    overall_tracker.print_stats()
    print(f"Number of failures: {len(overall_failures)}")
    breakpoint()
