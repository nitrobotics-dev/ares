"""
Orchestration script to run a simpler version of the 'Embodied Chain of Thought' paper. 
See original code https://github.com/MichalZawalski/embodied-CoT/blob/main/scripts/generate_embodied_data/full_reasonings.py

We utilize the `grounding_string`, `detections`, and `success_criteria` annotations + rollout fields to generate a pseudo-ECoT in a similar fashion.
"""

import asyncio
import os
import traceback
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from ares.configs.annotations import Annotation
from ares.configs.base import Rollout
from ares.constants import ANNOTATION_OUTER_BATCH_SIZE, ARES_DATA_DIR
from ares.databases.annotation_database import (
    ANNOTATION_DB_PATH,
    AnnotationDatabase,
    get_video_id,
)
from ares.databases.structured_database import ROBOT_DB_PATH, RolloutSQLModel
from ares.models.base import VLM, parse_response
from ares.models.shortcuts import get_vlm
from ares.utils.image_utils import load_video_frames

from .annotating_base import (
    AnnotatingFn,
    ErrorResult,
    ResultTracker,
    orchestrate_annotating,
)


def construct_pseudo_ecot_info(rollout: Rollout, ann_db: AnnotationDatabase):
    # we want the grounding string, detections, and success criteria
    video_id = get_video_id(rollout.dataset_filename, rollout.filename)
    anns = ann_db.get_annotations(video_id)

    # get the annotations from the database
    grounding_string = anns.get("grounding_string")[0].description
    grounding_string = ", ".join(grounding_string.split(".")[:-1]) + "."
    success_criteria = anns.get("success_criteria")[0].description
    detections = anns.get("detection")

    # separate and parse the detections into a string
    frame_0_detections: list[Annotation] = detections[0] if detections else []
    if frame_0_detections:
        text_detections = ", ".join(
            f"{d.category_name} at {[round(r, 2) for r in d.bbox]} (LTRB format)"
            for d in frame_0_detections
        )
    else:
        text_detections = None
    return dict(
        task=rollout.task.language_instruction,
        complexity_category_estimate=rollout.task.complexity_category_estimate,
        grounding_string=grounding_string,
        detections=text_detections,
        success_criteria=success_criteria,
    )


class PseudoECoTAnnotatingFn(AnnotatingFn):
    async def run_query(self, vlm: VLM, rollout: Rollout, ann_db: AnnotationDatabase):
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
            info = construct_pseudo_ecot_info(rollout, ann_db)
            messages, res = await vlm.ask_async(
                info=info,
                prompt_filename="pseudo_ecot.jinja2",
                images=[frames[0]],
            )
        except Exception as e:
            return ErrorResult(
                rollout_id=rollout.id,
                error_pattern="pseudo_ecot_failure",
                error=traceback.format_exc(),
            )
        try:
            # just a string, so use the default
            breakpoint()
            pseudo_ecot_str = parse_response(res.choices[0], load_json=False)
        except Exception as e:
            return ErrorResult(
                rollout_id=rollout.id,
                error_pattern="pseudo_ecot_parsing_failure",
                error=traceback.format_exc(),
            )
        return pseudo_ecot_str

    async def run_batch(
        self, vlm: VLM, rollouts_batch: List[Rollout], ann_db: AnnotationDatabase
    ) -> Tuple[ResultTracker, List[ErrorResult]]:
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
                    failures.append(result)
                else:
                    video_id = get_video_id(rollout.dataset_filename, rollout.filename)
                    # Add success criteria annotation to database
                    ann_db.add_annotation(
                        video_id=video_id,
                        key="string",
                        value=Annotation(
                            description=result, annotation_type="pseudo_ecot"
                        ).to_dict(),
                        annotation_type="pseudo_ecot",
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

        # HACK FIXME
        rollouts = rollouts[:2]

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
            annotating_fn=PseudoECoTAnnotatingFn(),
            dataset_filename=dataset_filename,
            outer_batch_size=ANNOTATION_OUTER_BATCH_SIZE,
            failures_path=os.path.join(
                ARES_DATA_DIR,
                "annotating_failures",
                f"pseudo_ecot_failures_{dataset_filename}.pkl",
            ),
        )
        overall_tracker.update_tracker(tracker)
        overall_failures.extend(failures)

    print(f"OVERALL STATS")
    overall_tracker.print_stats()
    print(f"Number of failures: {len(overall_failures)}")
    breakpoint()
