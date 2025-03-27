"""
Orchestration script to run a simpler version of the 'Embodied Chain of Thought' paper.
See original code https://github.com/MichalZawalski/embodied-CoT/blob/main/scripts/generate_embodied_data/full_reasonings.py

We utilize the `grounding_string`, `detections`, and `success_criteria` annotations (see other annotating scripts!) + rollout fields to generate a pseudo-ECoT in a similar fashion.
"""

import os
import traceback
import typing as t

from ares.annotating.annotating_base import ErrorResult, ResultTracker
from ares.annotating.annotating_fn import APIAnnotatingFn
from ares.annotating.orchestration import orchestrate_annotating
from ares.configs.annotations import Annotation
from ares.configs.base import Rollout
from ares.constants import ANNOTATION_OUTER_BATCH_SIZE, ARES_DATA_DIR, DATASET_NAMES
from ares.databases.annotation_database import (
    ANNOTATION_DB_PATH,
    AnnotationDatabase,
    get_video_id,
)
from ares.databases.structured_database import ROBOT_DB_PATH
from ares.models.base import VLM, parse_response
from ares.utils.image_utils import load_video_frames


def construct_pseudo_ecot_info(
    rollout: Rollout, ann_db: AnnotationDatabase
) -> dict[str, t.Any]:
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


class PseudoECoTAnnotatingFn(APIAnnotatingFn):
    def __init__(self) -> None:
        super().__init__(annotation_key="string", annotation_type="pseudo_ecot")

    async def run_query(
        self, vlm: VLM, rollout: Rollout, ann_db: AnnotationDatabase
    ) -> str | ErrorResult:
        try:
            frames, _ = load_video_frames(
                rollout.dataset_filename,
                rollout.filename,
                target_fps=0,
            )
        except Exception as e:
            return ErrorResult(
                rollout_id=str(rollout.id),
                error_pattern="loading_video_failure",
                error=traceback.format_exc(),
                exception=str(e),
            )
        try:
            info = construct_pseudo_ecot_info(rollout, ann_db)
            messages, res = await vlm.ask_async(
                info=info,
                prompt_filename="pseudo_ecot.jinja2",
                images=[frames[0]],
            )
            pseudo_ecot_str = parse_response(res.choices[0], load_json=False)
        except Exception as e:
            return ErrorResult(
                rollout_id=str(rollout.id),
                error_pattern="pseudo_ecot_failure",
                error=traceback.format_exc(),
                exception=str(e),
            )
        return pseudo_ecot_str


if __name__ == "__main__":
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
