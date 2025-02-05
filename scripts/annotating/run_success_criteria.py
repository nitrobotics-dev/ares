import os
import traceback

from ares.annotating.annotating_base import ErrorResult, ResultTracker
from ares.annotating.annotating_fn import APIAnnotatingFn
from ares.annotating.orchestration import orchestrate_annotating
from ares.configs.base import Rollout
from ares.constants import ANNOTATION_OUTER_BATCH_SIZE, ARES_DATA_DIR, DATASET_NAMES
from ares.databases.annotation_database import ANNOTATION_DB_PATH, AnnotationDatabase
from ares.databases.structured_database import ROBOT_DB_PATH
from ares.models.base import VLM, parse_response
from ares.utils.image_utils import load_video_frames


class SuccessCriteriaAnnotatingFn(APIAnnotatingFn):
    def __init__(self):
        super().__init__(annotation_key="string", annotation_type="success_criteria")

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
            _, res = await vlm.ask_async(
                info=dict(task=rollout.task.language_instruction),
                prompt_filename="success_constraint_generation.jinja2",
                images=[frames[0]],
            )
            success_criteria = parse_response(res.choices[0], load_json=False)
        except Exception as e:
            return ErrorResult(
                rollout_id=str(rollout.id),
                error_pattern="success_constraint_generation_failure",
                error=traceback.format_exc(),
                exception=str(e),
            )
        return success_criteria


if __name__ == "__main__":
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

    print(f"OVERALL STATS")
    overall_tracker.print_stats()
    print(f"Number of failures: {len(overall_failures)}")
    breakpoint()
