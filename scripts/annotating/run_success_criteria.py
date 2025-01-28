from typing import List, Tuple

from ares.configs.base import Rollout
from ares.constants import ANNOTATION_OUTER_BATCH_SIZE
from ares.databases.annotation_database import ANNOTATION_DB_PATH, AnnotationDatabase
from ares.databases.structured_database import ROBOT_DB_PATH, RolloutSQLModel
from ares.models.base import VLM
from ares.models.grounding import ANNOTATION_GROUNDING_FPS
from ares.models.shortcuts import get_gpt_4o

from .annotation_base import (
    AnnotatingFn,
    ErrorResult,
    ResultTracker,
    orchestrate_annotating,
)


class SuccessCriteriaAnnotatingFn(AnnotatingFn):
    def __call__(
        self,
        rollouts: List[Rollout],
        ann_db: AnnotationDatabase,
        outer_batch_size: int,
        annotation_fps: int,
    ) -> Tuple[ResultTracker, List[ErrorResult]]:
        pass


if __name__ == "__main__":
    ids_path = None
    orchestrate_annotating(
        engine_path=ROBOT_DB_PATH,
        ann_db_path=ANNOTATION_DB_PATH,
        annotating_fn=SuccessCriteriaAnnotatingFn(),
        ids_path=ids_path,
        outer_batch_size=ANNOTATION_OUTER_BATCH_SIZE,
    )
