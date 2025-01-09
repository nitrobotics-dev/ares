import asyncio
import time
import traceback
from concurrent.futures import as_completed

import numpy as np
from modal import Image, Stub, method

from ares.configs.annotations import Annotation
from ares.configs.base import Rollout
from ares.databases.annotation_database import AnnotationDatabase
from ares.models.base import VLM
from ares.models.grounding import GroundingAnnotator, convert_to_annotations
from run_grounding_annotations import setup_query

# Create image with necessary dependencies
image = (
    Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
    )
    .apt_install(["libgl1"])  # If you need OpenCV
)

# Create Modal app instance
app = Stub("ares-grounding-modal")


@app.cls(image=image)
class ModalWrapper:
    def __enter__(self) -> None:
        self.annotator = GroundingAnnotator(segmenter_id=None)

    @method()
    def annotate_video(
        self, frames: list[np.ndarray], label_str: str
    ) -> list[list[dict]]:
        return self.annotator.annotate_video(frames, label_str)


async def run_modal(
    rollouts: list[Rollout],
    vlm: VLM,
    annotator: ModalWrapper,
    ann_db: AnnotationDatabase,
    dataset_name: str,
    target_fps: int,
) -> tuple[int, int, int]:
    total_anns = 0
    total_processed = 0
    total_frames = 0
    failed_rollouts = []

    # Create tasks for all rollouts
    setup_tasks = []
    rollout_ids = []
    for rollout in rollouts:
        try:
            task = setup_query(dataset_name, rollout, vlm, target_fps)
            setup_tasks.append(task)
            rollout_ids.append(rollout.id)
        except Exception as e:
            failed_rollouts.append(
                ("setup futures", rollout.id, e, traceback.format_exc())
            )

    # Process all setup tasks concurrently using gather
    try:
        results = await asyncio.gather(*setup_tasks, return_exceptions=True)
        for rollout_id, result in zip(rollout_ids, results):
            if isinstance(result, Exception):
                failed_rollouts.append(
                    ("setup gather", rollout_id, result, traceback.format_exc())
                )
                continue

            frames, frame_indices, label_str = result
            print(label_str)
            # Start annotation for this result
            # ann_future = annotator.annotate_video.remote(frames, label_str)
            # annotation_futures[rollout_id] = ann_future
            # future_to_metadata[rollout_id] = (
            #     frames,
            #     frame_indices,
            #     label_str,
            #     id_to_rollout[rollout_id].path,
            # )
    except Exception as e:
        failed_rollouts.append(("gather", "all", e, traceback.format_exc()))

    return failed_rollouts


def run() -> None:
    from ares.databases.annotation_database import (
        TEST_ANNOTATION_DB_PATH,
        AnnotationDatabase,
    )
    from ares.databases.structured_database import (
        TEST_ROBOT_DB_PATH,
        RolloutSQLModel,
        get_rollout_by_name,
        setup_database,
    )
    from ares.models.shortcuts import get_gemini_2_flash, get_gpt_4o
    from run_grounding_annotations import setup_query, setup_rollouts

    formal_dataset_name = "CMU Stretch"
    dataset_name = "cmu_stretch"
    target_fps = 5

    ann_db = AnnotationDatabase(connection_string=TEST_ANNOTATION_DB_PATH)
    engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)

    rollouts = setup_rollouts(engine, formal_dataset_name)[:5]
    print(f"Len Rollouts: {len(rollouts)}")
    vlm = get_gpt_4o()
    annotator = ModalWrapper()

    tic = time.time()
    # total_anns, total_processed, total_frames = asyncio.run(
    out = asyncio.run(
        run_modal(rollouts, vlm, annotator, ann_db, dataset_name, target_fps)
    )
    breakpoint()
    print(f"Total annotations: {total_anns}")
    print(f"Total frames: {total_frames}")
    print(f"Total processed: {total_processed}")
    breakpoint()


if __name__ == "__main__":
    run()
