import asyncio
import time
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

stub = Stub("ares-grounding-modal")


@stub.cls(image=image)
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
    id_to_rollout = {rollout.id: rollout for rollout in rollouts}
    failed_rollouts = []

    # Start all setup_query tasks with error handling
    setup_futures = {}
    for rollout_id, rollout in id_to_rollout.items():
        try:
            setup_futures[rollout_id] = setup_query(
                dataset_name, rollout, vlm, target_fps
            )
        except Exception as e:
            failed_rollouts.append(rollout_id)
    rev_setup_futures = {v: k for k, v in setup_futures.items()}
    # Process setup results and start annotations
    annotation_futures = {}
    future_to_metadata = {}

    # As setup queries complete, start annotations
    for setup_future in asyncio.as_completed(setup_futures.values()):
        try:
            rollout_id = rev_setup_futures[setup_future]
            frames, frame_indices, label_str = await setup_future

            # Start annotation for this result
            ann_future = annotator.annotate_video.remote(frames, label_str)
            annotation_futures[rollout_id] = ann_future
            future_to_metadata[rollout_id] = (
                frames,
                frame_indices,
                label_str,
                id_to_rollout[rollout_id].path,
            )
        except Exception as e:
            failed_rollouts.append(rollout_id)
    rev_annotation_futures = {v: k for k, v in annotation_futures.items()}

    # Process annotation results as they complete
    for ann_future in as_completed(list(annotation_futures.values())):
        try:
            rollout_id = rev_annotation_futures[ann_future]
            frames, frame_indices, label_str, video_path = future_to_metadata[
                rollout_id
            ]
            detection_results = await ann_future
            video_annotations = convert_to_annotations(detection_results)

            total_anns += sum(len(anns) for anns in video_annotations)
            total_frames += len(frames)

            video_id = ann_db.add_video_with_annotations(
                dataset_name=dataset_name,
                video_path=video_path,
                frames=frames,
                frame_indices=frame_indices,
                annotations=video_annotations,
                label_str=label_str,
            )
            total_processed += 1
        except Exception as e:
            failed_rollouts.append(rollout_id)

    if failed_rollouts:
        print(f"Failed rollouts: {failed_rollouts}")

    return total_anns, total_processed, total_frames


if __name__ == "__main__":
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
    from ares.models.grounding import get_grounding_nouns
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
    total_anns, total_processed, total_frames = asyncio.run(
        run_modal(rollouts, vlm, annotator, ann_db, dataset_name, target_fps)
    )
    print(f"Total annotations: {total_anns}")
    print(f"Total frames: {total_frames}")
    print(f"Total processed: {total_processed}")
