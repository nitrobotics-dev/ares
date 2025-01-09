import asyncio
import pickle
import time
import traceback
import typing as t

import numpy as np
from modal import App, Image, build, enter, method

from ares.models.grounding import GroundingAnnotator

image = (
    Image.debian_slim()
    .apt_install("python3-opencv")
    .pip_install(
        "torch", "transformers", "numpy", "opencv-python", "tqdm", "numpy", "pillow"
    )
)

app = App("ares-grounding-modal", image=image)


@app.cls(image=image, gpu="any")
class ModalWrapper:
    @build()
    @enter()
    def setup(self) -> None:
        self.annotator = GroundingAnnotator(segmenter_id=None)

    @method()
    def annotate_video(self, rollout_id, frames, label_str):
        # Convert frames from list of lists back to numpy arrays and ensure uint8 type
        frames = [np.array(f, dtype=np.uint8) for f in frames]
        return self.annotator.annotate_video(rollout_id, frames, label_str)


async def run_annotation_parallel(
    annotator: ModalWrapper,
    rollout_ids: list[str],
    annotation_input_futures: list[t.Any],
    db: "AnnotationDatabase",
    rollouts: list["Rollout"],
    dataset_file_name: str,
) -> dict[str, int]:
    id_to_rollout = {r.id: r for r in rollouts}
    id_to_annotation_inputs = dict()

    # as the futures get completed, launch the annotate tasks
    tasks = []
    for future in asyncio.as_completed(annotation_input_futures):
        rollout_id, frames, frame_indices, label_str = await future
        print(
            f"received {rollout_id}: {len(frames)} frames, {len(frame_indices)} frame indices, label str: {label_str}"
        )
        id_to_annotation_inputs[rollout_id] = (
            rollout_id,
            frames,
            frame_indices,
            label_str,
        )
        # create annotation task for each rollout with the label_str from the future
        tasks.append(
            asyncio.create_task(
                annotator.annotate_video.remote.aio(rollout_id, frames, label_str)
            )
        )

    # Process and store results as they complete
    tracker = dict(videos=0, frames=0, annotations=0, video_ids=[])
    for coro in asyncio.as_completed(tasks):
        try:
            rollout_id, all_frame_annotation_dicts = await coro
            rollout = id_to_rollout[rollout_id]
            _, frames, frame_indices, label_str = id_to_annotation_inputs[rollout_id]
            tracker["videos"] += 1
            tracker["frames"] += len(all_frame_annotation_dicts)
            tracker["annotations"] += sum(
                len(frame_annotations)
                for frame_annotations in all_frame_annotation_dicts
            )
            # add to database
            video_id = db.add_video_with_annotations(
                dataset_name=dataset_file_name,
                video_path=rollout.path,
                frames=frames,
                frame_indices=frame_indices,
                annotations=all_frame_annotation_dicts,
                label_str=label_str,
            )
            tracker["video_ids"].append(video_id)
        except Exception as e:
            print(f"Error processing task: {e}; {traceback.format_exc()}")
    return tracker


# test remote modal
@app.local_entrypoint()
def test() -> None:
    # do whatever local test
    import numpy as np

    from ares.configs.base import Rollout
    from ares.databases.annotation_database import (
        TEST_ANNOTATION_DB_PATH,
        AnnotationDatabase,
    )
    from ares.databases.structured_database import (
        TEST_ROBOT_DB_PATH,
        RolloutSQLModel,
        setup_database,
        setup_rollouts,
    )
    from ares.models.base import VLM
    from ares.models.grounding_utils import get_grounding_nouns_async
    from ares.models.shortcuts import get_gpt_4o
    from ares.utils.image_utils import load_video_frames

    async def setup_query(
        dataset_name: str,
        rollout: Rollout,
        vlm: VLM,
        target_fps: int = 5,
    ) -> tuple[str, list[np.ndarray], list[int], str]:
        frames, frame_indices = load_video_frames(
            dataset_name,
            rollout.path,
            target_fps,
        )
        label_str = await get_grounding_nouns_async(
            vlm,
            frames[0],
            rollout.task.language_instruction,
        )
        return rollout.id, frames, frame_indices, label_str

    async def run_ground_and_annotate(
        dataset_file_name: str,
        rollouts: list[Rollout],
        vlm: VLM,
        ann_db: AnnotationDatabase,
        target_fps: int = 5,
    ):
        rollout_ids = [r.id for r in rollouts]

        # Create and gather the futures properly
        annotation_input_futures = [
            asyncio.create_task(
                setup_query(dataset_file_name, rollout, vlm, target_fps)
            )
            for rollout in rollouts
        ]

        annotator = ModalWrapper()
        stats = await run_annotation_parallel(
            annotator,
            rollout_ids,
            annotation_input_futures,
            ann_db,
            rollouts,
            dataset_file_name,
        )
        return stats

    formal_dataset_name = "CMU Stretch"
    dataset_file_name = "cmu_stretch"
    target_fps = 5

    ann_db = AnnotationDatabase(connection_string=TEST_ANNOTATION_DB_PATH)
    engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)
    rollouts = setup_rollouts(engine, formal_dataset_name)[100:]
    vlm = get_gpt_4o()
    tic = time.time()

    stats = asyncio.run(
        run_ground_and_annotate(
            dataset_file_name,
            rollouts,
            vlm,
            ann_db,
            target_fps,
        )
    )

    print("time taken", time.time() - tic)
    print(f"\n\nstats: {stats}\n\n")
    breakpoint()
