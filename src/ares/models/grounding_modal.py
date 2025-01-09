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
    annotation_inputs: list[t.Any],
    db: "AnnotationDatabase",
    rollouts: list["Rollout"],
    dataset_file_name: str,
) -> dict[str, int]:
    id_to_rollout = {r.id: r for r in rollouts}
    id_to_inputs = {
        r_id: (r_id, frames, frame_indices, label_str)
        for r_id, (frames, frame_indices, label_str) in zip(
            rollout_ids, annotation_inputs
        )
    }

    # Create coroutines with their metadata
    tasks = [
        asyncio.create_task(
            annotator.annotate_video.remote.aio(rollout_id, frames, label_str)
        )
        for rollout_id, (_, frames, frame_indices, label_str) in id_to_inputs.items()
    ]

    # Process and store results as they complete
    tracker = dict(videos=0, frames=0, annotations=0, video_ids=[])
    for coro in asyncio.as_completed(tasks):
        try:
            rollout_id, all_frame_annotation_dicts = await coro
            rollout = id_to_rollout[rollout_id]
            _, frames, frame_indices, label_str = id_to_inputs[rollout_id]
            tracker["videos"] += 1
            tracker["frames"] += len(all_frame_annotation_dicts)
            tracker["annotations"] += sum(
                len(frame_annotations)
                for frame_annotations in all_frame_annotation_dicts
            )
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
    from run_grounding_annotations import setup_query, setup_rollouts

    formal_dataset_name = "CMU Stretch"
    dataset_file_name = "cmu_stretch"
    target_fps = 5

    ann_db = AnnotationDatabase(connection_string=TEST_ANNOTATION_DB_PATH)
    engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)
    rollouts = setup_rollouts(engine, formal_dataset_name)

    annotator = ModalWrapper()
    rollout_ids, annotation_inputs = pickle.load(open("label_results.pkl", "rb"))
    print(f"loading {len(annotation_inputs)} annotations")
    rollout_ids = rollout_ids[:5]
    annotation_inputs = annotation_inputs[:5]
    tic = time.time()
    stats = asyncio.run(
        run_annotation_parallel(
            annotator,
            rollout_ids,
            annotation_inputs,
            ann_db,
            rollouts,
            dataset_file_name,
        )
    )
    print("time taken", time.time() - tic)
    print(stats)
    breakpoint()
