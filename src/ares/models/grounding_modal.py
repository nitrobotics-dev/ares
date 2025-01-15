import asyncio
import pickle
import time
import traceback
import typing as t

import numpy as np
from modal import App, Image, build, enter, method

from ares.models.grounding import ANNOTATION_GROUNDING_FPS, GroundingAnnotator

image = (
    Image.debian_slim()
    .apt_install("python3-opencv")
    .pip_install(
        "torch", "transformers", "numpy", "opencv-python", "tqdm", "numpy", "pillow"
    )
)

app = App("ares-grounding-modal", image=image)
MODAL_CONCURRENCY_LIMIT = 10
MODAL_TIMEOUT = 600


@app.cls(
    image=image,
    gpu="t4",
    concurrency_limit=MODAL_CONCURRENCY_LIMIT,
    timeout=MODAL_TIMEOUT,
)
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
) -> tuple[dict[str, int], list[dict]]:
    id_to_rollout = {r.id: r for r in rollouts}
    id_to_annotation_inputs = dict()

    # as the futures get completed, launch the annotate tasks
    tasks = []
    failures = []
    for future in asyncio.as_completed(annotation_input_futures):
        res = await future
        if isinstance(res, dict):
            # failure!
            failures.append(res)
            continue
        else:
            rollout_id, frames, frame_indices, label_str = res
        print(
            f"received grounding output/ annotation input for {rollout_id}: {len(frames)} frames, {len(frame_indices)} frame indices, label str: {label_str}"
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
        print(f"launched modal req for rollout id {rollout_id}")

    # Process and store results as they complete
    tracker = dict(videos=0, frames=0, annotations=0, video_ids=[])
    for coro in asyncio.as_completed(tasks):
        try:
            rollout_id, all_frame_annotation_dicts = await coro
        except Exception as e:
            print(f"Error processing task: {e}")
            failures.append(
                {
                    "rollout_id": rollout_id,
                    "error_pattern": "grounding_failure",
                    "error": traceback.format_exc(),
                }
            )
            continue

        try:
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
                dataset_filename=dataset_file_name,
                video_path=rollout.filename + ".mp4",
                frames=frames,
                frame_indices=frame_indices,
                annotations=all_frame_annotation_dicts,
                label_str=label_str,
            )
            tracker["video_ids"].append(video_id)
        except Exception as e:
            print(f"Error processing task: {e}; {traceback.format_exc()}")
            failures.append(
                {
                    "rollout_id": rollout_id,
                    "error_pattern": "",
                    "traceback": traceback.format_exc(),
                }
            )
    return tracker, failures


# test remote modal
@app.local_entrypoint()
def test() -> None:
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
    from ares.name_remapper import DATASET_NAMES
    from ares.utils.image_utils import load_video_frames

    async def setup_query(
        dataset_filename: str,
        rollout: Rollout,
        vlm: VLM,
        target_fps: int = 5,
        refusal_phrases: list[str] | None = None,
    ) -> tuple[str, list[np.ndarray], list[int], str] | dict[str, Rollout | str]:
        # Define local async functions here to use ARES imports in the Modal container

        try:
            frames, frame_indices = load_video_frames(
                dataset_filename,
                rollout.filename,
                target_fps,
            )
            print(f"loaded {len(frames)} frames of size {frames[0].shape}")
        except Exception as e:
            return {
                "rollout_id": rollout.id,
                "error_pattern": "loading_failure",
                "error": e,
            }

        try:
            label_str = await get_grounding_nouns_async(
                vlm,
                frames[0],
                rollout.task.language_instruction,
            )
        except Exception as e:
            return {
                "rollout_id": rollout.id,
                "error_pattern": "grounding_request_failure",
                "error": e,
            }

        refusal_phrases = refusal_phrases or ["I'm"]
        if any(phrase in label_str for phrase in refusal_phrases):
            return {
                "rollout_id": rollout.id,
                "error_pattern": "refusal",
                "error": label_str,
            }
        return rollout.id, frames, frame_indices, label_str

    async def run_ground_and_annotate(
        dataset_filename: str,
        rollouts: list[Rollout],
        vlm: VLM,
        ann_db: AnnotationDatabase,
        target_fps: int = 5,
    ) -> tuple[dict, list[dict]]:
        rollout_ids = [r.id for r in rollouts]

        # Create and gather the futures properly
        annotation_input_futures = [
            asyncio.create_task(setup_query(dataset_filename, rollout, vlm, target_fps))
            for rollout in rollouts
        ]

        annotator = ModalWrapper()
        stats, failures = await run_annotation_parallel(
            annotator,
            rollout_ids,
            annotation_input_futures,
            ann_db,
            rollouts,
            dataset_filename,
        )
        return stats, failures

    dataset_info = DATASET_NAMES[1]
    dataset_filename = dataset_info["dataset_filename"]
    dataset_formalname = dataset_info["dataset_formalname"]

    retry_failed = None  # path to pickle file with failures to retry
    # retry_failed = "failures.pkl"

    ann_db = AnnotationDatabase(connection_string=TEST_ANNOTATION_DB_PATH)
    engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)
    rollouts = setup_rollouts(engine, dataset_formalname)
    if len(rollouts) == 0:
        breakpoint()

    if retry_failed:
        with open(retry_failed, "rb") as f:
            failures = pickle.load(f)

        failed_ids = [str(f["rollout_id"]) for f in failures]
        rollouts = [r for r in rollouts if str(r.id) in failed_ids]

    print(f"\n\nfound {len(rollouts)} rollouts\n\n")
    vlm = get_gpt_4o()
    tic = time.time()

    stats, failures = asyncio.run(
        run_ground_and_annotate(
            dataset_filename,
            rollouts,
            vlm,
            ann_db,
            ANNOTATION_GROUNDING_FPS,
        )
    )

    print(f"\n\nfailures: {failures}\n\n")

    # write failures to file in order to retry
    with open("failures.pkl", "wb") as f:
        pickle.dump(failures, f)

    print("time taken", time.time() - tic)
    print(f"\n\n")
    for k, v in stats.items():
        print(f"{k}: {v}" if not isinstance(v, list) else f"{k}: {v[:10]}...")
    print(f"n fails: {len(failures)}")
    breakpoint()
