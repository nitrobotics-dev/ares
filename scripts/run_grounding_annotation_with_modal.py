"""
Helper script to run annotation predictions. We adopt Modal as the serverless compute provider in order to launch a large number of requests in parallel.
"""

import asyncio
import os
import pickle
import time
import traceback
import typing as t

import numpy as np
from modal import App, Image, build, enter, method

from ares.constants import ARES_DATA_DIR, OUTER_BATCH_SIZE
from ares.models.grounding import ANNOTATION_GROUNDING_FPS, GroundingAnnotator

FAILURES_PATH = os.path.join(ARES_DATA_DIR, "failures.pkl")

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
                dataset_filename=rollout.dataset_filename,
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


@app.local_entrypoint()
def run_modal_grounding(
    engine_path: str,
    ann_db_path: str,
    dataset_filename: str | None = None,
    split: str | None = None,
    rollout_ids: list[str] | None = None,
    outer_batch_size: int = OUTER_BATCH_SIZE,  # RAM limits number of concurrent rollouts formatted into requests
    retry_failed_path: str = None,  # path to pickle file with failures to retry
    annotation_fps: int = ANNOTATION_GROUNDING_FPS,
) -> None:
    assert (
        dataset_filename is not None or retry_failed_path is not None
    ), f"must provide either dataset_filename or retry_failed_path. Received: dataset_filename={dataset_filename}, retry_failed_path={retry_failed_path}"

    import numpy as np

    from ares.configs.base import Rollout
    from ares.constants import DATASET_NAMES
    from ares.databases.annotation_database import AnnotationDatabase
    from ares.databases.structured_database import (
        RolloutSQLModel,
        get_rollouts_by_ids,
        setup_database,
        setup_rollouts,
    )
    from ares.models.base import VLM
    from ares.models.grounding_utils import get_grounding_nouns_async
    from ares.models.shortcuts import get_gpt_4o
    from ares.utils.image_utils import load_video_frames

    async def setup_query(
        rollout: Rollout,
        vlm: VLM,
        target_fps: int = 5,
        refusal_phrases: list[str] | None = None,
    ) -> tuple[str, list[np.ndarray], list[int], str] | dict[str, Rollout | str]:
        # Define local async functions here to avoid using ARES imports in the Modal container

        try:
            frames, frame_indices = load_video_frames(
                rollout.dataset_filename,
                rollout.filename,
                target_fps,
            )
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
        rollouts: list[Rollout],
        vlm: VLM,
        ann_db: AnnotationDatabase,
        target_fps: int = 5,
    ) -> tuple[dict[str, int], list[dict]]:
        rollout_ids = [r.id for r in rollouts]

        # Create and gather the futures properly
        annotation_input_futures = [
            asyncio.create_task(setup_query(rollout, vlm, target_fps))
            for rollout in rollouts
        ]

        annotator = ModalWrapper()
        stats, failures = await run_annotation_parallel(
            annotator,
            rollout_ids,
            annotation_input_futures,
            ann_db,
            rollouts,
        )
        return stats, failures

    ann_db = AnnotationDatabase(connection_string=ANNOTATION_DB_PATH)
    engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)

    if retry_failed_path:
        if "pkl" in retry_failed_path:
            with open(retry_failed_path, "rb") as f:
                failures = pickle.load(f)
            failed_ids = [str(f["rollout_id"]) for f in failures]
        elif "txt" in retry_failed_path:
            with open(retry_failed_path, "r") as f:
                failed_ids = [line.strip() for line in f.readlines()]
            rollouts = get_rollouts_by_ids(engine, failed_ids)
        else:
            raise ValueError(f"Unknown file type: {retry_failed_path}")
    elif rollout_ids:
        rollouts = get_rollouts_by_ids(engine, rollout_ids)
    else:
        dataset_info = [
            d for d in DATASET_NAMES if d["dataset_filename"] == dataset_filename
        ][0]
        dataset_formalname = dataset_info["dataset_formalname"]
        rollouts = setup_rollouts(engine, dataset_formalname)
        rollouts = [r for r in rollouts if r.split == split]
    if len(rollouts) == 0:
        print(
            f"no rollouts found for dataset filename {dataset_filename}, retry failed path {retry_failed_path}"
        )
        return

    print(f"\n\nfound {len(rollouts)} total rollouts\n\n")
    tic = time.time()
    overall_stats = dict()
    overall_failures = []

    # limited by CPU RAM (cant actually create all the potential requests at once, so run in "outer" batches as opposed to on-device gpu "inner" batches)
    for i in range(0, len(rollouts), outer_batch_size):
        print(
            f"processing batch {i // outer_batch_size + 1} of {len(rollouts) // outer_batch_size}"
        )
        # We want to create the item within the outer loop as the vlm semaphore gets "bound"
        # to the async context it is used in
        vlm = get_gpt_4o()
        rollouts_batch = rollouts[i : i + outer_batch_size]
        stats, failures = asyncio.run(
            run_ground_and_annotate(
                rollouts_batch,
                vlm,
                ann_db,
                annotation_fps,
            )
        )
        print(
            f"completed batch {i // outer_batch_size + 1} of {max(1, len(rollouts) // outer_batch_size)}"
        )
        for k, v in stats.items():
            overall_stats[k] = overall_stats.get(k, 0 if isinstance(v, int) else []) + v
        overall_failures.extend(failures)

    print(f"\n\nfailures: {overall_failures}\n\n")

    # write failures to file in order to retry
    with open(FAILURES_PATH, "wb") as f:
        pickle.dump(overall_failures, f)

    print("time taken", time.time() - tic)
    print(f"\n\n")
    for k, v in overall_stats.items():
        print(f"{k}: {v}" if not isinstance(v, list) else f"{k}: {v[:10]}...")
    print(f"\nn fails: {len(overall_failures)}")
