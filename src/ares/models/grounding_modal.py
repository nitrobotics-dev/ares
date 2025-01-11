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
) -> dict[str, int]:
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
        refusal_phrases: list[str] | None = None,
    ) -> tuple[str, list[np.ndarray], list[int], str] | dict[str, Rollout | str]:
        try:
            frames, frame_indices = load_video_frames(
                dataset_name,
                rollout.path,
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
        dataset_file_name: str,
        rollouts: list[Rollout],
        vlm: VLM,
        ann_db: AnnotationDatabase,
        target_fps: int = 5,
    ) -> tuple[dict, list[dict]]:
        rollout_ids = [r.id for r in rollouts]

        # Create and gather the futures properly
        annotation_input_futures = [
            asyncio.create_task(
                setup_query(dataset_file_name, rollout, vlm, target_fps)
            )
            for rollout in rollouts
        ]

        annotator = ModalWrapper()
        stats, failures = await run_annotation_parallel(
            annotator,
            rollout_ids,
            annotation_input_futures,
            ann_db,
            rollouts,
            dataset_file_name,
        )
        return stats, failures

    # formal_dataset_name = "CMU Stretch"
    # dataset_file_name = "cmu_stretch"
    # formal_dataset_name, dataset_file_name = (
    #     "LSMO Dataset",
    #     "tokyo_u_lsmo_converted_externally_to_rlds",
    # )
    # formal_dataset_name, dataset_file_name = (
    #     "Berkeley Fanuc Manipulation",
    #     "berkeley_fanuc_manipulation",
    # )
    # formal_dataset_name, dataset_file_name = (
    #     "CMU Franka Exploration",
    #     "cmu_franka_exploration_dataset_converted_externally_to_rlds",
    # )
    # formal_dataset_name, dataset_file_name = (
    #     "CMU Play Fusion",
    #     "cmu_play_fusion",
    # )
    # formal_dataset_name, dataset_file_name = (
    #     "NYU ROT",
    #     "nyu_rot",
    # )
    # formal_dataset_name, dataset_file_name = (
    #     "UCSD Pick Place",
    #     "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
    # )
    formal_dataset_name, dataset_file_name = (
        "USC Jaco Play",
        "usc_jaco_play",
    )

    target_fps = 5

    retry_failed = None  # path to pickle file with failures
    # retry_failed = "failures.pkl"

    ann_db = AnnotationDatabase(connection_string=TEST_ANNOTATION_DB_PATH)
    engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)
    rollouts = setup_rollouts(engine, formal_dataset_name)[700:900]

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
            dataset_file_name,
            rollouts,
            vlm,
            ann_db,
            target_fps,
        )
    )

    print(f"\n\nfailures: {failures}\n\n")
    # write failures to file
    with open("failures.pkl", "wb") as f:
        pickle.dump(failures, f)

    print("time taken", time.time() - tic)
    print(f"\n\n")
    for k, v in stats.items():
        print(f"{k}: {v}" if not isinstance(v, list) else f"{k}: {v[:10]}...")
    breakpoint()
