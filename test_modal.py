import asyncio
import pickle
import time
import traceback

# import typing as t
# from concurrent.futures import as_completed

# import numpy as np
# from modal import App, Image, Stub, method

# from ares.configs.annotations import Annotation
# from ares.configs.base import Rollout
# from ares.databases.annotation_database import AnnotationDatabase
# from ares.models.base import VLM

# # from ares.models.grounding import GroundingAnnotator
# from ares.models.grounding_utils import convert_to_annotations
# from run_grounding_annotations import setup_query


# async def run_modal(
#     rollouts: list[Rollout],
#     vlm: VLM,
#     annotator: ModalWrapper,
#     ann_db: AnnotationDatabase,
#     dataset_name: str,
#     target_fps: int,
# ) -> dict:
#     total_anns = 0
#     total_processed = 0
#     total_frames = 0
#     failed_setup_labeling_attempts = []

#     # Create tasks for all rollouts
#     setup_tasks = []
#     rollout_ids = []
#     id_to_rollout = {rollout.id: rollout for rollout in rollouts}
#     # for rollout in rollouts:
#     #     try:
#     #         task = setup_query(dataset_name, rollout, vlm, target_fps)
#         setup_tasks.append(task)
#         rollout_ids.append(rollout.id)
#     except Exception as e:
#         failed_setup_labeling_attempts.append(
#             ("setup futures", rollout.id, e, traceback.format_exc())
#         )

# Process all setup tasks concurrently using gather
#     failed_setup_annotation_attempts = []
#     failed_annotation_attempts = []
#     annotation_futures = {}
#     future_to_metadata = {}
#     try:
#         # results = await asyncio.gather(*setup_tasks, return_exceptions=True)
#         rollout_ids, results = pickle.load(open("label_results.pkl", "rb"))
#         for rollout_id, result in zip(rollout_ids[:1], results[:1]):
#             if isinstance(result, Exception):
#                 failed_setup_annotation_attempts.append(
#                     ("setup gather", rollout_id, result, traceback.format_exc())
#                 )
#                 continue

#             frames, frame_indices, label_str = result
#             print(label_str)
#             print(f"launching req for {rollout_id}")
#             frames = [
#                 f.tolist() for f in frames
#             ]  # Convert to list before sending to Modal
#             ann_future = annotator.annotate_video.remote(frames, label_str)
#             annotation_futures[rollout_id] = ann_future
#             future_to_metadata[rollout_id] = (
#                 frames,
#                 frame_indices,
#                 label_str,
#                 id_to_rollout[rollout_id].path,
#             )

#         # Wait for all Modal futures to complete asynchronously
#         annotation_results = []
#         for rollout_id in annotation_futures:
#         #     try:
#         #         result = await annotation_futures[
#         #             rollout_id
#         #         ].get_async()  # Use .get_async() for async waiting
#         #         annotation_results.append(result)
#         #     except Exception as e:
#         #         print(
#         #             f"failed to get annotation for {rollout_id}; {e}; {traceback.format_exc()}"
#         #         )
#         #         failed_annotation_attempts.append(
#         #             ("annotation", rollout_id, e, traceback.format_exc())
#         #         )
#         #         continue

#         # print(f"got {len(annotation_results)} annotation results")
#         # for rollout_id, result in zip(annotation_futures.keys(), annotation_results):
#         #     if isinstance(result, Exception):
#         #         failed_annotation_attempts.append(
#         #             ("annotation", rollout_id, result, traceback.format_exc())
#         #         )
#         #         continue
#         #     # Process successful results here
#         #     frames, frame_indices, label_str, path = future_to_metadata[rollout_id]
#             total_frames += len(frames)
# #             total_processed += 1
# #             total_anns += sum(len(frame_anns) for frame_anns in result)

# #     except Exception as e:
# #         failed_annotation_attempts.append(("gather", "all", e, traceback.format_exc()))
# #     breakpoint()
# #     return {
# #         "results": results,
# #         "failures": {
# #             "failed_setup_labeling_attempts": failed_setup_labeling_attempts,
# #             "failed_setup_annotation_attempts": failed_setup_annotation_attempts,
# #             "failed_annotation_attempts": failed_annotation_attempts,
# #         },
# #         "totals": {
# #             "total_anns": total_anns,
# #             "total_processed": total_processed,
# #             "total_frames": total_frames,
# #         },
# #     }


# # @app.local_entrypoint()
# # def run() -> None:
# #     from ares.databases.annotation_database import (
# #         TEST_ANNOTATION_DB_PATH,
# #         AnnotationDatabase,
# #     )
# #     from ares.databases.structured_database import (
# #         TEST_ROBOT_DB_PATH,
# #         RolloutSQLModel,
# #         get_rollout_by_name,
# #         setup_database,
# #     )
# #     from ares.models.shortcuts import get_gemini_2_flash, get_gpt_4o
# #     from run_grounding_annotations import setup_query, setup_rollouts

# #     formal_dataset_name = "CMU Stretch"
# #     dataset_name = "cmu_stretch"
# #     target_fps = 5

# #     ann_db = AnnotationDatabase(connection_string=TEST_ANNOTATION_DB_PATH)
# #     engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)

# #     rollouts = setup_rollouts(engine, formal_dataset_name)[:1]
# #     print(f"Len Rollouts: {len(rollouts)}")
# #     vlm = get_gpt_4o()
# #     annotator = ModalWrapper()

#     tic = time.time()
#     # total_anns, total_processed, total_frames = asyncio.run(
#     out = asyncio.run(
#         run_modal(rollouts, vlm, annotator, ann_db, dataset_name, target_fps)
#     )
#     breakpoint()
#     print(f"Total annotations: {total_anns}")
#     print(f"Total frames: {total_frames}")
#     print(f"Total processed: {total_processed}")
#     breakpoint()
