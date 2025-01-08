import os
import pickle
import tempfile
import time
from typing import List, Tuple

import cv2
import numpy as np
import pymongo
import torch
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm

from ares.databases.annotation_database import (
    TEST_ANNOTATION_DB_PATH,
    AnnotationDatabase,
)
from ares.models.base import VLM
from ares.models.grounding import GroundingAnnotator, get_grounding_nouns
from ares.models.shortcuts import get_gemini_2_flash, get_gpt_4o
from ares.utils.image_utils import load_video_frames

if __name__ == "__main__":
    from ares.databases.structured_database import (
        TEST_ROBOT_DB_PATH,
        RolloutSQLModel,
        get_rollout_by_name,
        setup_database,
    )

    # Initialize components
    ann_db = AnnotationDatabase(connection_string=TEST_ANNOTATION_DB_PATH)
    annotator = GroundingAnnotator(
        segmenter_id=None,
    )
    engine = setup_database(RolloutSQLModel, path=TEST_ROBOT_DB_PATH)

    # Process video
    # Process video
    # dataset_name = "cmu_play_fusion"
    # fname = "data/train/episode_208.mp4"
    formal_dataset_name = "UCSD Kitchen"
    dataset_name = "ucsd_kitchen_dataset_converted_externally_to_rlds"
    fnames = [f"data/train/episode_{i}.mp4" for i in range(50, 150)]
    target_fps = 5

    # vlm = get_gemini_2_flash()
    vlm = get_gpt_4o()

    # Create progress bar for total files
    tic = time.time()
    total_anns = 0
    total_processed = 0
    total_frames = 0
    pbar = tqdm(total=len(fnames), desc="Processing videos")

    for fname in fnames:
        rollout = get_rollout_by_name(
            engine, formal_dataset_name, fname.replace("mp4", "npy")
        )
        frames, frame_indices = load_video_frames(dataset_name, fname, target_fps)
        label_str = get_grounding_nouns(
            vlm,
            frames,
            rollout.task.language_instruction,
        )

        # Pass description to annotate_video
        video_annotations = annotator.annotate_video(
            frames, label_str, desc=f"Frames for {os.path.basename(fname)}"
        )
        total_anns += sum(len(anns) for anns in video_annotations)
        total_frames += len(frames)

        video_id = ann_db.add_video_with_annotations(
            dataset_name=dataset_name,
            video_path=fname,
            frames=frames,
            frame_indices=frame_indices,
            annotations=video_annotations,
            label_str=label_str,
        )

        total_processed += 1
        pbar.update(1)
        pbar.set_postfix({"Last file": os.path.basename(fname)})

    pbar.close()
    toc = time.time()
    print(f"Total time: {toc - tic:.2f} seconds")
    print(f"Total annotations: {total_anns}")
    print(f"Total processed: {total_processed}")
    print(f"Total frames: {total_frames}")
    breakpoint()
