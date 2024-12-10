import base64
import io
import os
import random
import tempfile
import typing as t

import cv2
import numpy as np
import pandas as pd
from moviepy.editor import ImageSequenceClip
from PIL import Image

ARES_DATASET_VIDEO_PATH = "/workspaces/ares/data/videos"


def get_video_from_path(
    dataset: str, path: str
) -> str | bytes | io.BytesIO | np.ndarray:
    return os.path.join(ARES_DATASET_VIDEO_PATH, dataset, path)


def get_video_from_cloud(
    dataset: str, path: str
) -> str | bytes | io.BytesIO | np.ndarray:
    raise NotImplementedError("Not implemented")


# def get_video(inp: t.Any) -> str | bytes | io.BytesIO | np.ndarray:
#     if isinstance(inp, pd.Series):
#         return get_video_from_path(inp["dataset"], inp["path"])
#     elif isinstance(inp, str):
#         if ARES_DATASET_VIDEO_PATH in inp:
#             return inp
#         else:
#             raise ValueError(f"Invalid video path: {inp}")
#     elif isinstance(inp, (np.ndarray, io.BytesIO)):
#         return inp
#     else:
#         raise ValueError(f"Invalid video input type: {type(inp)}")


def save_video(
    video: t.Union[str, bytes | io.BytesIO | np.ndarray | list[np.ndarray]],
    dataset: str,
    filename: str,
) -> tuple[str, str]:
    """Save video as both MP4 and individual frames.

    Returns:
        tuple[str, str]: (mp4_path, frames_dir)
    """
    # Remove .mp4 extension if present and create paths
    base_filename = filename.replace(".mp4", "")
    mp4_path = os.path.join(ARES_DATASET_VIDEO_PATH, dataset, f"{base_filename}.mp4")
    frames_dir = os.path.join(ARES_DATASET_VIDEO_PATH, dataset, base_filename)

    # Create frames directory if it doesn't exist
    os.makedirs(frames_dir, exist_ok=True)

    # Convert video to list of frames if needed
    if isinstance(video, np.ndarray):
        if len(video.shape) == 4:
            frames = [video[i] for i in range(len(video))]
        else:
            raise ValueError(
                "Video numpy array must be 4D [frames, height, width, channels]"
            )
    elif isinstance(video, list) and all(isinstance(f, np.ndarray) for f in video):
        frames = video
    else:
        raise TypeError(
            "Unsupported video format. Use numpy array or list of numpy arrays."
        )

    if not frames:
        raise ValueError("No frames to save")

    # Save MP4 using moviepy
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile(mp4_path, codec="libx264", logger=None)

    # Save individual frames
    for i, frame in enumerate(frames):
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)

    return mp4_path, frames_dir


def get_video_frames(dataset: str, filename: str) -> list[np.ndarray]:
    """Get video as a list of frames from the frames directory."""
    base_filename = filename.replace(".mp4", "")
    frames_dir = os.path.join(ARES_DATASET_VIDEO_PATH, dataset, base_filename)

    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
    frames = [cv2.imread(os.path.join(frames_dir, f)) for f in frame_files]
    return frames


def get_video_mp4(dataset: str, filename: str) -> str:
    """Get path to the MP4 video file."""
    if not filename.endswith(".mp4"):
        filename += ".mp4"
    mp4_path = os.path.join(ARES_DATASET_VIDEO_PATH, dataset, filename)

    if not os.path.exists(mp4_path):
        # choose random file from dir
        # mp4_path = "/workspaces/ares/data/videos/cmu_play_fusion/data/train/"
        # mp4_path = mp4_path + random.choice(
        #     [x for x in os.listdir(mp4_path) if x.endswith(".mp4")]
        # )

        raise FileNotFoundError(f"MP4 file not found: {mp4_path}")
    return mp4_path


def encode_image(image: t.Union[str, np.ndarray, Image.Image]) -> str:
    if isinstance(image, str):  # file path
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image, (np.ndarray, Image.Image)):  # numpy array or PIL image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        raise TypeError(
            "Unsupported image format. Use file path, numpy array, or PIL image."
        )


def split_video_to_frames(
    video_path: str, filesize_limit_mb: int = 20
) -> list[np.ndarray | str]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # large video files are too big to process in one go, so we split them into frames
    # and only load the frames into memory that we need later
    filesize = os.path.getsize(video_path)
    write_images_flag = filesize > filesize_limit_mb * 1024 * 1024
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray | str] = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if write_images_flag:
            frame_path = os.path.join(
                tempfile.gettempdir(), f"frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg"
            )
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
        else:
            frames.append(frame)
    cap.release()
    return frames


def choose_and_preprocess_frames(
    all_frames: list[np.ndarray | str],
    n_frames: int = 10,
    specified_frames: list[int] | None = None,
    resize: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    assert n_frames > 0
    if specified_frames is None:
        if n_frames == 1:
            # if only one unspecified frame is requested, use the last frame
            frames = [all_frames[-1]]
        else:
            # otherwise, use evenly spaced frames
            # TODO: consider using biased samples
            total_frames = len(all_frames)
            indices = np.linspace(
                0, total_frames - 1, n_frames, dtype=int, endpoint=True
            )
            frames = [all_frames[i] for i in indices]
    else:
        frames = [all_frames[i] for i in specified_frames]

    if isinstance(frames[0], str):
        frames = [cv2.imread(str(frame)) for frame in frames]

    if resize:
        frames = [cv2.resize(frame, resize) for frame in frames]
    return frames
