import base64
import io
import os
import random
import tempfile
import typing as t
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from moviepy.editor import ImageSequenceClip
from PIL import Image

from ares.constants import ARES_VIDEO_DIR

MAX_N_FRAMES = 40


def get_image_from_path(path: str) -> Image.Image:
    if path.startswith(("http")):
        return Image.open(requests.get(path, stream=True).raw)
    else:
        return Image.open(path)


def get_video_from_path(
    dataset: str, path: str
) -> str | bytes | io.BytesIO | np.ndarray:
    path = Path(path).with_suffix(".mp4")
    return os.path.join(ARES_VIDEO_DIR, dataset, path)


def save_video(
    video: t.Union[str, bytes | io.BytesIO | np.ndarray | list[np.ndarray] | list[str]],
    dataset: str,
    filename: str,
) -> tuple[str, str]:
    """Save video as both MP4 and individual frames.

    Returns:
        tuple[str, str]: (mp4_path, frames_dir)
    """
    # Remove .mp4 extension if present and create paths
    if "mp4" in filename:
        raise ValueError("Base filename should not contain .mp4; received: ", filename)
    mp4_path = os.path.join(ARES_VIDEO_DIR, dataset, f"{filename}.mp4")
    frames_dir = os.path.join(ARES_VIDEO_DIR, dataset, filename)

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
    elif isinstance(video, list) and all(isinstance(f, str) for f in video):
        frames = video
    else:
        raise TypeError(
            f"Unsupported video format. Use numpy array or list of numpy arrays. Got {type(video)}{'with slices of type ' + str(type(video[0])) if (isinstance(video, list) and video) else ''}"
        )

    if not frames:
        raise ValueError(f"No frames to save for {filename}; received {frames}")

    # Save MP4 using moviepy
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile(mp4_path, codec="libx264", logger=None)

    # Save individual frames
    for i, frame in enumerate(frames):
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
        if isinstance(frame, str):
            # constant memory usage
            frame = cv2.imread(frame)
        cv2.imwrite(frame_path, frame)
    return mp4_path, frames_dir


def get_video_frames(
    dataset_filename: str,
    filename: str,
    n_frames: int | None = None,
    just_path: bool = False,
) -> list[np.ndarray | str]:
    """Get video as a list of frames from the frames directory."""

    frames_dir = os.path.join(ARES_VIDEO_DIR, dataset_filename, filename)

    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
    if n_frames is not None:
        frame_files = frame_files[:n_frames]

    frame_paths = [os.path.join(frames_dir, f) for f in frame_files]
    if just_path:
        return frame_paths
    frames = [cv2.imread(f) for f in frame_paths]
    return frames


def get_video_mp4(dataset_filename: str, filename: str) -> str:
    """Get path to the MP4 video file."""
    if not filename.endswith(".mp4"):
        filename += ".mp4"
    print("mp4: ", dataset_filename, filename)
    mp4_path = os.path.join(ARES_VIDEO_DIR, dataset_filename, filename)
    if not os.path.exists(mp4_path):
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
            f"Unsupported image format. Use file path, numpy array, or PIL image. Received {type(image)}"
            f"{type(image)}"
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
    n_frames: int | None = None,
    specified_frames: list[int] | None = None,
    resize: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    if specified_frames is not None:
        # Filter out any indices that exceed the frame count
        specified_frames = [i for i in specified_frames if i < len(all_frames)]
        frames = [all_frames[i] for i in specified_frames]
    elif n_frames is not None:
        if n_frames == 1:
            # if only one unspecified frame is requested, use the last frame
            frames = [all_frames[-1]]
        else:
            # otherwise, use evenly spaced frames
            total_frames = len(all_frames)
            indices = np.linspace(
                0, total_frames - 1, n_frames, dtype=int, endpoint=True
            )
            frames = [all_frames[i] for i in indices]
    else:
        raise ValueError("Either n_frames or specified_frames must be provided")

    if isinstance(frames[0], str):
        frames = [cv2.imread(str(frame)) for frame in frames]

    if resize:
        frames = [cv2.resize(frame, resize) for frame in frames]
    return frames


def get_frame_indices_for_fps(
    video_path: str, target_fps: int | float = 1
) -> list[int]:
    """Calculate frame indices to sample a video at a target FPS rate.

    Args:
        video_path: Path to the video file
        target_fps: Desired frames per second to sample (default: 1)

    Returns:
        List of frame indices to sample
    """
    video_path = str(Path(video_path).with_suffix(".mp4"))
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Calculate frame indices to sample at desired fps_rate
    sample_interval = int(video_fps / target_fps)
    # Ensure we don't generate indices beyond total_frames - 1
    return list(range(0, min(total_frames - 1, total_frames), sample_interval))


def load_video_frames(
    dataset_filename: str, fname: str, target_fps: int | float = 1
) -> tuple[t.List[np.ndarray], t.List[int]]:
    """Load video frames at specified FPS."""
    video_path = get_video_from_path(dataset_filename, fname)

    # FPS of 0 means to just use the first and last frames
    if target_fps == 0:
        frame_indices = [0, -1]
    else:
        frame_indices = get_frame_indices_for_fps(video_path, target_fps=target_fps)
    # some videos/frame sequences are too long for context lengths or API limits
    # so we downsample to a max number of frames but maintain the first and last frames
    if len(frame_indices) > MAX_N_FRAMES:
        print(
            f"Downsampling video from {len(frame_indices)} frames to {MAX_N_FRAMES} frames"
        )
        middle_indices = np.linspace(
            1, len(frame_indices) - 2, MAX_N_FRAMES - 2, dtype=int
        )
        frame_indices = (
            [frame_indices[0]]
            + [frame_indices[i] for i in middle_indices]
            + [frame_indices[-1]]
        )

    # get all frame paths (just strings to avoid memory issues)
    all_frames = get_video_frames(
        dataset_filename, fname, n_frames=None, just_path=True
    )
    # select the frames
    frames_to_process = choose_and_preprocess_frames(
        all_frames, specified_frames=frame_indices
    )
    return frames_to_process, frame_indices
