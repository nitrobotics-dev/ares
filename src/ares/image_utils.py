import base64
import io
import os
import tempfile
import typing as t

import cv2
import numpy as np
from PIL import Image


def encode_image(image: t.Union[str, np.ndarray, Image.Image]) -> str:
    if isinstance(image, str):  # file path
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image, (np.ndarray, Image.Image)):  # numpy array or PIL image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
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
                tempfile.gettempdir(), f"frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.png"
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
)-> list[np.ndarray]:
    if specified_frames is None:
        total_frames = len(all_frames)
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int, endpoint=True)
        frames = [all_frames[i] for i in indices]
    else:
        frames = [all_frames[i] for i in specified_frames]

    if isinstance(frames[0], str):
        frames = [cv2.imread(frame) for frame in frames]

    if resize:
        frames = [cv2.resize(frame, (224, 224)) for frame in frames]
    return frames
