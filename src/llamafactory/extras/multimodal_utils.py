import re
import os
import base64
import requests
import io
import numpy as np
import numpy.typing as npt
# pylint: disable=no-member
import cv2
from typing import Union
import tempfile
from .packages import is_pillow_available, is_vllm_available

if is_pillow_available():
    from PIL import Image

if is_vllm_available():
    from vllm.multimodal.utils import fetch_video

def save_video_to_temp(video_url: str) -> str:
    """将视频URL（HTTP或base64）保存为临时文件。

    Args:
        video_url: 视频的URL，可以是HTTP URL或base64编码的数据URL

    Returns:
        str: 临时文件的路径

    Raises:
        ValueError: 当URL格式无效或无法保存文件时
    """
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_path = temp_file.name

    try:
        if re.match(r"^data:video\/(mp4|webm|ogg);base64,(.+)$", video_url):
            # 处理base64编码的视频
            video_bytes = base64.b64decode(video_url.split(",", maxsplit=1)[1])
            temp_file.write(video_bytes)
        elif os.path.isfile(video_url):
            # 如果已经是本地文件，直接返回路径
            temp_file.close()
            os.unlink(temp_path)  # 删除创建的临时文件
            return video_url
        else:
            # 处理HTTP URL
            try:
                response = requests.get(video_url, stream=True)
                response.raise_for_status()  # 检查请求是否成功
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Failed to download video from URL: {str(e)}")

        temp_file.close()
        return temp_path

    except Exception as e:
        # 如果出现错误，清理临时文件
        temp_file.close()
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise ValueError(f"Failed to save video: {str(e)}")


def video_to_ndarrays(path: str, num_frames: int = -1) -> npt.NDArray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    frames = np.stack(frames)
    frames = sample_frames_from_video(frames, num_frames)
    if len(frames) < num_frames:
        raise ValueError(f"Could not read enough frames from video file {path}"
                         f" (expected {num_frames} frames, got {len(frames)})")
    return frames

def _load_video_from_bytes(video_bytes: bytes, num_frames: int = -1) -> npt.NDArray:
    """Load video from bytes using OpenCV.
    Args:
        video_bytes: Video bytes
        num_frames: Number of frames to extract. Default is -1 (all frames).

    Returns:
        numpy.ndarray: Array of video frames
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(video_bytes, np.uint8)

    # Create a memory file object for OpenCV to read from
    cap = cv2.VideoCapture()
    cap.open(nparr)
    if not cap.isOpened():
        raise ValueError("Could not open video from bytes")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("No frames were extracted from the video")

    frames = np.stack(frames)
    frames = sample_frames_from_video(frames, num_frames)
    if num_frames > 0 and len(frames) < num_frames:
        raise ValueError(f"Could not read enough frames from video"
                         f" (expected {num_frames} frames, got {len(frames)})")
    return frames


def process_video_url(video_url: str, num_frames: int = 16) -> npt.NDArray:
    """Process video URL and extract frames.

    Args:
        video_url: URL/path to video file, or base64 data URL
        num_frames: Number of frames to extract from the video. Default is 16.

    Returns:
        numpy.ndarray: Array of video frames with shape (num_frames, height, width, 3)
    """
    if isinstance(video_url, str):
        try:
            # 保存到临时文件
            temp_path = save_video_to_temp(video_url)
            # 处理视频文件
            try:
                return video_to_ndarrays(temp_path, num_frames)
            finally:
                # 如果是临时文件，处理完后删除
                if temp_path != video_url and os.path.exists(temp_path):
                    os.unlink(temp_path)
        except Exception:
            import traceback
            raise ValueError(f"Failed to process video: {str(video_url)}\n {traceback.format_exc()}")

    raise ValueError(
        "Invalid video_url format. Must be file path, URL, or base64 data URL.")


def sample_frames_from_video(frames: npt.NDArray,
                             num_frames: int) -> npt.NDArray:
    """Sample frames from video array.

    Args:
        frames: Video frames array
        num_frames: Number of frames to sample. If -1, return all frames.

    Returns:
        numpy.ndarray: Sampled frames array
    """
    total_frames = frames.shape[0]
    if num_frames == -1:
        return frames

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled_frames = frames[frame_indices, ...]
    return sampled_frames


def process_image_url(image_url):
    if re.match(r"^data:image\/(png|jpg|jpeg|gif|bmp);base64,(.+)$", image_url):  # base64 image
        image_stream = io.BytesIO(base64.b64decode(
            image_url.split(",", maxsplit=1)[1]))
    elif os.path.isfile(image_url):  # local file
        image_stream = open(image_url, "rb")
    else:  # web uri
        image_stream = requests.get(image_url, stream=True).raw
    return Image.open(image_stream).convert("RGB")
