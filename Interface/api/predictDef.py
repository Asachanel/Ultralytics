import os
import random
import shutil
import string
import subprocess
import warnings
from collections import OrderedDict
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    import ffmpeg
    import imageio_ffmpeg as iff

    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    warnings.warn("ffmpeg-python or imageio-ffmpeg not installed. Video output will have no audio.")

from .utils import clear_gpu_memory

# ---------- LRU 模型缓存 ----------
_MODEL_CACHE = OrderedDict()
MAX_CACHE_SIZE = 2


def _get_model(model_name: str) -> YOLO:
    model_path = f'Source/weights/{model_name}.pt'
    if model_path in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(model_path)
        return _MODEL_CACHE[model_path]

    if len(_MODEL_CACHE) >= MAX_CACHE_SIZE:
        oldest_path, oldest_model = _MODEL_CACHE.popitem(last=False)
        del oldest_model
        clear_gpu_memory()

    model = YOLO(model_path)
    _MODEL_CACHE[model_path] = model
    return model


def clear_model_cache():
    global _MODEL_CACHE
    for model in _MODEL_CACHE.values():
        del model
    _MODEL_CACHE.clear()
    clear_gpu_memory()


def _generate_random_filename(length: int = 8) -> str:
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))


def _merge_audio_video(video_path: str, audio_source_path: str, output_path: str) -> None:
    """
    将处理后的视频与原始视频的音频合并，确保视频和音频同时存在。
    采用 H.264 + AAC 编码以保证最大兼容性。
    """
    try:
        ffmpeg_exe = iff.get_ffmpeg_exe()
    except Exception as e:
        warnings.warn(f"FFmpeg not found: {e}")
        shutil.copy2(video_path, output_path)
        return

    # 检查输入视频是否包含视频流（预防万一）
    probe_cmd = [ffmpeg_exe, '-i', video_path]
    probe_result = subprocess.run(
        probe_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if 'Video:' not in probe_result.stderr:
        print("Input video has no video stream, copying directly.")
        shutil.copy2(video_path, output_path)
        return

    # 使用重新编码方式，保证视频和音频都正确写入
    cmd = [
        ffmpeg_exe,
        '-i', video_path,  # 0: 处理后的无声视频
        '-i', audio_source_path,  # 1: 原始带音频的视频
        '-c:v', 'libx264',  # 视频编码为 H.264
        '-preset', 'ultrafast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',  # 音频编码为 AAC
        '-b:a', '128k',  # 音频比特率
        '-map', '0:v:0',  # 取第一个输入的视频流
        '-map', '1:a:0?',  # 取第二个输入的第一个音频流（若不存在则忽略）
        '-shortest',  # 以较短的流为准
        '-y',  # 覆盖输出文件
        output_path
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        if result.returncode != 0:
            print(f"FFmpeg merge failed (code {result.returncode}):\n{result.stderr}")
            # 回退：仅复制视频（无音频），但至少保留画面
            _transcode_video_only(video_path, output_path, ffmpeg_exe)
        else:
            print(f"Audio merged successfully: {output_path}")
    except Exception as e:
        print(f"Subprocess error during merge: {e}")
        _transcode_video_only(video_path, output_path, ffmpeg_exe)


def _transcode_video_only(video_path: str, output_path: str, ffmpeg_exe: str) -> None:
    """仅转码视频为 H.264 编码，不包含音频。"""
    cmd = [
        ffmpeg_exe,
        '-i', video_path,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-an',  # 禁用音频
        '-y',
        output_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"Video transcoded (no audio): {output_path}")
    except Exception as e:
        print(f"Video transcoding failed: {e}")
        shutil.copy2(video_path, output_path)


def predictDef(
        image: Optional[np.ndarray] = None,
        video: Optional[str] = None,
        model: str = "yolo26n",
        imgsz: int = 640,
        conf_threshold: float = 0.25
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    os.makedirs("Source/inputs", exist_ok=True)
    os.makedirs("Source/outputs", exist_ok=True)

    model_obj = _get_model(model)

    if image is not None:
        results = model_obj.predict(source=image, imgsz=imgsz, conf=conf_threshold)
        annotated_image = results[0].plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        del results
        return annotated_image_rgb, None

    else:
        video_input_path = video
        filename = _generate_random_filename()
        temp_video_path = f'Source/outputs/{filename}_temp.mp4'
        final_output_path = f'Source/outputs/{filename}.mp4'

        cap = None
        out = None

        try:
            cap = cv2.VideoCapture(video_input_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_input_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise RuntimeError("Cannot open VideoWriter.")

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model_obj.predict(source=frame, imgsz=imgsz, conf=conf_threshold)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

                del results
                if frame_count % 50 == 0:
                    torch.cuda.empty_cache()

                frame_count += 1

            print(f"Processed {frame_count} frames.")

            out.release()
            cap.release()
            out = None
            cap = None

            # 合并音频（重新编码方式，确保画面和声音）
            _merge_audio_video(temp_video_path, video_input_path, final_output_path)

            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

            return None, final_output_path

        except Exception as e:
            for path in [temp_video_path, final_output_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            raise RuntimeError(f"Video processing failed: {str(e)}")

        finally:
            if cap:
                cap.release()
            if out:
                out.release()
            torch.cuda.empty_cache()
