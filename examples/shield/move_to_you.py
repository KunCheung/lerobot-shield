from __future__ import annotations

import base64
import json
import os
import platform
import struct
import sys
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any
import zlib

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
EXPECTED_LEROBOT_INIT_PATH = (SRC_PATH / "lerobot" / "__init__.py").resolve()
DOTENV_PATH = Path(__file__).with_name(".env")
MEMORY_DB_PATH = Path(os.getenv("TEMP", str(Path.cwd()))) / "robocrew_robot_memory.db"
TMP_IMAGES_DIR = Path(__file__).with_name("tmp_images")
LATEST_INPUT_IMAGE_PATH = TMP_IMAGES_DIR / "latest_input.png"
LATEST_INPUT_METADATA_PATH = TMP_IMAGES_DIR / "latest_input.json"
LATEST_RAW_CAMERA_IMAGE_PATH = TMP_IMAGES_DIR / "latest_raw_camera.png"
LATEST_RAW_CAMERA_METADATA_PATH = TMP_IMAGES_DIR / "latest_raw_camera.json"
LATEST_LLM_INPUT_STEM = TMP_IMAGES_DIR / "latest_llm_input"
LATEST_LLM_INPUT_METADATA_PATH = TMP_IMAGES_DIR / "latest_llm_input.json"
_LAST_DEBUG_SAVE_SIGNATURE: tuple[str, str] | None = None

MODEL_NAME = "openai:kimi-k2.5"
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
TASK = "Approach a human."
CAMERA_ID = 1
ROBOT_ID = "my_xlerobot_2wheels_lab"
PORT1 = "COM5"  # left arm + head
PORT2 = "COM4"  # right arm + 2-wheel base
LINEAR_SPEED_MPS = 0.10
ANGULAR_SPEED_DPS = 30.0


if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import lerobot as local_lerobot

LOCAL_LEROBOT_INIT_PATH = Path(local_lerobot.__file__).resolve()
if LOCAL_LEROBOT_INIT_PATH != EXPECTED_LEROBOT_INIT_PATH:
    raise RuntimeError(
        "This script must run against the local repository source, but imported "
        f"lerobot from {LOCAL_LEROBOT_INIT_PATH}. Expected {EXPECTED_LEROBOT_INIT_PATH}. "
        "Please run: conda activate lerobot && cd C:\\projects\\lerobot && "
        "python -m pip install -e . && python examples/shield/move_to_you.py"
    )

if load_dotenv is not None and DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH, override=False)

kimi_api_key = os.getenv("KIMI_API_KEY")
moonshot_api_key = os.getenv("MOONSHOT_API_KEY")
if kimi_api_key and not moonshot_api_key:
    moonshot_api_key = kimi_api_key
    os.environ["MOONSHOT_API_KEY"] = moonshot_api_key
if not moonshot_api_key:
    raise RuntimeError(f"Missing KIMI_API_KEY or MOONSHOT_API_KEY in {DOTENV_PATH}")

os.environ.setdefault("OPENAI_API_KEY", moonshot_api_key)
os.environ.setdefault("OPENAI_BASE_URL", MOONSHOT_BASE_URL)

import robocrew.core.memory as robocrew_memory


def _patched_memory_init(self, db_filename="robot_memory.db"):
    self.db_path = MEMORY_DB_PATH
    self.init_db()


robocrew_memory.Memory.__init__ = _patched_memory_init

from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels, XLerobot2WheelsConfig, TwoWheelsServoAdapter
from lerobot.cameras.configs import ColorMode, Cv2Backends
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.errors import DeviceNotConnectedError
from robocrew.core.camera import RobotCamera
import robocrew.core.LLMAgent as robocrew_llm_module
from robocrew.core.utils import basic_augmentation
from robocrew.robots.XLeRobot.tools import create_move_forward, create_turn_left, create_turn_right
from langchain.chat_models import init_chat_model as lc_init_chat_model

try:
    from lerobot.scripts.lerobot_find_cameras import (
        build_opencv_candidate_configs as _build_opencv_candidate_configs,
        is_blank_frame as _is_blank_frame,
    )
except Exception:
    def _is_blank_frame(img_array: Any) -> tuple[bool, str]:
        if np is None:
            raise RuntimeError("numpy is required to compute blank-frame statistics.")

        mean_value = float(img_array.mean())
        std_value = float(img_array.std())
        min_value = int(img_array.min())
        max_value = int(img_array.max())

        is_white = mean_value >= 250.0 and std_value <= 2.0
        is_black = mean_value <= 5.0 and std_value <= 2.0

        if is_white:
            return True, (
                f"blank-white frame detected (mean={mean_value:.1f}, std={std_value:.1f}, "
                f"min={min_value}, max={max_value})"
            )
        if is_black:
            return True, (
                f"blank-black frame detected (mean={mean_value:.1f}, std={std_value:.1f}, "
                f"min={min_value}, max={max_value})"
            )

        return False, (
            f"valid frame stats (mean={mean_value:.1f}, std={std_value:.1f}, "
            f"min={min_value}, max={max_value})"
        )


    def _build_opencv_candidate_configs(cam_id: str | int) -> list[OpenCVCameraConfig]:
        configs = [
            OpenCVCameraConfig(index_or_path=cam_id, color_mode=ColorMode.RGB, backend=Cv2Backends.ANY),
        ]

        if platform.system() == "Windows" and isinstance(cam_id, int):
            configs.extend(
                [
                    OpenCVCameraConfig(index_or_path=cam_id, color_mode=ColorMode.RGB, backend=Cv2Backends.MSMF),
                    OpenCVCameraConfig(
                        index_or_path=cam_id,
                        color_mode=ColorMode.RGB,
                        backend=Cv2Backends.MSMF,
                        fourcc="MJPG",
                    ),
                    OpenCVCameraConfig(
                        index_or_path=cam_id,
                        color_mode=ColorMode.RGB,
                        backend=Cv2Backends.MSMF,
                        fourcc="YUY2",
                    ),
                    OpenCVCameraConfig(index_or_path=cam_id, color_mode=ColorMode.RGB, backend=Cv2Backends.DSHOW),
                    OpenCVCameraConfig(
                        index_or_path=cam_id,
                        color_mode=ColorMode.RGB,
                        backend=Cv2Backends.DSHOW,
                        fourcc="MJPG",
                    ),
                    OpenCVCameraConfig(
                        index_or_path=cam_id,
                        color_mode=ColorMode.RGB,
                        backend=Cv2Backends.DSHOW,
                        fourcc="YUY2",
                    ),
                ]
            )

        return configs


def _now_isoformat() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _frame_to_uint8_array(frame: Any) -> Any:
    if np is None:
        raise RuntimeError("numpy is not available, cannot convert debug image frame.")

    array = np.asarray(frame)
    if array.ndim not in (2, 3):
        raise RuntimeError(f"Unsupported frame ndim: {array.ndim}")

    if array.dtype == np.uint8:
        return array

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        scale = 255.0 if max_value <= 1.0 else 1.0
        return np.clip(array * scale, 0, 255).astype(np.uint8)

    return np.clip(array, 0, 255).astype(np.uint8)


def _encode_png_bytes(frame: Any) -> bytes:
    array = _frame_to_uint8_array(frame)

    if array.ndim == 2:
        color_type = 0
        height, width = array.shape
        payload = array
    else:
        height, width, channels = array.shape
        if channels == 1:
            color_type = 0
            payload = array[:, :, 0]
        elif channels == 3:
            color_type = 2
            payload = array
        elif channels == 4:
            color_type = 6
            payload = array
        else:
            raise RuntimeError(f"Unsupported channel count for debug image: {channels}")

    raw_rows = b"".join(b"\x00" + payload[row].tobytes() for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _write_png_chunk(b"IHDR", ihdr)
        + _write_png_chunk(b"IDAT", zlib.compress(raw_rows))
        + _write_png_chunk(b"IEND", b"")
    )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def _save_debug_frame(frame: Any, *, source_method: str, camera: Any = None) -> None:
    global _LAST_DEBUG_SAVE_SIGNATURE

    TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "saved_at": _now_isoformat(),
        "source_method": source_method,
        "shape": list(frame.shape) if hasattr(frame, "shape") else None,
        "dtype": str(frame.dtype) if hasattr(frame, "dtype") else None,
        "camera_repr": repr(camera) if camera is not None else None,
        "image_path": str(LATEST_INPUT_IMAGE_PATH),
    }

    try:
        if cv2 is not None:
            tmp_image_path = LATEST_INPUT_IMAGE_PATH.with_name(
                f"{LATEST_INPUT_IMAGE_PATH.stem}.tmp{LATEST_INPUT_IMAGE_PATH.suffix}"
            )
            success = cv2.imwrite(str(tmp_image_path), frame)
            if not success:
                raise RuntimeError(f"cv2.imwrite returned False for {tmp_image_path}")
            tmp_image_path.replace(LATEST_INPUT_IMAGE_PATH)
        else:
            _atomic_write_bytes(LATEST_INPUT_IMAGE_PATH, _encode_png_bytes(frame))

        metadata["save_status"] = "ok"
        save_signature = (source_method, str(LATEST_INPUT_IMAGE_PATH))
        if _LAST_DEBUG_SAVE_SIGNATURE != save_signature:
            print(f"[Camera Debug] Saved model input image via {source_method}: {LATEST_INPUT_IMAGE_PATH}")
            _LAST_DEBUG_SAVE_SIGNATURE = save_signature
    except Exception as exc:
        metadata["save_status"] = "error"
        metadata["save_error"] = str(exc)
        print(f"[Camera Debug] Failed to save input frame: {exc}")
    finally:
        tmp_metadata_path = LATEST_INPUT_METADATA_PATH.with_name(
            f"{LATEST_INPUT_METADATA_PATH.stem}.tmp{LATEST_INPUT_METADATA_PATH.suffix}"
        )
        tmp_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        tmp_metadata_path.replace(LATEST_INPUT_METADATA_PATH)


def _guess_extension_from_mime(mime_type: str | None) -> str:
    if not mime_type:
        return ".bin"
    mime_type = mime_type.lower()
    if "png" in mime_type:
        return ".png"
    if "jpeg" in mime_type or "jpg" in mime_type:
        return ".jpg"
    if "webp" in mime_type:
        return ".webp"
    if "gif" in mime_type:
        return ".gif"
    if "bmp" in mime_type:
        return ".bmp"
    return ".bin"


def _compute_frame_stats(frame: Any) -> dict[str, Any]:
    if np is None:
        return {
            "shape": list(frame.shape) if hasattr(frame, "shape") else None,
            "dtype": str(frame.dtype) if hasattr(frame, "dtype") else None,
        }

    array = np.asarray(frame)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": round(float(array.mean()), 3),
        "std": round(float(array.std()), 3),
        "min": int(array.min()),
        "max": int(array.max()),
    }


def _save_raw_camera_frame(
    frame: Any,
    *,
    source_method: str,
    camera_meta: dict[str, Any],
    blank_stats: str | None = None,
) -> None:
    TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    stats = _compute_frame_stats(frame)

    try:
        _atomic_write_bytes(LATEST_RAW_CAMERA_IMAGE_PATH, _encode_png_bytes(frame))
        save_status = "ok"
        save_error = None
    except Exception as exc:
        save_status = "error"
        save_error = str(exc)
        print(f"[Raw Camera] Failed to save raw camera frame: {exc}")

    metadata = {
        "saved_at": _now_isoformat(),
        "source_method": source_method,
        "camera_id": camera_meta.get("id"),
        "camera_type": camera_meta.get("type"),
        "selected_backend": camera_meta.get("selected_backend"),
        "selected_fourcc": camera_meta.get("selected_fourcc"),
        "validation_stats": camera_meta.get("validation_stats"),
        "current_frame_stats": stats,
        "blank_stats": blank_stats,
        "image_path": str(LATEST_RAW_CAMERA_IMAGE_PATH),
        "save_status": save_status,
        "save_error": save_error,
    }

    tmp_metadata_path = LATEST_RAW_CAMERA_METADATA_PATH.with_name(
        f"{LATEST_RAW_CAMERA_METADATA_PATH.stem}.tmp{LATEST_RAW_CAMERA_METADATA_PATH.suffix}"
    )
    tmp_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    tmp_metadata_path.replace(LATEST_RAW_CAMERA_METADATA_PATH)


def _save_llm_input_bytes(data: bytes, *, source_method: str, mime_type: str | None, note: str | None = None) -> None:
    TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ext = _guess_extension_from_mime(mime_type)
    image_path = LATEST_LLM_INPUT_STEM.with_suffix(ext)
    metadata = {
        "saved_at": _now_isoformat(),
        "source_method": source_method,
        "mime_type": mime_type,
        "byte_length": len(data),
        "image_path": str(image_path),
        "note": note,
    }

    _atomic_write_bytes(image_path, data)
    tmp_metadata_path = LATEST_LLM_INPUT_METADATA_PATH.with_name(
        f"{LATEST_LLM_INPUT_METADATA_PATH.stem}.tmp{LATEST_LLM_INPUT_METADATA_PATH.suffix}"
    )
    tmp_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    tmp_metadata_path.replace(LATEST_LLM_INPUT_METADATA_PATH)
    print(f"[LLM Debug] Saved exact model input image via {source_method}: {image_path}")


def _try_save_llm_content_block(block: Any, *, source_method: str) -> bool:
    if not isinstance(block, dict):
        return False

    block_type = block.get("type")
    if block_type == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url")
        else:
            url = image_url

        if isinstance(url, str) and url.startswith("data:image/"):
            header, encoded = url.split(",", 1)
            mime_type = header.split(";", 1)[0].split(":", 1)[1]
            _save_llm_input_bytes(
                base64.b64decode(encoded),
                source_method=source_method,
                mime_type=mime_type,
                note="Saved model input image with RoboCrew grid overlay.",
            )
            return True
        return False

    if block_type in {"image", "input_image"}:
        if block.get("source_type") == "base64" and isinstance(block.get("data"), str):
            _save_llm_input_bytes(
                base64.b64decode(block["data"]),
                source_method=source_method,
                mime_type=block.get("mime_type") or block.get("media_type"),
                note="Saved model input image with RoboCrew grid overlay.",
            )
            return True

        image_url = block.get("image_url")
        if isinstance(image_url, str) and image_url.startswith("data:image/"):
            header, encoded = image_url.split(",", 1)
            mime_type = header.split(";", 1)[0].split(":", 1)[1]
            _save_llm_input_bytes(
                base64.b64decode(encoded),
                source_method=source_method,
                mime_type=mime_type,
                note="Saved model input image with RoboCrew grid overlay.",
            )
            return True

    return False


def _payload_contains_image_block(payload: Any, *, seen: set[int] | None = None) -> bool:
    if seen is None:
        seen = set()

    payload_id = id(payload)
    if payload_id in seen:
        return False
    seen.add(payload_id)

    if payload is None:
        return False

    if isinstance(payload, dict):
        block_type = payload.get("type")
        if block_type == "image_url":
            image_url = payload.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if isinstance(url, str) and url.startswith("data:image/"):
                return True

        if block_type in {"image", "input_image"}:
            if payload.get("source_type") == "base64" and isinstance(payload.get("data"), str):
                return True

            image_url = payload.get("image_url")
            if isinstance(image_url, str) and image_url.startswith("data:image/"):
                return True

        return any(_payload_contains_image_block(value, seen=seen) for value in payload.values())

    if isinstance(payload, (list, tuple)):
        return any(_payload_contains_image_block(item, seen=seen) for item in payload)

    content = getattr(payload, "content", None)
    if content is not None and _payload_contains_image_block(content, seen=seen):
        return True

    messages = getattr(payload, "messages", None)
    if messages is not None and _payload_contains_image_block(messages, seen=seen):
        return True

    return False


def _select_latest_message_payload(payload: Any) -> Any:
    if not isinstance(payload, (list, tuple)):
        return payload

    for item in reversed(payload):
        content = getattr(item, "content", None)
        if content is not None and _payload_contains_image_block(content):
            return content

    return payload


def _save_llm_input_image_from_payload(payload: Any, *, source_method: str) -> bool:
    try:
        latest_payload = _select_latest_message_payload(payload)
        return _save_llm_input_image_from_payload_inner(latest_payload, source_method=source_method, seen=set())
    except Exception as exc:
        print(f"[LLM Debug] Failed to inspect model input payload: {exc}")
        return False


def _save_llm_input_image_from_payload_inner(payload: Any, *, source_method: str, seen: set[int]) -> bool:
    payload_id = id(payload)
    if payload_id in seen:
        return False
    seen.add(payload_id)

    if payload is None:
        return False

    if hasattr(payload, "to_messages") and callable(payload.to_messages):
        try:
            messages = payload.to_messages()
        except Exception:
            messages = None
        if messages is not None and _save_llm_input_image_from_payload_inner(
            messages,
            source_method=source_method,
            seen=seen,
        ):
            return True

    if isinstance(payload, dict):
        if _try_save_llm_content_block(payload, source_method=source_method):
            return True
        for value in payload.values():
            if _save_llm_input_image_from_payload_inner(value, source_method=source_method, seen=seen):
                return True
        return False

    if isinstance(payload, (list, tuple)):
        for item in payload:
            if _save_llm_input_image_from_payload_inner(item, source_method=source_method, seen=seen):
                return True
        return False

    content = getattr(payload, "content", None)
    if content is not None and _save_llm_input_image_from_payload_inner(content, source_method=source_method, seen=seen):
        return True

    messages = getattr(payload, "messages", None)
    if messages is not None and _save_llm_input_image_from_payload_inner(messages, source_method=source_method, seen=seen):
        return True

    return False


def _camera_wrapper(method: Any, wrapped_name: str):
    @wraps(method)
    def _wrapped(self, *args: Any, **kwargs: Any) -> Any:
        frame = method(self, *args, **kwargs)
        _save_debug_frame(frame, source_method=wrapped_name, camera=self)
        return frame

    return _wrapped


def _patch_robot_camera_debug_io() -> None:
    if getattr(RobotCamera, "_lerobot_debug_io_patched", False):
        return

    for method_name in ("read", "async_read", "read_latest", "capture", "get_frame", "get_image", "snap"):
        original_method = getattr(RobotCamera, method_name, None)
        if not callable(original_method):
            continue

        setattr(RobotCamera, method_name, _camera_wrapper(original_method, f"RobotCamera.{method_name}"))

    RobotCamera._lerobot_debug_io_patched = True


def _patch_camera_instance_methods(camera: Any, *, label: str) -> list[str]:
    patched_methods: list[str] = []
    already_patched = getattr(camera, "_lerobot_debug_instance_patched", set())
    candidate_names = (
        "read",
        "async_read",
        "read_latest",
        "capture",
        "get_frame",
        "get_image",
        "take_picture",
        "take_photo",
        "get_picture",
        "get_latest_frame",
        "get_latest_image",
        "snap",
    )

    for method_name in candidate_names:
        if method_name in already_patched:
            continue

        original_method = getattr(camera, method_name, None)
        if not callable(original_method):
            continue

        @wraps(original_method)
        def _wrapped(*args: Any, _method=original_method, _method_name=method_name, **kwargs: Any) -> Any:
            frame = _method(*args, **kwargs)
            _save_debug_frame(frame, source_method=f"{label}.{_method_name}", camera=camera)
            return frame

        setattr(camera, method_name, _wrapped)
        patched_methods.append(method_name)
        already_patched.add(method_name)

    camera._lerobot_debug_instance_patched = already_patched
    return patched_methods


def _instrument_camera_debug_io(camera: Any) -> None:
    patched = _patch_camera_instance_methods(camera, label=camera.__class__.__name__)
    if patched:
        print(f"[Camera Debug] Patched camera instance methods: {', '.join(patched)}")

    for nested_name in ("camera", "cam", "_camera", "cap"):
        nested = getattr(camera, nested_name, None)
        if nested is None:
            continue

        nested_patched = _patch_camera_instance_methods(nested, label=f"{camera.__class__.__name__}.{nested_name}")
        if nested_patched:
            print(
                f"[Camera Debug] Patched nested camera methods on {nested_name}: {', '.join(nested_patched)}"
            )


def _format_backend_label(config: OpenCVCameraConfig) -> str:
    return getattr(config.backend, "name", str(config.backend))


def _build_validated_opencv_camera(camera_id: int) -> tuple[OpenCVCamera, dict[str, Any]]:
    for cv_config in _build_opencv_candidate_configs(camera_id):
        instance = OpenCVCamera(cv_config)
        backend_label = _format_backend_label(cv_config)
        fourcc_label = cv_config.fourcc or "default"

        try:
            print(
                f"[Raw Camera] Trying OpenCV camera {camera_id} with "
                f"backend={backend_label}, fourcc={fourcc_label}"
            )
            instance.connect(warmup=True)
            validation_frame = instance.read()
            is_blank, stats_message = _is_blank_frame(validation_frame)
            if is_blank:
                print(
                    f"[Raw Camera] Rejected OpenCV camera {camera_id} for "
                    f"{backend_label}/{fourcc_label}: {stats_message}"
                )
                instance.disconnect()
                continue

            meta = {
                "type": "OpenCV",
                "id": camera_id,
                "selected_backend": backend_label,
                "selected_fourcc": fourcc_label,
                "validation_stats": stats_message,
            }
            _save_raw_camera_frame(
                validation_frame,
                source_method="validated_opencv.connect",
                camera_meta=meta,
                blank_stats=stats_message,
            )
            print(
                f"[Raw Camera] Selected backend={backend_label}, fourcc={fourcc_label}: {stats_message}"
            )
            return instance, meta
        except Exception as exc:
            print(
                f"[Raw Camera] Candidate backend={backend_label}, fourcc={fourcc_label} failed: {exc}"
            )
            try:
                if instance.is_connected:
                    instance.disconnect()
            except Exception:
                pass

    raise RuntimeError(
        "Failed to find a usable OpenCV camera configuration for "
        f"CAMERA_ID={camera_id}. Run "
        "`python -m lerobot.scripts.lerobot_find_cameras opencv --record-time-s 1` "
        "from this repository to inspect camera backends."
    )


class ValidatedMainCamera:
    """RoboCrew-compatible adapter around a validated local OpenCV camera."""

    def __init__(self, camera: OpenCVCamera, camera_meta: dict[str, Any]) -> None:
        self._camera = camera
        self.camera_meta = camera_meta
        self.camera_id = camera_meta.get("id")
        self.id = self.camera_id
        self.selected_backend = camera_meta.get("selected_backend")
        self.selected_fourcc = camera_meta.get("selected_fourcc")
        self.validation_stats = camera_meta.get("validation_stats")
        self.index_or_path = getattr(camera.config, "index_or_path", self.camera_id)
        self.backend = self.selected_backend
        self.fourcc = self.selected_fourcc
        self.camera = camera
        self.cam = camera
        self.capture = camera.videocapture
        self.cap = self.capture

    def __repr__(self) -> str:
        return (
            "ValidatedMainCamera("
            f"id={self.camera_id}, "
            f"backend={self.selected_backend}, "
            f"fourcc={self.selected_fourcc})"
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._camera, name)

    @property
    def is_connected(self) -> bool:
        return self._camera.is_connected

    def _refresh_capture_handle(self) -> None:
        self.capture = self._camera.videocapture
        self.cap = self.capture

    def _capture_and_record(self, method_name: str, *args: Any, source_method: str, **kwargs: Any) -> Any:
        method = getattr(self._camera, method_name)
        frame = method(*args, **kwargs)
        is_blank, stats_message = _is_blank_frame(frame)
        _save_raw_camera_frame(
            frame,
            source_method=source_method,
            camera_meta=self.camera_meta,
            blank_stats=stats_message,
        )
        if is_blank:
            print(f"[Raw Camera] Warning: blank frame detected during {source_method}: {stats_message}")
        return frame

    def _frame_to_data_url(self, frame: Any) -> str:
        png_bytes = _encode_png_bytes(frame)
        encoded = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _build_overlay_frame(
        self,
        frame: Any,
        *,
        camera_fov: int = 120,
        center_angle: int = 0,
        navigation_mode: str = "normal",
    ) -> Any:
        if cv2 is None:
            raise RuntimeError("opencv-python is required to encode RoboCrew camera frames.")

        overlay_frame = np.asarray(frame).copy() if np is not None else frame.copy()
        if getattr(self._camera, "color_mode", None) == ColorMode.RGB:
            overlay_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR)

        return basic_augmentation(
            overlay_frame,
            h_fov=camera_fov,
            center_angle=center_angle,
            navigation_mode=navigation_mode,
        )

    def capture_image(
        self,
        camera_fov: int = 120,
        center_angle: int = 0,
        navigation_mode: str = "normal",
    ) -> bytes:
        if cv2 is None:
            raise RuntimeError("opencv-python is required to encode RoboCrew camera frames.")

        frame = self._capture_and_record("read", source_method="ValidatedMainCamera.capture_image.raw")
        overlay_frame = self._build_overlay_frame(
            frame,
            camera_fov=camera_fov,
            center_angle=center_angle,
            navigation_mode=navigation_mode,
        )
        success, buffer = cv2.imencode(".jpg", overlay_frame)
        if not success:
            raise RuntimeError("cv2.imencode('.jpg', overlay_frame) returned False.")
        return buffer.tobytes()

    def read(self, *args: Any, **kwargs: Any) -> Any:
        return self._capture_and_record("read", source_method="ValidatedMainCamera.read", *args, **kwargs)

    def async_read(self, *args: Any, **kwargs: Any) -> Any:
        return self._capture_and_record("async_read", source_method="ValidatedMainCamera.async_read", *args, **kwargs)

    def read_latest(self, *args: Any, **kwargs: Any) -> Any:
        return self._capture_and_record("read_latest", source_method="ValidatedMainCamera.read_latest", *args, **kwargs)

    def get_frame(self, *args: Any, **kwargs: Any) -> Any:
        return self.read(*args, **kwargs)

    def get_image(self, *args: Any, **kwargs: Any) -> Any:
        return self.read(*args, **kwargs)

    def snap(self, *args: Any, **kwargs: Any) -> Any:
        return self.read(*args, **kwargs)

    def take_picture(self, *args: Any, **kwargs: Any) -> Any:
        return self.read(*args, **kwargs)

    def take_photo(self, *args: Any, **kwargs: Any) -> Any:
        return self.read(*args, **kwargs)

    def get_image_base64(self, *args: Any, **kwargs: Any) -> str:
        return self._frame_to_data_url(self.read(*args, **kwargs))

    def capture_base64(self, *args: Any, **kwargs: Any) -> str:
        return self._frame_to_data_url(self.read(*args, **kwargs))

    def read_base64(self, *args: Any, **kwargs: Any) -> str:
        return self._frame_to_data_url(self.read(*args, **kwargs))

    def reopen(self) -> None:
        try:
            if self._camera.is_connected:
                self._camera.disconnect()
        except DeviceNotConnectedError:
            pass
        self._camera.connect(warmup=True)
        self._refresh_capture_handle()

    def disconnect(self) -> None:
        try:
            if self._camera.is_connected:
                self._camera.disconnect()
        except DeviceNotConnectedError:
            pass
        finally:
            self._refresh_capture_handle()

    def release(self) -> None:
        self.disconnect()

    def close(self) -> None:
        self.disconnect()


def build_main_camera() -> ValidatedMainCamera:
    raw_camera, camera_meta = _build_validated_opencv_camera(CAMERA_ID)
    return ValidatedMainCamera(raw_camera, camera_meta)


def _extract_text_fragments(payload: Any) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, str):
        text = payload.strip()
        return [text] if text else []
    if isinstance(payload, dict):
        fragments: list[str] = []
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            fragments.append(text.strip())
        if "content" in payload:
            fragments.extend(_extract_text_fragments(payload["content"]))
        return fragments
    if isinstance(payload, (list, tuple)):
        fragments: list[str] = []
        for item in payload:
            fragments.extend(_extract_text_fragments(item))
        return fragments
    return []


def _extract_text_from_response(response: Any) -> str | None:
    if isinstance(response, str):
        fragments = _extract_text_fragments(response)
    else:
        content = getattr(response, "content", response)
        fragments = _extract_text_fragments(content)

    if not fragments:
        return None

    text = "\n".join(fragment for fragment in fragments if fragment)
    return text if text else None


def _print_llm_text_output(response: Any) -> None:
    text = _extract_text_from_response(response)
    if text:
        print(f"[LLM]\n{text}")
    else:
        print("[LLM] [LLM text empty]")


class DebugChatModel:
    """Wrap a LangChain chat model and print only the text content of each response."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    @property
    def __class__(self):
        return self._wrapped.__class__

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def __repr__(self) -> str:
        return repr(self._wrapped)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.invoke(*args, **kwargs)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            _save_llm_input_image_from_payload(args[0], source_method="invoke")
        if "input" in kwargs:
            _save_llm_input_image_from_payload(kwargs["input"], source_method="invoke")
        response = self._wrapped.invoke(*args, **kwargs)
        _print_llm_text_output(response)
        return response

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            _save_llm_input_image_from_payload(args[0], source_method="ainvoke")
        if "input" in kwargs:
            _save_llm_input_image_from_payload(kwargs["input"], source_method="ainvoke")
        response = await self._wrapped.ainvoke(*args, **kwargs)
        _print_llm_text_output(response)
        return response

    def batch(self, *args: Any, **kwargs: Any) -> Any:
        responses = self._wrapped.batch(*args, **kwargs)
        for response in responses:
            _print_llm_text_output(response)
        return responses

    async def abatch(self, *args: Any, **kwargs: Any) -> Any:
        responses = await self._wrapped.abatch(*args, **kwargs)
        for response in responses:
            _print_llm_text_output(response)
        return responses

    def stream(self, *args: Any, **kwargs: Any):
        if args:
            _save_llm_input_image_from_payload(args[0], source_method="stream")
        if "input" in kwargs:
            _save_llm_input_image_from_payload(kwargs["input"], source_method="stream")
        fragments: list[str] = []
        for chunk in self._wrapped.stream(*args, **kwargs):
            text = _extract_text_from_response(chunk)
            if text:
                fragments.append(text)
            yield chunk

        if fragments:
            streamed_text = "\n".join(fragments)
            print(f"[LLM]\n{streamed_text}")
        else:
            print("[LLM] [LLM text empty]")

    async def astream(self, *args: Any, **kwargs: Any):
        if args:
            _save_llm_input_image_from_payload(args[0], source_method="astream")
        if "input" in kwargs:
            _save_llm_input_image_from_payload(kwargs["input"], source_method="astream")
        fragments: list[str] = []
        async for chunk in self._wrapped.astream(*args, **kwargs):
            text = _extract_text_from_response(chunk)
            if text:
                fragments.append(text)
            yield chunk

        if fragments:
            streamed_text = "\n".join(fragments)
            print(f"[LLM]\n{streamed_text}")
        else:
            print("[LLM] [LLM text empty]")

    def bind(self, *args: Any, **kwargs: Any) -> "DebugChatModel":
        return DebugChatModel(self._wrapped.bind(*args, **kwargs))

    def bind_tools(self, *args: Any, **kwargs: Any) -> "DebugChatModel":
        return DebugChatModel(self._wrapped.bind_tools(*args, **kwargs))

    def with_structured_output(self, *args: Any, **kwargs: Any) -> "DebugChatModel":
        return DebugChatModel(self._wrapped.with_structured_output(*args, **kwargs))


def _patched_init_chat_model(model: str, *args, **kwargs):
    if model in {"kimi-k2.5", "openai:kimi-k2.5"}:
        kwargs.setdefault("base_url", MOONSHOT_BASE_URL)
        kwargs.setdefault("api_key", moonshot_api_key)
        if ":" not in model:
            model = "openai:kimi-k2.5"
    return DebugChatModel(lc_init_chat_model(model, *args, **kwargs))


robocrew_llm_module.init_chat_model = _patched_init_chat_model
_patch_robot_camera_debug_io()
LLMAgent = robocrew_llm_module.LLMAgent


def build_robot() -> XLerobot2Wheels:
    robot_config = XLerobot2WheelsConfig(
        id=ROBOT_ID,
        port1=PORT1,
        port2=PORT2,
    )
    robot = XLerobot2Wheels(robot_config)
    robot.connect()
    return robot


def _print_runtime_environment() -> None:
    print(f"[Env] Python executable: {sys.executable}")
    print(f"[Env] lerobot module: {LOCAL_LEROBOT_INIT_PATH}")
    print(f"[Env] robocrew memory module: {Path(robocrew_memory.__file__).resolve()}")
    print(
        "[Env] Recommended invocation: "
        "conda activate lerobot && cd C:\\projects\\lerobot && "
        "python examples/shield/move_to_you.py"
    )


def main() -> None:
    main_camera = None
    servo_controler = None

    try:
        _print_runtime_environment()
        main_camera = build_main_camera()
        robot = build_robot()
        servo_controler = TwoWheelsServoAdapter(
            robot,
            right_arm_wheel_usb=PORT2,
            linear_speed_mps=LINEAR_SPEED_MPS,
            angular_speed_dps=ANGULAR_SPEED_DPS,
        )

        tools = [
            create_move_forward(servo_controler),
            create_turn_left(servo_controler),
            create_turn_right(servo_controler),
        ]

        print(f"[Debug] Model: {MODEL_NAME}")
        print(f"[Debug] Task: {TASK}")
        print(f"[Debug] Camera ID: {CAMERA_ID}")
        print(
            f"[Debug] Raw camera backend/fourcc: "
            f"{main_camera.selected_backend}/{main_camera.selected_fourcc}"
        )
        print(f"[Debug] Raw camera validation stats: {main_camera.validation_stats}")
        print(f"[Debug] Saving latest raw camera image to: {LATEST_RAW_CAMERA_IMAGE_PATH}")
        print(f"[Debug] Saving latest model input image to: {LATEST_LLM_INPUT_STEM}.*")

        agent = LLMAgent(
            model=MODEL_NAME,
            tools=tools,
            main_camera=main_camera,
            servo_controler=servo_controler,
        )
        agent.task = TASK
        agent.go()
    finally:
        if servo_controler is not None:
            servo_controler.disconnect()
        if main_camera is not None:
            try:
                main_camera.disconnect()
            except Exception:
                close = getattr(main_camera, "close", None)
                if callable(close):
                    close()


if __name__ == "__main__":
    main()
