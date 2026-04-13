from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
SHIELD_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SRC_PATH = REPO_ROOT / "src"
EXPECTED_LEROBOT_INIT_PATH = (SRC_PATH / "lerobot" / "__init__.py").resolve()
DOTENV_PATH = SHIELD_DIR / ".env"

DEFAULT_CAMERA_ID = 1
DEFAULT_ROBOT_ID = "my_xlerobot_2wheels_lab"
DEFAULT_PORT1 = "COM5"  # left arm + head
DEFAULT_PORT2 = "COM4"  # right arm + 2-wheel base
DEFAULT_LINEAR_SPEED_MPS = 0.10
DEFAULT_ANGULAR_SPEED_DPS = 30.0
DEFAULT_FACE_TO_BODY_WIDTH_MULTIPLIER = 2.4
DEFAULT_FACE_TO_BODY_HEIGHT_MULTIPLIER = 4.8
DEFAULT_FACE_TO_BODY_TOP_SHIFT_RATIO = 0.35

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
INVALID_PERSON_NAME_CHARS = set('<>:"/\\|?*')


if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import lerobot as local_lerobot

LOCAL_LEROBOT_INIT_PATH = Path(local_lerobot.__file__).resolve()

if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH, override=False)

from lerobot.cameras.configs import ColorMode, Cv2Backends
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels, XLerobot2WheelsConfig, TwoWheelsServoAdapter
from lerobot.utils.errors import DeviceNotConnectedError

try:
    from lerobot.scripts.lerobot_find_cameras import is_blank_frame as _is_blank_frame
except Exception:

    def _is_blank_frame(img_array: Any) -> tuple[bool, str]:
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


def assert_local_lerobot_source(context: str) -> None:
    if LOCAL_LEROBOT_INIT_PATH != EXPECTED_LEROBOT_INIT_PATH:
        raise RuntimeError(
            f"{context} must run against the local repository source, but imported "
            f"lerobot from {LOCAL_LEROBOT_INIT_PATH}. Expected {EXPECTED_LEROBOT_INIT_PATH}. "
            "Please run: conda activate lerobot && cd C:\\projects\\lerobot && "
            "python -m pip install -e ."
        )


def _format_backend_label(config: OpenCVCameraConfig) -> str:
    return getattr(config.backend, "name", str(config.backend))


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_debug_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_debug_image(path: Path, image_bgr: np.ndarray | None) -> None:
    if image_bgr is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iter_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_person_name(raw_value: str) -> str:
    person_name = raw_value.strip()
    if not person_name:
        raise ValueError("Person name cannot be empty.")
    if person_name in {".", ".."}:
        raise ValueError("Person name cannot be '.' or '..'.")
    if person_name.lower() == "archive":
        raise ValueError("Person name 'archive' is reserved.")
    if any(char in INVALID_PERSON_NAME_CHARS for char in person_name):
        raise ValueError(
            "Person name cannot contain any of these characters: "
            + "".join(sorted(INVALID_PERSON_NAME_CHARS))
        )
    if ".." in person_name:
        raise ValueError("Person name cannot contain '..'.")
    return person_name


def prompt_person_name(prompt_text: str = "Target person name: ") -> str:
    return validate_person_name(input(prompt_text))


def list_available_person_names(reference_root: Path) -> list[str]:
    if not reference_root.exists():
        return []
    return sorted(
        path.name
        for path in reference_root.iterdir()
        if path.is_dir() and path.name != "archive"
    )


def archive_person_directory(person_dir: Path, archive_root: Path, *, timestamp: str | None = None) -> Path | None:
    if not person_dir.exists() or not any(person_dir.iterdir()):
        return None

    archive_destination = archive_root / person_dir.name / (timestamp or current_timestamp())
    archive_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(person_dir), str(archive_destination))
    return archive_destination


def clip_box(
    x: float,
    y: float,
    w: float,
    h: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    left = max(0, int(round(x)))
    top = max(0, int(round(y)))
    right = min(frame_width, int(round(x + w)))
    bottom = min(frame_height, int(round(y + h)))
    clipped_w = right - left
    clipped_h = bottom - top
    if clipped_w <= 0 or clipped_h <= 0:
        return None
    return left, top, clipped_w, clipped_h


def expand_face_to_body_box(
    face_x: float,
    face_y: float,
    face_w: float,
    face_h: float,
    frame_width: int,
    frame_height: int,
    *,
    width_multiplier: float = DEFAULT_FACE_TO_BODY_WIDTH_MULTIPLIER,
    height_multiplier: float = DEFAULT_FACE_TO_BODY_HEIGHT_MULTIPLIER,
    top_shift_ratio: float = DEFAULT_FACE_TO_BODY_TOP_SHIFT_RATIO,
) -> tuple[int, int, int, int] | None:
    return clip_box(
        face_x - ((width_multiplier - 1.0) * face_w / 2.0),
        face_y - (top_shift_ratio * face_h),
        face_w * width_multiplier,
        face_h * height_multiplier,
        frame_width,
        frame_height,
    )


@dataclass
class ValidatedLocalCamera:
    _camera: OpenCVCamera
    camera_id: int | str | None
    selected_backend: str
    selected_fourcc: str
    validation_stats: str

    def read_bgr(self) -> np.ndarray:
        frame = self._camera.read()
        is_blank, stats_message = _is_blank_frame(frame)
        if is_blank:
            raise RuntimeError(f"ValidatedLocalCamera.read_bgr: {stats_message}")

        frame_bgr = np.asarray(frame).copy()
        if getattr(self._camera, "color_mode", None) == ColorMode.RGB:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
        return frame_bgr

    def reopen(self) -> None:
        try:
            if self._camera.is_connected:
                self._camera.disconnect()
        except DeviceNotConnectedError:
            pass
        self._camera.connect(warmup=True)

    def disconnect(self) -> None:
        try:
            if self._camera.is_connected:
                self._camera.disconnect()
        except DeviceNotConnectedError:
            pass


def build_camera(camera_id: int = DEFAULT_CAMERA_ID) -> ValidatedLocalCamera:
    config = OpenCVCameraConfig(
        index_or_path=camera_id,
        color_mode=ColorMode.RGB,
        backend=Cv2Backends.DSHOW,
    )
    camera = OpenCVCamera(config)
    backend_label = _format_backend_label(config)
    fourcc_label = config.fourcc or "default"

    print(
        f"[Camera] Opening OpenCV camera {camera_id} with "
        f"backend={backend_label}, fourcc={fourcc_label}"
    )

    camera.connect(warmup=True)
    validation_frame = camera.read()
    is_blank, stats_message = _is_blank_frame(validation_frame)
    if is_blank:
        camera.disconnect()
        raise RuntimeError(
            f"Fixed camera configuration {backend_label}/{fourcc_label} failed validation: {stats_message}"
        )

    print(f"[Camera] Using backend={backend_label}, fourcc={fourcc_label}: {stats_message}")
    return ValidatedLocalCamera(
        _camera=camera,
        camera_id=camera_id,
        selected_backend=backend_label,
        selected_fourcc=fourcc_label,
        validation_stats=stats_message,
    )


def build_robot(
    robot_id: str = DEFAULT_ROBOT_ID,
    port1: str = DEFAULT_PORT1,
    port2: str = DEFAULT_PORT2,
) -> XLerobot2Wheels:
    robot_config = XLerobot2WheelsConfig(
        id=robot_id,
        port1=port1,
        port2=port2,
    )
    robot = XLerobot2Wheels(robot_config)
    robot.connect()
    return robot


def build_servo_controller(
    robot: XLerobot2Wheels,
    *,
    port2: str = DEFAULT_PORT2,
    linear_speed_mps: float = DEFAULT_LINEAR_SPEED_MPS,
    angular_speed_dps: float = DEFAULT_ANGULAR_SPEED_DPS,
    max_distance_per_step_m: float,
) -> TwoWheelsServoAdapter:
    return TwoWheelsServoAdapter(
        robot,
        right_arm_wheel_usb=port2,
        linear_speed_mps=linear_speed_mps,
        angular_speed_dps=angular_speed_dps,
        max_distance_per_step_m=max_distance_per_step_m,
    )
