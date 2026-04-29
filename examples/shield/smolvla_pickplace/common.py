from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SHIELD_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SRC_PATH = REPO_ROOT / "src"
EXPECTED_LEROBOT_INIT_PATH = (SRC_PATH / "lerobot" / "__init__.py").resolve()
DOTENV_PATH = SHIELD_DIR / ".env"

DEFAULT_MODEL_ID = "Grigorij/smolvla_collect_tissues"  #"Sa74ll/smolvla_so101_pickandplace"
DEFAULT_TASK = "Collect the tissues."
DEFAULT_ROBOT_ID = "my_xlerobot_2wheels_lab"
DEFAULT_PORT1 = "COM5"
DEFAULT_PORT2 = "COM4"
DEFAULT_DEBUG_ROOT = SCRIPT_DIR / "debug"
DEFAULT_TERMINAL_DEBUG_EVERY = 1
DEFAULT_DEBUG_WRITE_EVERY = 1

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_VALIDATION_ATTEMPTS = 5
CAMERA_VALIDATION_RETRY_DELAY_S = 0.20
RUNTIME_CAMERA_READ_ATTEMPTS = 3
RUNTIME_CAMERA_MAX_AGE_MS = 250

WINDOW_NAME = "SmolVLA Collect Dry Run"
PREVIEW_TILE_WIDTH = 480
PREVIEW_TILE_HEIGHT = 360
PREVIEW_PANEL_WIDTH = 520
PREVIEW_TEXT_LINE_HEIGHT = 22
PREVIEW_TEXT_FONT_SCALE = 0.52
PREVIEW_TEXT_MARGIN_X = 12
PREVIEW_TEXT_MARGIN_Y = 24

CAMERA_SLOT_SPECS = (
    {
        "short_key": "camera1",
        "full_key": "observation.images.camera1",
        "semantic_label": "Head",
        "default_source": 1,
    },
    {
        "short_key": "camera2",
        "full_key": "observation.images.camera2",
        "semantic_label": "Right Arm",
        "default_source": 2,
    },
)
CAMERA_SHORT_KEY_TO_SPEC = {spec["short_key"]: spec for spec in CAMERA_SLOT_SPECS}
SUPPORTED_CAMERA_FULL_KEYS = {spec["full_key"] for spec in CAMERA_SLOT_SPECS}
IGNORED_CHECKPOINT_CAMERA_FULL_KEYS = {"observation.images.camera3"}
ACCEPTED_CHECKPOINT_CAMERA_FULL_KEYS = SUPPORTED_CAMERA_FULL_KEYS | IGNORED_CHECKPOINT_CAMERA_FULL_KEYS
DEFAULT_CAMERA_MAPPING_TEXT = ", ".join(
    f"{spec['short_key']}={spec['default_source']}" for spec in CAMERA_SLOT_SPECS
)

RIGHT_ARM_6D_STATE_NAMES = (
    "right_arm_shoulder_pan.pos",
    "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos",
    "right_arm_gripper.pos",
)
JOINT_SHORT_LABELS = {
    "right_arm_shoulder_pan.pos": "pan",
    "right_arm_shoulder_lift.pos": "lift",
    "right_arm_elbow_flex.pos": "elbow",
    "right_arm_wrist_flex.pos": "wrist",
    "right_arm_wrist_roll.pos": "roll",
    "right_arm_gripper.pos": "grip",
}


if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@dataclass(frozen=True)
class CameraAssignment:
    full_key: str
    short_key: str
    semantic_label: str
    source: int | Path


@dataclass(frozen=True)
class RuntimeArtifacts:
    args: argparse.Namespace
    run_dir: Path
    device: torch.device
    camera_assignments: list[CameraAssignment]
    dataset_features: dict[str, dict[str, Any]]
    policy: Any
    preprocessor: Any
    postprocessor: Any
    robot: XLerobot2Wheels
    cameras: list["ValidatedLocalCamera"]
    ignored_checkpoint_cameras: set[str]


@dataclass(frozen=True)
class RuntimeObservation:
    raw_observation: dict[str, Any]
    frames_rgb: dict[str, np.ndarray]
    observation_ms: float
    camera_ms: float


@dataclass(frozen=True)
class ActionSelection:
    action_vector: np.ndarray
    named_action: dict[str, float]
    queue_depth_before: int | None
    queue_depth_after: int | None
    preprocess_ms: float
    model_ms: float
    postprocess_ms: float


@dataclass(frozen=True)
class JointDebugSnapshot:
    current_state: dict[str, float]
    predicted_action: dict[str, float]
    predicted_delta: dict[str, float]


def parse_non_negative_int(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer, got '{raw_value}'.") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative integer.")
    return value


def parse_positive_int(raw_value: str) -> int:
    value = parse_non_negative_int(raw_value)
    if value == 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone dry-run for a SmolVLA SO-101 pick-and-place checkpoint on the right arm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="SmolVLA model repo id or local path.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Natural-language task passed to the policy.")
    parser.add_argument("--device", default=None, help="Torch device override, such as cuda, cpu, or mps.")
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID, help="Robot id used for calibration lookup.")
    parser.add_argument("--port1", default=DEFAULT_PORT1, help="Serial port for left arm + head.")
    parser.add_argument("--port2", default=DEFAULT_PORT2, help="Serial port for right arm + base.")
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        metavar="CAMERA=INDEX",
        help=(
            "Override physical camera sources. Accepts camera1/camera2 or "
            "observation.images.camera1/camera2. "
            f"If omitted, defaults are used: {DEFAULT_CAMERA_MAPPING_TEXT}."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=parse_non_negative_int,
        default=0,
        help="Maximum dry-run steps. Use 0 to run until q, ESC, or Ctrl+C.",
    )
    parser.add_argument(
        "--debug-root",
        type=Path,
        default=DEFAULT_DEBUG_ROOT,
        help="Root directory for debug/<run_id> outputs.",
    )
    parser.add_argument(
        "--debug-write-every",
        type=parse_positive_int,
        default=DEFAULT_DEBUG_WRITE_EVERY,
        help="Write latest preview/runtime files every N dry-run steps.",
    )
    parser.add_argument(
        "--terminal-debug-every",
        type=parse_positive_int,
        default=DEFAULT_TERMINAL_DEBUG_EVERY,
        help="Print joint/action diagnostics every N dry-run steps.",
    )
    parser.add_argument("--no-window", action="store_true", help="Do not open the OpenCV preview window.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Ask Hugging Face loaders to use local files only where supported.",
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional Hugging Face cache directory.")
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def load_runtime_dependencies() -> None:
    global ACTION
    global OBS_IMAGES
    global OBS_STR
    global ColorMode
    global Cv2Backends
    global DeviceNotConnectedError
    global OpenCVCamera
    global OpenCVCameraConfig
    global PreTrainedConfig
    global XLerobot2Wheels
    global XLerobot2WheelsConfig
    global _is_blank_frame
    global build_inference_frame
    global combine_feature_dicts
    global cv2
    global hw_to_dataset_features
    global load_dotenv
    global make_pre_post_processors
    global make_robot_action
    global np
    global torch

    try:
        import cv2 as _cv2
        import numpy as _np
        import torch as _torch
        from dotenv import load_dotenv as _load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "Missing a runtime dependency for the dry-run. Activate the lerobot environment first, "
            "for example: conda activate lerobot"
        ) from exc

    import lerobot as local_lerobot

    local_lerobot_init_path = Path(local_lerobot.__file__).resolve()
    if local_lerobot_init_path != EXPECTED_LEROBOT_INIT_PATH:
        raise RuntimeError(
            "This script must run against the local repository source, but imported "
            f"lerobot from {local_lerobot_init_path}. Expected {EXPECTED_LEROBOT_INIT_PATH}. "
            "Run from the repo root after installing editable sources: "
            "python -m pip install -e ."
        )

    if DOTENV_PATH.exists():
        _load_dotenv(DOTENV_PATH, override=False)

    from lerobot.cameras.configs import ColorMode as _ColorMode
    from lerobot.cameras.configs import Cv2Backends as _Cv2Backends
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera as _OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig as _OpenCVCameraConfig
    from lerobot.configs.policies import PreTrainedConfig as _PreTrainedConfig
    from lerobot.datasets.feature_utils import combine_feature_dicts as _combine_feature_dicts
    from lerobot.datasets.feature_utils import hw_to_dataset_features as _hw_to_dataset_features
    from lerobot.policies.factory import make_pre_post_processors as _make_pre_post_processors
    from lerobot.policies.utils import build_inference_frame as _build_inference_frame
    from lerobot.policies.utils import make_robot_action as _make_robot_action
    from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels as _XLerobot2Wheels
    from lerobot.robots.xlerobot_2wheels import XLerobot2WheelsConfig as _XLerobot2WheelsConfig
    from lerobot.utils.constants import ACTION as _ACTION
    from lerobot.utils.constants import OBS_IMAGES as _OBS_IMAGES
    from lerobot.utils.constants import OBS_STR as _OBS_STR
    from lerobot.utils.errors import DeviceNotConnectedError as _DeviceNotConnectedError

    try:
        from lerobot.scripts.lerobot_find_cameras import is_blank_frame as blank_frame_fn
    except Exception:

        def blank_frame_fn(img_array: Any) -> tuple[bool, str]:
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

    ACTION = _ACTION
    OBS_IMAGES = _OBS_IMAGES
    OBS_STR = _OBS_STR
    ColorMode = _ColorMode
    Cv2Backends = _Cv2Backends
    DeviceNotConnectedError = _DeviceNotConnectedError
    OpenCVCamera = _OpenCVCamera
    OpenCVCameraConfig = _OpenCVCameraConfig
    PreTrainedConfig = _PreTrainedConfig
    XLerobot2Wheels = _XLerobot2Wheels
    XLerobot2WheelsConfig = _XLerobot2WheelsConfig
    _is_blank_frame = blank_frame_fn
    build_inference_frame = _build_inference_frame
    combine_feature_dicts = _combine_feature_dicts
    cv2 = _cv2
    hw_to_dataset_features = _hw_to_dataset_features
    load_dotenv = _load_dotenv
    make_pre_post_processors = _make_pre_post_processors
    make_robot_action = _make_robot_action
    np = _np
    torch = _torch


def make_run_dir(debug_root: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = debug_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def normalize_camera_key(raw_key: str, expected_keys: set[str]) -> str:
    candidate = raw_key.strip()
    if candidate in expected_keys:
        return candidate
    prefixed_candidate = candidate
    if not candidate.startswith(f"{OBS_IMAGES}."):
        prefixed_candidate = f"{OBS_IMAGES}.{candidate}"
    if prefixed_candidate in expected_keys:
        return prefixed_candidate
    raise ValueError(
        f"Unknown camera key '{raw_key}'. Expected one of: {', '.join(sorted(expected_keys)) or '(none)'}"
    )


def parse_camera_source(raw_value: str) -> int | Path:
    value = raw_value.strip()
    if value == "":
        raise ValueError("Camera source cannot be empty.")
    try:
        return int(value)
    except ValueError:
        return Path(value)


def parse_camera_assignments(
    raw_assignments: list[str],
    expected_image_features: dict[str, Any],
) -> tuple[list[CameraAssignment], set[str]]:
    expected_keys = set(expected_image_features)
    extra_keys = expected_keys - ACCEPTED_CHECKPOINT_CAMERA_FULL_KEYS
    if extra_keys:
        raise ValueError(
            "Unsupported checkpoint image features for this standalone dry-run.\n"
            f"- Accepted checkpoint keys: {', '.join(sorted(ACCEPTED_CHECKPOINT_CAMERA_FULL_KEYS))}\n"
            f"- Extra keys: {', '.join(sorted(extra_keys)) or 'None'}"
        )
    if not expected_keys:
        raise ValueError(
            "Checkpoint does not declare image features. Expected at least one of: "
            + ", ".join(sorted(ACCEPTED_CHECKPOINT_CAMERA_FULL_KEYS))
        )

    source_by_key: dict[str, int | Path] = {
        spec["full_key"]: spec["default_source"] for spec in CAMERA_SLOT_SPECS if spec["full_key"] in expected_keys
    }
    seen_keys: set[str] = set()

    for raw_assignment in raw_assignments:
        if "=" not in raw_assignment:
            raise ValueError(f"Invalid --camera value '{raw_assignment}'. Expected CAMERA=INDEX.")
        raw_key, raw_value = raw_assignment.split("=", 1)
        full_key = normalize_camera_key(raw_key, expected_keys)
        if full_key not in SUPPORTED_CAMERA_FULL_KEYS:
            raise ValueError(f"'{full_key}' is ignored by this dry-run and cannot be mapped to hardware.")
        if full_key in seen_keys:
            raise ValueError(f"Duplicate --camera mapping for '{full_key}'.")
        source_by_key[full_key] = parse_camera_source(raw_value)
        seen_keys.add(full_key)

    assignments: list[CameraAssignment] = []
    for spec in CAMERA_SLOT_SPECS:
        full_key = spec["full_key"]
        if full_key not in expected_keys:
            continue
        shape = expected_image_features[full_key].shape
        if len(shape) != 3:
            raise ValueError(f"Unsupported image feature shape for '{full_key}': {shape}")
        assignments.append(
            CameraAssignment(
                full_key=full_key,
                short_key=spec["short_key"],
                semantic_label=spec["semantic_label"],
                source=source_by_key[full_key],
            )
        )

    if not assignments:
        raise ValueError(
            "Checkpoint does not declare camera1/camera2. This dry-run ignores camera3 and needs "
            "at least one physical camera."
        )

    ignored_keys = expected_keys & IGNORED_CHECKPOINT_CAMERA_FULL_KEYS
    return assignments, ignored_keys


def build_policy_config(args: argparse.Namespace) -> PreTrainedConfig:
    cli_overrides: list[str] = []
    if args.device:
        cli_overrides.append(f"--device={args.device}")

    try:
        config = PreTrainedConfig.from_pretrained(
            args.model_id,
            cli_overrides=cli_overrides,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the SmolVLA checkpoint config. If Hugging Face is unavailable, "
            "download the model first or rerun with --model-id pointing at a local checkpoint directory.\n"
            f"model_id={args.model_id}"
        ) from exc

    if config.type != "smolvla":
        raise ValueError(f"Expected a SmolVLA checkpoint, but '{args.model_id}' is type '{config.type}'.")
    state_feature = config.robot_state_feature
    action_feature = config.action_feature
    if state_feature is None:
        raise ValueError("Checkpoint does not declare observation.state in input_features.")
    if action_feature is None:
        raise ValueError("Checkpoint does not declare action in output_features.")

    state_dim = int(state_feature.shape[0])
    action_dim = int(action_feature.shape[0])
    if (state_dim, action_dim) != (6, 6):
        raise ValueError(
            "This standalone test only supports a right-arm SO-101 6D layout.\n"
            f"- Checkpoint observation.state dim: {state_dim}\n"
            f"- Checkpoint action dim: {action_dim}"
        )
    return config


def load_policy_artifacts(args: argparse.Namespace, config: PreTrainedConfig) -> tuple[Any, Any, Any]:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    try:
        print("[Startup] Loading SmolVLA policy weights...")
        policy = SmolVLAPolicy.from_pretrained(
            args.model_id,
            config=config,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load SmolVLA policy weights. If the network is unavailable, "
            "download/cache the model or pass a local checkpoint directory with --model-id."
        ) from exc

    try:
        print("[Startup] Loading policy processor pipelines...")
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=args.model_id,
            preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load SmolVLA processor files. If the checkpoint is local, make sure it contains "
            "policy_preprocessor.json and policy_postprocessor.json."
        ) from exc

    policy.reset()
    return policy, preprocessor, postprocessor


def build_robot_instance(args: argparse.Namespace) -> XLerobot2Wheels:
    robot_config = XLerobot2WheelsConfig(
        id=args.robot_id,
        port1=args.port1,
        port2=args.port2,
        cameras={},
    )
    return XLerobot2Wheels(robot_config)


def build_dataset_features(
    cameras: list["ValidatedLocalCamera"],
    frames_rgb: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    observation_hw_features: dict[str, type | tuple[int, int, int]] = dict.fromkeys(
        RIGHT_ARM_6D_STATE_NAMES, float
    )
    observation_hw_features.update(
        {
            camera.assignment.short_key: tuple(frames_rgb[camera.assignment.short_key].shape)
            for camera in cameras
        }
    )
    action_hw_features: dict[str, type] = dict.fromkeys(RIGHT_ARM_6D_STATE_NAMES, float)
    return combine_feature_dicts(
        hw_to_dataset_features(action_hw_features, ACTION),
        hw_to_dataset_features(observation_hw_features, OBS_STR, use_video=False),
    )


def build_opencv_candidate_configs(source: int | Path) -> list[OpenCVCameraConfig]:
    if sys.platform.startswith("win") and isinstance(source, int):
        return [
            OpenCVCameraConfig(
                index_or_path=source,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=CAMERA_FPS,
                color_mode=ColorMode.RGB,
                backend=Cv2Backends.DSHOW,
            ),
            OpenCVCameraConfig(
                index_or_path=source,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=CAMERA_FPS,
                color_mode=ColorMode.RGB,
                backend=Cv2Backends.DSHOW,
                fourcc="MJPG",
            ),
            OpenCVCameraConfig(
                index_or_path=source,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=CAMERA_FPS,
                color_mode=ColorMode.RGB,
                backend=Cv2Backends.DSHOW,
                fourcc="YUY2",
            ),
        ]

    return [
        OpenCVCameraConfig(
            index_or_path=source,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            color_mode=ColorMode.RGB,
            backend=Cv2Backends.ANY,
        )
    ]


def describe_camera_config(config: OpenCVCameraConfig) -> str:
    backend_label = getattr(config.backend, "name", str(config.backend))
    return f"backend={backend_label}, fourcc={config.fourcc or 'default'}"


def format_camera_label(assignment: CameraAssignment) -> str:
    return f"{assignment.semantic_label} ({assignment.short_key})"


def disconnect_camera_quietly(camera: OpenCVCamera) -> None:
    try:
        if camera.is_connected or getattr(camera, "videocapture", None) is not None:
            camera.disconnect()
    except Exception:
        videocapture = getattr(camera, "videocapture", None)
        if videocapture is not None:
            videocapture.release()
            camera.videocapture = None


def read_valid_camera_frame(
    camera: OpenCVCamera,
    label: str,
    *,
    attempts: int,
    retry_delay_s: float,
    step_name: str,
) -> tuple[np.ndarray, str]:
    last_issue: str | None = None
    for attempt_index in range(1, attempts + 1):
        try:
            frame = np.asarray(camera.read()).copy()
        except Exception as exc:
            last_issue = f"read failed: {exc}"
        else:
            is_blank, stats_message = _is_blank_frame(frame)
            if not is_blank:
                return frame, stats_message
            last_issue = stats_message

        if attempt_index < attempts:
            print(
                f"[Camera] {label}: {step_name} attempt "
                f"{attempt_index}/{attempts} failed ({last_issue}). Retrying."
            )
            time.sleep(retry_delay_s)

    assert last_issue is not None
    raise RuntimeError(f"{label}: {step_name} failed after {attempts} attempts. Last issue: {last_issue}")


class ValidatedLocalCamera:
    def __init__(self, assignment: CameraAssignment, camera: OpenCVCamera) -> None:
        self.assignment = assignment
        self._camera = camera
        self.last_good_frame: np.ndarray | None = None
        self.last_good_timestamp: float | None = None
        self.last_frame_was_reused = False
        self.last_frame_age_ms: float | None = None
        self.validation_stats = "not validated"
        self.selected_config = describe_camera_config(camera.config)

    @property
    def label(self) -> str:
        return format_camera_label(self.assignment)

    def seed_last_good_frame(self, frame: np.ndarray, stats_message: str) -> None:
        self.last_good_frame = np.asarray(frame).copy()
        self.last_good_timestamp = time.perf_counter()
        self.last_frame_was_reused = False
        self.last_frame_age_ms = 0.0
        self.validation_stats = stats_message

    def describe_runtime_status(self) -> str:
        if self.last_frame_age_ms is None:
            return f"{self.assignment.short_key}=unread"
        source = "reused" if self.last_frame_was_reused else "latest"
        return f"{self.assignment.short_key}={source} age={self.last_frame_age_ms:.1f}ms"

    def _mark_latest_frame(self, frame: np.ndarray, age_ms: float | None) -> np.ndarray:
        copied_frame = np.asarray(frame).copy()
        self.last_good_frame = copied_frame.copy()
        self.last_good_timestamp = time.perf_counter()
        self.last_frame_was_reused = False
        self.last_frame_age_ms = max(0.0, age_ms or 0.0)
        return copied_frame

    def _reuse_last_good_frame(self) -> np.ndarray:
        if self.last_good_frame is None:
            raise RuntimeError(f"{self.label}: no cached frame is available for reuse.")
        self.last_frame_was_reused = True
        if self.last_good_timestamp is not None:
            self.last_frame_age_ms = (time.perf_counter() - self.last_good_timestamp) * 1e3
        return self.last_good_frame.copy()

    def read_rgb(self) -> np.ndarray:
        last_error: Exception | None = None
        for _ in range(RUNTIME_CAMERA_READ_ATTEMPTS):
            try:
                frame = self._camera.read_latest(max_age_ms=RUNTIME_CAMERA_MAX_AGE_MS)
                latest_timestamp = getattr(self._camera, "latest_timestamp", None)
                age_ms = None
                if latest_timestamp is not None:
                    age_ms = (time.perf_counter() - latest_timestamp) * 1e3
                frame_array = np.asarray(frame)
                is_blank, stats_message = _is_blank_frame(frame_array)
                if is_blank:
                    raise RuntimeError(stats_message)
                self.validation_stats = stats_message
                return self._mark_latest_frame(frame_array, age_ms)
            except Exception as exc:
                last_error = exc

        if self.last_good_frame is not None:
            return self._reuse_last_good_frame()

        assert last_error is not None
        raise RuntimeError(f"{self.label}: latest-frame read failed and no cached frame is available.") from last_error

    def disconnect(self) -> None:
        try:
            if self._camera.is_connected:
                self._camera.disconnect()
        except DeviceNotConnectedError:
            pass


def build_camera(assignment: CameraAssignment) -> ValidatedLocalCamera:
    attempts = build_opencv_candidate_configs(assignment.source)
    camera_label = format_camera_label(assignment)
    attempted_labels: list[str] = []
    for index, camera_config in enumerate(attempts, start=1):
        camera = OpenCVCamera(camera_config)
        config_label = describe_camera_config(camera_config)
        attempted_labels.append(config_label)
        print(f"[Camera] Opening {camera_label} from {assignment.source} with {config_label}")
        try:
            camera.connect(warmup=True)
            validation_frame, stats_message = read_valid_camera_frame(
                camera,
                camera_label,
                attempts=CAMERA_VALIDATION_ATTEMPTS,
                retry_delay_s=CAMERA_VALIDATION_RETRY_DELAY_S,
                step_name="validation",
            )
            validated_camera = ValidatedLocalCamera(assignment, camera)
            validated_camera.seed_last_good_frame(validation_frame, stats_message)
            print(f"[Camera] {camera_label} ready: {stats_message}")
            return validated_camera
        except Exception as exc:
            disconnect_camera_quietly(camera)
            if index < len(attempts):
                print(f"[Camera] {camera_label}: {exc}. Trying next candidate.")
                continue
            raise RuntimeError(
                f"{camera_label}: no usable OpenCV camera configuration found after trying "
                f"{', '.join(attempted_labels)}. Last error: {exc}"
            ) from exc

    raise RuntimeError(f"{camera_label}: no camera candidates were generated.")


def read_camera_frames(cameras: list[ValidatedLocalCamera]) -> dict[str, np.ndarray]:
    return {camera.assignment.short_key: camera.read_rgb() for camera in cameras}


def collect_runtime_observation(
    robot: XLerobot2Wheels,
    cameras: list[ValidatedLocalCamera],
) -> RuntimeObservation:
    observation_start = time.perf_counter()
    robot_observation = robot.get_observation()
    observation_ms = (time.perf_counter() - observation_start) * 1e3

    camera_start = time.perf_counter()
    frames_rgb = read_camera_frames(cameras)
    camera_ms = (time.perf_counter() - camera_start) * 1e3

    raw_observation = dict(robot_observation)
    raw_observation.update(frames_rgb)
    return RuntimeObservation(raw_observation, frames_rgb, observation_ms, camera_ms)


def get_policy_action_queue_depth(policy: Any) -> int | None:
    queues = getattr(policy, "_queues", None)
    if not isinstance(queues, dict):
        return None
    action_queue = queues.get(ACTION)
    try:
        return len(action_queue)
    except Exception:
        return None


def select_policy_action(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    inference_frame: dict[str, Any],
    dataset_features: dict[str, dict[str, Any]],
) -> ActionSelection:
    queue_depth_before = get_policy_action_queue_depth(policy)

    preprocess_start = time.perf_counter()
    model_input = preprocessor(inference_frame)
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1e3

    model_start = time.perf_counter()
    action_tensor = policy.select_action(model_input)
    model_ms = (time.perf_counter() - model_start) * 1e3

    postprocess_start = time.perf_counter()
    action_tensor = postprocessor(action_tensor)
    postprocess_ms = (time.perf_counter() - postprocess_start) * 1e3

    queue_depth_after = get_policy_action_queue_depth(policy)
    action_vector = action_tensor.squeeze(0).detach().cpu().numpy()
    named_action = make_robot_action(action_tensor, dataset_features)
    return ActionSelection(
        action_vector=action_vector,
        named_action=named_action,
        queue_depth_before=queue_depth_before,
        queue_depth_after=queue_depth_after,
        preprocess_ms=preprocess_ms,
        model_ms=model_ms,
        postprocess_ms=postprocess_ms,
    )


def extract_named_values(source: dict[str, Any], names: tuple[str, ...]) -> dict[str, float]:
    missing = [name for name in names if name not in source]
    if missing:
        raise KeyError(f"Missing expected keys: {missing}")
    return {name: float(source[name]) for name in names}


def compute_named_delta(target: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {name: float(target[name] - baseline[name]) for name in target}


def build_joint_debug_snapshot(
    observation: dict[str, Any],
    named_action: dict[str, float],
) -> JointDebugSnapshot:
    current_state = extract_named_values(observation, RIGHT_ARM_6D_STATE_NAMES)
    predicted_action = extract_named_values(named_action, RIGHT_ARM_6D_STATE_NAMES)
    predicted_delta = compute_named_delta(predicted_action, current_state)
    return JointDebugSnapshot(current_state, predicted_action, predicted_delta)


def compact_joint_line(prefix: str, values: dict[str, float]) -> str:
    tokens = [f"{JOINT_SHORT_LABELS[name]}={values[name]:.1f}" for name in RIGHT_ARM_6D_STATE_NAMES]
    return f"{prefix}: " + " ".join(tokens)


def format_vector(action_vector: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in action_vector.tolist()) + "]"


def format_queue_status(before: int | None, after: int | None) -> str:
    if before is None or after is None:
        return "queue unavailable"
    if before == 0:
        return f"queue refill 0->{after}"
    return f"queue cached {before}->{after}"


def format_active_camera_mapping(assignments: list[CameraAssignment]) -> str:
    return ", ".join(f"{assignment.short_key}={assignment.source}" for assignment in assignments) or "(none)"


def format_camera_runtime_status(cameras: list[ValidatedLocalCamera]) -> str:
    return ", ".join(camera.describe_runtime_status() for camera in cameras) or "(none)"


def build_debug_text(
    *,
    step_index: int,
    args: argparse.Namespace,
    runtime: RuntimeArtifacts,
    observation: RuntimeObservation,
    action_step: ActionSelection,
    joint_snapshot: JointDebugSnapshot,
    camera_runtime_status: str,
) -> str:
    lines = [
        f"step={step_index}",
        "mode=dry-run",
        f"model_id={args.model_id}",
        f"task={args.task}",
        "layout=right_arm_6d",
        f"camera_mapping={format_active_camera_mapping(runtime.camera_assignments)}",
        f"ignored_checkpoint_cameras={','.join(sorted(runtime.ignored_checkpoint_cameras)) or 'none'}",
        f"camera_frames={camera_runtime_status}",
        f"queue_status={format_queue_status(action_step.queue_depth_before, action_step.queue_depth_after)}",
        f"observation_ms={observation.observation_ms:.1f}",
        f"camera_ms={observation.camera_ms:.1f}",
        f"preprocess_ms={action_step.preprocess_ms:.1f}",
        f"model_ms={action_step.model_ms:.1f}",
        f"postprocess_ms={action_step.postprocess_ms:.1f}",
        f"action_vector={format_vector(action_step.action_vector)}",
        compact_joint_line("current", joint_snapshot.current_state),
        compact_joint_line("predicted", joint_snapshot.predicted_action),
        compact_joint_line("delta", joint_snapshot.predicted_delta),
    ]
    return "\n".join(lines) + "\n"


def draw_text_block(canvas: np.ndarray, lines: list[str]) -> None:
    y = PREVIEW_TEXT_MARGIN_Y
    for line in lines:
        if y > canvas.shape[0] - 12:
            break
        cv2.putText(
            canvas,
            line,
            (PREVIEW_TEXT_MARGIN_X, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            PREVIEW_TEXT_FONT_SCALE,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        y += PREVIEW_TEXT_LINE_HEIGHT


def render_preview_grid(frames_rgb: dict[str, np.ndarray]) -> np.ndarray:
    if not frames_rgb:
        return np.zeros((PREVIEW_TILE_HEIGHT, PREVIEW_TILE_WIDTH, 3), dtype=np.uint8)
    labeled_tiles: list[np.ndarray] = []
    for key, frame_rgb in frames_rgb.items():
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        tile = cv2.resize(frame_bgr, (PREVIEW_TILE_WIDTH, PREVIEW_TILE_HEIGHT), interpolation=cv2.INTER_AREA)
        camera_spec = CAMERA_SHORT_KEY_TO_SPEC.get(key)
        label = f"{camera_spec['semantic_label']} ({key})" if camera_spec is not None else key
        cv2.putText(tile, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 3, cv2.LINE_AA)
        cv2.putText(tile, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        labeled_tiles.append(tile)
    if len(labeled_tiles) == 1:
        return labeled_tiles[0]
    return np.concatenate(labeled_tiles[:2], axis=1)


def build_preview_frame(frames_rgb: dict[str, np.ndarray], status_lines: list[str]) -> np.ndarray:
    grid = render_preview_grid(frames_rgb)
    panel = np.zeros((grid.shape[0], PREVIEW_PANEL_WIDTH, 3), dtype=np.uint8)
    draw_text_block(panel, status_lines)
    return np.concatenate([grid, panel], axis=1)


def write_runtime_debug(
    run_dir: Path,
    *,
    step_index: int,
    preview_frame_bgr: np.ndarray,
    debug_text: str,
    action_payload: dict[str, Any],
) -> None:
    cv2.imwrite(str(run_dir / "latest_preview.jpg"), preview_frame_bgr)
    (run_dir / "latest_runtime.txt").write_text(debug_text, encoding="utf-8")
    write_json(run_dir / "latest_action.json", action_payload)
    append_jsonl(run_dir / "actions.jsonl", {"step": step_index, **action_payload})


def write_session_config(
    run_dir: Path,
    args: argparse.Namespace,
    config: PreTrainedConfig,
    camera_assignments: list[CameraAssignment],
    ignored_checkpoint_cameras: set[str],
) -> None:
    write_json(
        run_dir / "session.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "dry-run",
            "model_id": args.model_id,
            "task": args.task,
            "device": str(config.device),
            "robot_id": args.robot_id,
            "port1": args.port1,
            "port2": args.port2,
            "state_names": list(RIGHT_ARM_6D_STATE_NAMES),
            "action_names": list(RIGHT_ARM_6D_STATE_NAMES),
            "camera_mapping": [
                {
                    "full_key": assignment.full_key,
                    "short_key": assignment.short_key,
                    "semantic_label": assignment.semantic_label,
                    "source": assignment.source,
                }
                for assignment in camera_assignments
            ],
            "ignored_checkpoint_cameras": sorted(ignored_checkpoint_cameras),
            "max_steps": args.max_steps,
            "debug_write_every": args.debug_write_every,
        },
    )


def print_startup_summary(
    args: argparse.Namespace,
    config: PreTrainedConfig,
    run_dir: Path,
    camera_assignments: list[CameraAssignment],
    ignored_checkpoint_cameras: set[str],
) -> None:
    print(f"[Model] model_id: {args.model_id}")
    print(f"[Model] type: {config.type}")
    print(f"[Model] device: {config.device}")
    print("[Model] layout: right_arm_6d state_dim=6 action_dim=6")
    print(f"[Task] {args.task}")
    print(f"[Robot] id={args.robot_id} port1={args.port1} port2={args.port2}")
    print(f"[Camera] active mapping: {format_active_camera_mapping(camera_assignments)}")
    if ignored_checkpoint_cameras:
        print(f"[Camera] ignored checkpoint cameras: {', '.join(sorted(ignored_checkpoint_cameras))}")
    print(f"[Debug] run_dir: {run_dir}")
    print("[Mode] Dry-run only. This script never executes predicted policy actions.")
    if args.no_window:
        print("[Mode] Preview window disabled (--no-window).")
    else:
        print("[Mode] Press q or ESC in the preview window to stop.")
    if str(config.device) == "cpu":
        print("[Hint] CPU inference can stall between visible updates.")


def build_runtime(args: argparse.Namespace) -> RuntimeArtifacts:
    run_dir = make_run_dir(args.debug_root)
    config = build_policy_config(args)
    raw_camera_assignments = args.camera if args.camera is not None else []
    camera_assignments, ignored_checkpoint_cameras = parse_camera_assignments(
        raw_camera_assignments,
        config.image_features,
    )
    write_session_config(run_dir, args, config, camera_assignments, ignored_checkpoint_cameras)
    print_startup_summary(args, config, run_dir, camera_assignments, ignored_checkpoint_cameras)

    policy, preprocessor, postprocessor = load_policy_artifacts(args, config)
    robot = build_robot_instance(args)
    cameras = [build_camera(assignment) for assignment in camera_assignments]
    initial_frames_rgb = read_camera_frames(cameras)
    dataset_features = build_dataset_features(cameras, initial_frames_rgb)

    return RuntimeArtifacts(
        args=args,
        run_dir=run_dir,
        device=torch.device(policy.config.device),
        camera_assignments=camera_assignments,
        dataset_features=dataset_features,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot=robot,
        cameras=cameras,
        ignored_checkpoint_cameras=ignored_checkpoint_cameras,
    )


def should_run_step(step_index: int, max_steps: int) -> bool:
    return max_steps == 0 or step_index <= max_steps


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()
    runtime: RuntimeArtifacts | None = None
    robot: XLerobot2Wheels | None = None
    cameras: list[ValidatedLocalCamera] = []

    try:
        runtime = build_runtime(args)
        robot = runtime.robot
        cameras = runtime.cameras

        print("[Robot] Connecting for observation-only dry-run...")
        robot.connect()

        action_names = runtime.dataset_features[ACTION]["names"]
        step_index = 0
        while True:
            step_index += 1
            if not should_run_step(step_index, args.max_steps):
                print(f"[Exit] Reached --max-steps={args.max_steps}.")
                break

            observation = collect_runtime_observation(robot, cameras)
            inference_frame = build_inference_frame(
                observation=observation.raw_observation,
                ds_features=runtime.dataset_features,
                device=runtime.device,
                task=args.task,
            )
            action_step = select_policy_action(
                runtime.policy,
                runtime.preprocessor,
                runtime.postprocessor,
                inference_frame,
                runtime.dataset_features,
            )
            if len(action_step.action_vector) != len(action_names):
                raise RuntimeError(
                    f"Action preview mismatch: tensor has {len(action_step.action_vector)} dims "
                    f"but action names has {len(action_names)}."
                )

            joint_snapshot = build_joint_debug_snapshot(observation.raw_observation, action_step.named_action)
            camera_runtime_status = format_camera_runtime_status(cameras)
            status_lines = [
                "Dry-run only",
                f"task: {args.task}",
                f"cameras: {format_active_camera_mapping(runtime.camera_assignments)}",
                f"camera_frames: {camera_runtime_status}",
                (
                    f"{format_queue_status(action_step.queue_depth_before, action_step.queue_depth_after)} "
                    f"obs={observation.observation_ms:.1f}ms cam={observation.camera_ms:.1f}ms"
                ),
                (
                    f"prep={action_step.preprocess_ms:.1f}ms "
                    f"model={action_step.model_ms:.1f}ms "
                    f"post={action_step.postprocess_ms:.1f}ms"
                ),
                f"action[6D]={format_vector(action_step.action_vector)}",
                compact_joint_line("cur", joint_snapshot.current_state),
                compact_joint_line("prd", joint_snapshot.predicted_action),
                compact_joint_line("dlt", joint_snapshot.predicted_delta),
                "q or ESC to quit",
            ]
            preview_frame = build_preview_frame(observation.frames_rgb, status_lines)
            debug_text = build_debug_text(
                step_index=step_index,
                args=args,
                runtime=runtime,
                observation=observation,
                action_step=action_step,
                joint_snapshot=joint_snapshot,
                camera_runtime_status=camera_runtime_status,
            )
            action_payload = {
                "saved_at": datetime.now().isoformat(timespec="milliseconds"),
                "action_vector": action_step.action_vector,
                "current_state": joint_snapshot.current_state,
                "predicted_action": joint_snapshot.predicted_action,
                "predicted_delta": joint_snapshot.predicted_delta,
                "timing_ms": {
                    "observation": observation.observation_ms,
                    "camera": observation.camera_ms,
                    "preprocess": action_step.preprocess_ms,
                    "model": action_step.model_ms,
                    "postprocess": action_step.postprocess_ms,
                },
                "queue": {
                    "before": action_step.queue_depth_before,
                    "after": action_step.queue_depth_after,
                },
            }

            if step_index == 1 or step_index % args.terminal_debug_every == 0:
                print(debug_text.rstrip())

            if step_index == 1 or step_index % args.debug_write_every == 0:
                write_runtime_debug(
                    runtime.run_dir,
                    step_index=step_index,
                    preview_frame_bgr=preview_frame,
                    debug_text=debug_text,
                    action_payload=action_payload,
                )

            if not args.no_window:
                cv2.imshow(WINDOW_NAME, preview_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), 27}:
                    print("[Exit] User requested stop from preview window.")
                    break

    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        for camera in reversed(cameras):
            camera.disconnect()
        if robot is not None:
            try:
                if robot.is_connected:
                    robot.disconnect()
            except DeviceNotConnectedError:
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[Exit] Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        raise SystemExit(1)
