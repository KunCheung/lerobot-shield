from __future__ import annotations

import argparse
import math
import sys
import textwrap
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
EXPECTED_LEROBOT_INIT_PATH = (SRC_PATH / "lerobot" / "__init__.py").resolve()
DOTENV_PATH = Path(__file__).with_name(".env")

DEFAULT_MODEL_ID = "Grigorij/smolvla_collect_tissues"
DEFAULT_TASK = "Collect tissues."
DEFAULT_ROBOT_ID = "my_xlerobot_2wheels_lab"
DEFAULT_PORT1 = "COM5"  # left arm + head
DEFAULT_PORT2 = "COM4"  # right arm + 2-wheel base
DEFAULT_CAMERA_FPS = 30
WINDOW_NAME = "SmolVLA Dry Run"
PANEL_WIDTH = 540
TEXT_LINE_HEIGHT = 22
TEXT_FONT_SCALE = 0.52
TEXT_MARGIN_X = 12
TEXT_MARGIN_Y = 24
LATENCY_WINDOW = 20
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
    {
        "short_key": "camera3",
        "full_key": "observation.images.camera3",
        "semantic_label": "Left Arm",
        "default_source": 3,
    },
)
CAMERA_SHORT_KEY_TO_SPEC = {spec["short_key"]: spec for spec in CAMERA_SLOT_SPECS}
CAMERA_FULL_KEY_TO_SPEC = {spec["full_key"]: spec for spec in CAMERA_SLOT_SPECS}
SUPPORTED_CAMERA_FULL_KEYS = {spec["full_key"] for spec in CAMERA_SLOT_SPECS}
DEFAULT_CAMERA_MAPPING_TEXT = ", ".join(
    f"{spec['short_key']}={spec['default_source']}" for spec in CAMERA_SLOT_SPECS
)


if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import lerobot as local_lerobot

LOCAL_LEROBOT_INIT_PATH = Path(local_lerobot.__file__).resolve()
if LOCAL_LEROBOT_INIT_PATH != EXPECTED_LEROBOT_INIT_PATH:
    raise RuntimeError(
        "This script must run against the local repository source, but imported "
        f"lerobot from {LOCAL_LEROBOT_INIT_PATH}. Expected {EXPECTED_LEROBOT_INIT_PATH}. "
        "Please run: conda activate lerobot && cd C:\\projects\\lerobot && "
        "python -m pip install -e . && python examples/shield/3_try_smolvla_dry_run.py --help"
    )

if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH, override=False)

from lerobot.cameras.configs import ColorMode, Cv2Backends
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.feature_utils import combine_feature_dicts, hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels, XLerobot2WheelsConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STR
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


@dataclass(frozen=True)
class CameraAssignment:
    full_key: str
    short_key: str
    semantic_label: str
    source: int | Path
    default_source: int
    requested_width: int
    requested_height: int


class ValidatedLocalCamera:
    def __init__(self, assignment: CameraAssignment, camera: OpenCVCamera, validation_stats: str) -> None:
        self.assignment = assignment
        self._camera = camera
        self.validation_stats = validation_stats
        self.backend = getattr(camera.config.backend, "name", str(camera.config.backend))
        self.fourcc = camera.config.fourcc or "default"

    def read_rgb(self) -> np.ndarray:
        frame = self._camera.read()
        is_blank, stats_message = _is_blank_frame(frame)
        if is_blank:
            raise RuntimeError(f"{self.assignment.short_key}: {stats_message}")
        return np.asarray(frame).copy()

    def disconnect(self) -> None:
        try:
            if self._camera.is_connected:
                self._camera.disconnect()
        except DeviceNotConnectedError:
            pass

    @property
    def is_connected(self) -> bool:
        return self._camera.is_connected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run SmolVLA inference with xlerobot_2wheels and user-managed OpenCV cameras.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="SmolVLA model repo id or local path.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Natural-language task passed to the model.")
    parser.add_argument("--device", default=None, help="Torch device override, such as cuda, cpu, or mps.")
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID, help="Robot id used for calibration lookup.")
    parser.add_argument("--port1", default=DEFAULT_PORT1, help="Serial port for left arm + head.")
    parser.add_argument("--port2", default=DEFAULT_PORT2, help="Serial port for right arm + base.")
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="MODEL_CAMERA_KEY=INDEX",
        help=(
            "Override one camera slot mapping. Accepts camera1/2/3 or the full observation.images.camera* key. "
            f"If omitted, defaults are used: {DEFAULT_CAMERA_MAPPING_TEXT}."
        ),
    )
    return parser.parse_args()


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


def validate_checkpoint_image_features(expected_image_features: dict[str, Any]) -> None:
    expected_keys = set(expected_image_features)
    missing_keys = SUPPORTED_CAMERA_FULL_KEYS - expected_keys
    extra_keys = expected_keys - SUPPORTED_CAMERA_FULL_KEYS

    if missing_keys or extra_keys:
        raise ValueError(
            "Unsupported checkpoint image features for this dry-run script.\n"
            f"- Required keys: {', '.join(spec['full_key'] for spec in CAMERA_SLOT_SPECS)}\n"
            f"- Missing keys: {', '.join(sorted(missing_keys)) or 'None'}\n"
            f"- Extra keys: {', '.join(sorted(extra_keys)) or 'None'}"
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
) -> list[CameraAssignment]:
    validate_checkpoint_image_features(expected_image_features)
    expected_keys = set(expected_image_features)

    source_by_key: dict[str, int | Path] = {
        spec["full_key"]: spec["default_source"] for spec in CAMERA_SLOT_SPECS if spec["full_key"] in expected_keys
    }
    seen_keys: set[str] = set()

    for raw_assignment in raw_assignments:
        if "=" not in raw_assignment:
            raise ValueError(
                f"Invalid --camera value '{raw_assignment}'. Expected the form MODEL_CAMERA_KEY=INDEX."
            )
        raw_key, raw_value = raw_assignment.split("=", 1)
        full_key = normalize_camera_key(raw_key, expected_keys)
        if full_key in seen_keys:
            raise ValueError(f"Duplicate --camera mapping for '{full_key}'.")
        source_by_key[full_key] = parse_camera_source(raw_value)
        seen_keys.add(full_key)

    missing_keys = expected_keys - set(source_by_key)
    if missing_keys:
        raise ValueError(
            "Missing required camera mappings after applying defaults.\n"
            f"Expected model camera keys: {', '.join(sorted(expected_keys))}\n"
            f"Missing: {', '.join(sorted(missing_keys))}"
        )

    extra_keys = set(source_by_key) - expected_keys
    if extra_keys:
        raise ValueError(f"Unexpected camera keys: {', '.join(sorted(extra_keys))}")

    normalized_assignments: list[CameraAssignment] = []
    for spec in CAMERA_SLOT_SPECS:
        full_key = spec["full_key"]
        if full_key not in expected_keys:
            continue

        shape = expected_image_features[full_key].shape
        if len(shape) != 3:
            raise ValueError(f"Unsupported image feature shape for '{full_key}': {shape}")

        normalized_assignments.append(
            CameraAssignment(
                full_key=full_key,
                short_key=spec["short_key"],
                semantic_label=spec["semantic_label"],
                source=source_by_key[full_key],
                default_source=int(spec["default_source"]),
                requested_width=int(shape[-1]),
                requested_height=int(shape[-2]),
            )
        )

    return normalized_assignments


def build_policy_config(model_id: str, device_override: str | None) -> PreTrainedConfig:
    cli_overrides: list[str] = []
    if device_override:
        cli_overrides.append(f"--device={device_override}")

    try:
        config = PreTrainedConfig.from_pretrained(model_id, cli_overrides=cli_overrides)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load checkpoint config. Make sure the model id is correct and that "
            "the checkpoint is reachable on Hugging Face or available locally.\n"
            f"model_id={model_id}"
        ) from exc

    if config.type != "smolvla":
        raise ValueError(f"Expected a SmolVLA checkpoint, but '{model_id}' is of type '{config.type}'.")
    return config


def build_robot_instance(args: argparse.Namespace) -> XLerobot2Wheels:
    robot_config = XLerobot2WheelsConfig(
        id=args.robot_id,
        port1=args.port1,
        port2=args.port2,
        cameras={},
    )
    return XLerobot2Wheels(robot_config)


def validate_checkpoint_compatibility(config: PreTrainedConfig, robot: XLerobot2Wheels) -> tuple[int, int]:
    robot_state_feature = config.robot_state_feature
    if robot_state_feature is None:
        raise ValueError("Checkpoint does not declare observation.state in input_features.")

    action_feature = config.action_feature
    if action_feature is None:
        raise ValueError("Checkpoint does not declare action in output_features.")

    robot_state_dim = len(robot.observation_features)
    robot_action_dim = len(robot.action_features)
    model_state_dim = int(robot_state_feature.shape[0])
    model_action_dim = int(action_feature.shape[0])

    if model_state_dim != robot_state_dim:
        raise ValueError(
            "State dimension mismatch.\n"
            f"- Checkpoint observation.state dim: {model_state_dim}\n"
            f"- xlerobot_2wheels state dim: {robot_state_dim}\n"
            "This dry-run script does not guess a 12D/16D mapping."
        )

    if model_action_dim != robot_action_dim:
        raise ValueError(
            "Action dimension mismatch.\n"
            f"- Checkpoint action dim: {model_action_dim}\n"
            f"- xlerobot_2wheels action dim: {robot_action_dim}\n"
            "This dry-run script does not guess a 12D/16D mapping."
        )

    return model_state_dim, model_action_dim


def select_camera_backend(source: int | Path) -> Cv2Backends:
    return Cv2Backends.DSHOW if isinstance(source, int) else Cv2Backends.ANY


def build_camera(assignment: CameraAssignment) -> ValidatedLocalCamera:
    camera_config = OpenCVCameraConfig(
        index_or_path=assignment.source,
        width=assignment.requested_width,
        height=assignment.requested_height,
        fps=DEFAULT_CAMERA_FPS,
        color_mode=ColorMode.RGB,
        backend=select_camera_backend(assignment.source),
    )
    camera = OpenCVCamera(camera_config)
    backend_label = getattr(camera_config.backend, "name", str(camera_config.backend))
    fourcc_label = camera_config.fourcc or "default"
    print(
        f"[Camera] Opening {assignment.semantic_label} ({assignment.short_key}) from {assignment.source} "
        f"backend={backend_label}, fourcc={fourcc_label}, "
        f"requested={assignment.requested_width}x{assignment.requested_height}@{DEFAULT_CAMERA_FPS}, "
        f"default_source={assignment.default_source}"
    )
    camera.connect(warmup=True)
    validation_frame = camera.read()
    is_blank, stats_message = _is_blank_frame(validation_frame)
    if is_blank:
        camera.disconnect()
        raise RuntimeError(f"{assignment.short_key}: validation failed: {stats_message}")

    print(f"[Camera] {assignment.semantic_label} ({assignment.short_key}): {stats_message}")
    return ValidatedLocalCamera(assignment=assignment, camera=camera, validation_stats=stats_message)


def build_camera_feature_dict(cameras: list[ValidatedLocalCamera], frames_rgb: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    feature_dict: dict[str, dict[str, Any]] = {}
    for camera in cameras:
        frame = frames_rgb[camera.assignment.short_key]
        height, width = frame.shape[:2]
        feature_dict[camera.assignment.full_key] = {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return feature_dict


def build_dataset_features(
    robot: XLerobot2Wheels,
    cameras: list[ValidatedLocalCamera],
    frames_rgb: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    action_features = hw_to_dataset_features(robot.action_features, ACTION, use_video=False)
    observation_features = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False)
    camera_features = build_camera_feature_dict(cameras, frames_rgb)
    return combine_feature_dicts(action_features, observation_features, camera_features)


def read_camera_frames(cameras: list[ValidatedLocalCamera]) -> dict[str, np.ndarray]:
    frames_rgb: dict[str, np.ndarray] = {}
    for camera in cameras:
        frames_rgb[camera.assignment.short_key] = camera.read_rgb()
    return frames_rgb


def build_runtime_observation(robot_observation: dict[str, Any], frames_rgb: dict[str, np.ndarray]) -> dict[str, Any]:
    runtime_observation = dict(robot_observation)
    runtime_observation.update(frames_rgb)
    return runtime_observation


def format_vector(action_vector: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in action_vector.tolist()) + "]"


def format_named_action_lines(named_action: dict[str, float]) -> list[str]:
    entries = [f"{name}={value:.3f}" for name, value in named_action.items()]
    if not entries:
        return ["named_action: (none)"]

    lines: list[str] = []
    for index in range(0, len(entries), 2):
        lines.append("  ".join(entries[index : index + 2]))
    return lines


def draw_text_block(canvas: np.ndarray, lines: list[str]) -> None:
    y = TEXT_MARGIN_Y
    max_text_width = max(10, canvas.shape[1] - (TEXT_MARGIN_X * 2))
    max_chars = max(20, max_text_width // 8)

    for line in lines:
        wrapped_lines = textwrap.wrap(line, width=max_chars) or [""]
        for wrapped_line in wrapped_lines:
            if y > canvas.shape[0] - 12:
                return
            cv2.putText(
                canvas,
                wrapped_line,
                (TEXT_MARGIN_X, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_FONT_SCALE,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            y += TEXT_LINE_HEIGHT


def render_camera_grid(frames_bgr: list[tuple[str, np.ndarray]]) -> np.ndarray:
    if not frames_bgr:
        return np.zeros((360, 640, 3), dtype=np.uint8)

    tile_width = 420
    tile_height = 315
    num_frames = len(frames_bgr)
    columns = max(1, math.ceil(math.sqrt(num_frames)))
    rows = math.ceil(num_frames / columns)
    grid = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)

    for index, (label, frame_bgr) in enumerate(frames_bgr):
        row = index // columns
        column = index % columns
        resized = cv2.resize(frame_bgr, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        cv2.putText(
            resized,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (40, 40, 40),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            resized,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y0 = row * tile_height
        x0 = column * tile_width
        grid[y0 : y0 + tile_height, x0 : x0 + tile_width] = resized

    return grid


def build_preview_frame(frames_rgb: dict[str, np.ndarray], status_lines: list[str]) -> np.ndarray:
    labeled_frames = []
    for key, frame_rgb in frames_rgb.items():
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        camera_spec = CAMERA_SHORT_KEY_TO_SPEC.get(key)
        label = f"{camera_spec['semantic_label']} ({key})" if camera_spec is not None else key
        labeled_frames.append((label, frame_bgr))

    grid = render_camera_grid(labeled_frames)
    panel = np.zeros((grid.shape[0], PANEL_WIDTH, 3), dtype=np.uint8)
    draw_text_block(panel, status_lines)
    return np.concatenate([grid, panel], axis=1)


def print_runtime_summary(
    args: argparse.Namespace,
    config: PreTrainedConfig,
    state_dim: int,
    action_dim: int,
    camera_assignments: list[CameraAssignment],
) -> None:
    print(f"[Env] Python executable: {sys.executable}")
    print(f"[Env] lerobot module: {LOCAL_LEROBOT_INIT_PATH}")
    print(f"[Model] model_id: {args.model_id}")
    print(f"[Model] device: {config.device}")
    print(f"[Model] state_dim: {state_dim}")
    print(f"[Model] action_dim: {action_dim}")
    if camera_assignments:
        print(f"[Camera] default mapping: {DEFAULT_CAMERA_MAPPING_TEXT}")
        for assignment in camera_assignments:
            source_suffix = ""
            if assignment.source != assignment.default_source:
                source_suffix = f" (override, default={assignment.default_source})"
            print(
                f"[Camera] {assignment.semantic_label} ({assignment.full_key}) <- {assignment.source}{source_suffix} "
                f"(requested {assignment.requested_width}x{assignment.requested_height})"
            )
    else:
        print("[Model] checkpoint does not declare image inputs.")
    print("[Mode] Dry-run only. No robot actions will be sent.")


def main() -> None:
    args = parse_args()
    policy = None
    robot: XLerobot2Wheels | None = None
    cameras: list[ValidatedLocalCamera] = []

    try:
        config = build_policy_config(args.model_id, args.device)
        camera_assignments = parse_camera_assignments(args.camera, config.image_features)

        robot = build_robot_instance(args)
        state_dim, action_dim = validate_checkpoint_compatibility(config, robot)

        print_runtime_summary(args, config, state_dim, action_dim, camera_assignments)

        for assignment in camera_assignments:
            cameras.append(build_camera(assignment))

        initial_frames_rgb = read_camera_frames(cameras)
        dataset_features = build_dataset_features(robot, cameras, initial_frames_rgb)

        robot.connect()

        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(args.model_id, config=config)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=args.model_id,
            preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
        )
        policy.reset()

        action_names = dataset_features[ACTION]["names"]
        latency_history: deque[float] = deque(maxlen=LATENCY_WINDOW)
        fps_history: deque[float] = deque(maxlen=LATENCY_WINDOW)
        previous_loop_end: float | None = None

        while True:
            loop_start = time.perf_counter()
            robot_observation = robot.get_observation()
            frames_rgb = read_camera_frames(cameras)
            runtime_observation = build_runtime_observation(robot_observation, frames_rgb)

            inference_frame = build_inference_frame(
                observation=runtime_observation,
                ds_features=dataset_features,
                device=torch.device(policy.config.device),
                task=args.task,
            )

            inference_start = time.perf_counter()
            model_input = preprocessor(inference_frame)
            action_tensor = policy.select_action(model_input)
            action_tensor = postprocessor(action_tensor)
            inference_latency_ms = (time.perf_counter() - inference_start) * 1e3
            latency_history.append(inference_latency_ms)

            action_vector = action_tensor.squeeze(0).detach().cpu().numpy()
            named_action = make_robot_action(action_tensor, dataset_features)

            loop_end = time.perf_counter()
            if previous_loop_end is not None:
                dt = loop_end - previous_loop_end
                if dt > 0:
                    fps_history.append(1.0 / dt)
            previous_loop_end = loop_end

            mean_latency = sum(latency_history) / len(latency_history)
            mean_fps = (sum(fps_history) / len(fps_history)) if fps_history else 0.0
            camera_summary = ", ".join(
                (
                    f"{camera.assignment.semantic_label}({camera.assignment.short_key})={camera.assignment.source}"
                    if camera.assignment.source == camera.assignment.default_source
                    else (
                        f"{camera.assignment.semantic_label}({camera.assignment.short_key})="
                        f"{camera.assignment.source} [default {camera.assignment.default_source}]"
                    )
                )
                for camera in cameras
            )
            status_lines = [
                "Dry-run only. This script never calls robot.send_action(...).",
                f"model: {args.model_id}",
                f"device: {policy.config.device}",
                f"task: {args.task}",
                f"robot: id={args.robot_id} port1={args.port1} port2={args.port2}",
                f"state_dim={state_dim} action_dim={action_dim}",
                f"default_camera_mapping: {DEFAULT_CAMERA_MAPPING_TEXT}",
                f"cameras: {camera_summary}",
                f"latency_ms: current={inference_latency_ms:.1f} avg={mean_latency:.1f}",
                f"fps: avg={mean_fps:.2f}",
                f"raw_action ({len(action_vector)}D): {format_vector(action_vector)}",
                "named_action:",
                *format_named_action_lines(named_action),
                f"q or ESC to quit | loop_ms={(loop_end - loop_start) * 1e3:.1f}",
            ]

            preview_frame = build_preview_frame(frames_rgb, status_lines)
            cv2.imshow(WINDOW_NAME, preview_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                print("[Exit] User requested stop from keyboard.")
                break

            if len(action_vector) != len(action_names):
                raise RuntimeError(
                    f"Action preview mismatch: tensor has {len(action_vector)} dims but action names has {len(action_names)}."
                )

    finally:
        cv2.destroyAllWindows()
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
