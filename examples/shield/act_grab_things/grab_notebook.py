from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
SHIELD_DIR = SCRIPT_DIR.parent
DOTENV_PATH = SHIELD_DIR / ".env"
MEMORY_DB_PATH = Path(os.getenv("TEMP", str(Path.cwd()))) / "robocrew_robot_memory.db"
TMP_IMAGES_DIR = SHIELD_DIR / "tmp_images"
LATEST_RAW_CAMERA_IMAGE_PATH = TMP_IMAGES_DIR / "latest_raw_camera.png"
LATEST_RAW_CAMERA_JSON_PATH = TMP_IMAGES_DIR / "latest_raw_camera.json"
LATEST_LLM_INPUT_IMAGE_PATH = TMP_IMAGES_DIR / "latest_llm_input.jpg"
LATEST_LLM_INPUT_JSON_PATH = TMP_IMAGES_DIR / "latest_llm_input.json"
VLA_CAMERA_DEBUG_ENABLED = True
VLA_CAMERA_DEBUG_DIR = TMP_IMAGES_DIR / "vla_grab_debug"
VLA_CAMERA_DEBUG_INTERVAL_S = 0.5
VLA_ACTION_DEBUG_ENABLED = True
KEEP_GRASP_AFTER_VLA = True
RESET_ARMS_AFTER_GRAB = False
VLA_EXECUTION_TIME_S = 60
VLA_START_POSE_NAME = "cobra"
TASK_COMPLETION_STATE: dict[str, bool] = {"completed": False}

MODEL_NAME = "openai:kimi-k2.6"
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
SYSTEM_PROMPT = """
## ROBOT SPECS
- Mobile household robot with a two-wheel differential base, two arms, and one head
- ARM REACH: ~30cm only (VERY SHORT)
- Navigation modes: NORMAL (long-distance, forward camera) and PRECISION (close-range, downward camera)
- The base can only move forward, move backward, turn left, and turn right. It cannot move sideways.
- The notebook manipulation policy uses the RIGHT ARM only. Keep the left arm out of the task.

## MANIPULATION RULES - CRITICAL
- ALWAYS switch to PRECISION mode BEFORE any manipulation attempt
- GREEN LINES show your arm reach boundary (only visible in PRECISION mode)
- ONLY manipulate when the BASE of target object is BELOW the green line
- If target is above green line: TOO FAR - move closer first using small forward steps (0.1m)
- If target is off-center: use small turns to align first
- After Grab_a_notebook succeeds, stop the task and do not call more movement tools

## NAVIGATION RULES
- Can't see target? Use look_around FIRST (don't wander blindly)
- Check angle grid at top of image - target must be within +/-20 degrees of center before moving forward
- Watch for obstacles in your path - if obstacle blocks the way, navigate around it first
- STUCK (standing on same place after moving)? Switch to PRECISION, use move_backward, then turn to re-align
- Never call move_forward 3+ times if nothing changes
- Never request or assume sideways motion; this robot has only two wheels

## NORMAL MODE (Long-distance)
- Use for: navigation 0.5-1m, exploring
- If target is off-center: use turn_left or turn_right to align BEFORE moving forward
- Before EVERY move_forward: verify target is centered (+/-20 degrees on angle grid)
- Reference floor meters only if floor visible and scale not on objects
- Watch for obstacles between you and target - plan path to avoid them
- Switch to PRECISION ONLY when target is at the VERY BOTTOM of camera view (almost touching bottom edge)

## PRECISION MODE (Close-range)
- Enter when: target or obstacle is at very bottom of view (intersects with view bottom edge), stuck, or about to manipulate
- You will see: your arms, black basket (your body), and green reach lines
- Small movements only: 0.05-0.3m
- Green lines show arm reach - check if BASE of target is below green line before manipulating
- If target above green line: move forward 0.1m increments until base crosses below line
- Remember that your body is a wide rectangle. Use turn tools to align the body edges with the target edges.
- Exit when: far from obstacles/target, or lost target - switch to NORMAL and look_around

## OPERATION SEQUENCE
1. Don't know where target is? Use look_around
2. Target visible but far? NORMAL mode, turn to center it, then move_forward
3. Target at bottom of view? Switch to PRECISION mode
4. In PRECISION, target off-center? Turn in small increments to center it
5. In PRECISION, target above green line? Move forward until below line
6. Target centered AND below green line? Use Grab_a_notebook
7. Stuck or lost target? Use PRECISION mode + move_backward/turn OR switch to NORMAL + look_around
""".strip()
TASK = "Approach the notebook, grab it from the table, and stop after grabbing it."
PICKUP_TOOL_NAME = "Grab_a_notebook"
PICKUP_TASK_PROMPT = "Grab a notebook."
PICKUP_SUCCESS_MESSAGE = "Right arm completed the notebook grab and is holding the grasp."

MAIN_CAMERA_ID = 1
RIGHT_ARM_CAMERA_ID = 2
RIGHT_ARM_WHEEL_USB = "COM4"
LEFT_ARM_HEAD_USB = "COM5"
# Differential base wheel IDs on this 2-wheel robot.
BASE_LEFT_WHEEL_ID = 9
BASE_RIGHT_WHEEL_ID = 10
TWO_WHEEL_ACTION_MAP = {
    # Left base wheel (id=9) is inverted on this robot calibration.
    "forward": {BASE_LEFT_WHEEL_ID: -1.0, BASE_RIGHT_WHEEL_ID: 1.0},
    "backward": {BASE_LEFT_WHEEL_ID: 1.0, BASE_RIGHT_WHEEL_ID: -1.0},
    "turn_left": {BASE_LEFT_WHEEL_ID: 1.0, BASE_RIGHT_WHEEL_ID: 1.0},
    "turn_right": {BASE_LEFT_WHEEL_ID: -1.0, BASE_RIGHT_WHEEL_ID: -1.0},
}
POLICY_SERVER_ADDRESS = "127.0.0.1:8080"
POLICY_NAME = "Grigorij/act_right-arm-grab-notebook-2"
POLICY_TYPE = "act"
POLICY_DEVICE = "cpu"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_WARMUP_FRAMES = 6
CAMERA_READ_ATTEMPTS = 3
VLA_CAMERA_DEBUG_IO_LOCK = threading.Lock()
LAST_VLA_CAMERA_DEBUG_SESSION: dict[str, object] | None = None
VLA_ACTION_DEBUG_JOINT_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
VLA_ACTION_EXTENSION_JOINT_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
)

if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH, override=False)

MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
if MOONSHOT_API_KEY:
    os.environ["MOONSHOT_API_KEY"] = MOONSHOT_API_KEY
os.environ.setdefault("OPENAI_BASE_URL", MOONSHOT_BASE_URL)

import robocrew.core.memory as robocrew_memory


def _patched_memory_init(self, db_filename: str = "robot_memory.db") -> None:
    self.db_path = str(MEMORY_DB_PATH)
    self.init_db()


robocrew_memory.Memory.__init__ = _patched_memory_init

import robocrew.core.LLMAgent as robocrew_llm_module
import robocrew.robots.XLeRobot.servo_controls as servo_controls_module
from langchain.chat_models import init_chat_model as lc_init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
import lerobot.cameras.opencv as lerobot_opencv_package
import lerobot.cameras.opencv.camera_opencv as opencv_camera_module
from lerobot.cameras.opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import ColorMode, Cv2Backends, OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from robocrew.core.camera import RobotCamera
from robocrew.core.utils import basic_augmentation
from robocrew.robots.XLeRobot.tools import (
    create_go_to_precision_mode,
    create_look_around,
    create_move_backward,
    create_move_forward,
    create_turn_left,
    create_turn_right,
)


def _patched_init_chat_model(model: str, *args, **kwargs):
    if model in {"kimi-k2.5", "openai:kimi-k2.5", "kimi-k2.6", "openai:kimi-k2.6"}:
        kwargs.setdefault("base_url", MOONSHOT_BASE_URL)
        if MOONSHOT_API_KEY:
            kwargs.setdefault("api_key", MOONSHOT_API_KEY)
        if ":" not in model:
            model = f"openai:{model}"
    return lc_init_chat_model(model, *args, **kwargs)


robocrew_llm_module.init_chat_model = _patched_init_chat_model
LLMAgent = robocrew_llm_module.LLMAgent


class NotebookGrabAgent(LLMAgent):
    """LLM agent variant that stops issuing tools after the notebook grab succeeds."""

    def __init__(
        self,
        *args,
        completion_state: dict[str, bool],
        completion_tool_name: str,
        **kwargs,
    ) -> None:
        self._completion_state = completion_state
        self._completion_tool_name = completion_tool_name
        super().__init__(*args, **kwargs)

    def invoke_tool(self, tool_call):
        if self._completion_state.get("completed"):
            tool_call_id = tool_call.get("id", "notebook_grab_completed")
            return (
                ToolMessage(
                    "Notebook grab already completed; ignoring further tool calls for this task.",
                    tool_call_id=tool_call_id,
                ),
                None,
            )

        tool_response, additional_response = super().invoke_tool(tool_call)
        if tool_call["name"] == self._completion_tool_name and self._completion_state.get("completed"):
            self.task = None
            print("[Task] Notebook grab completed; stopping the active task.")
        return tool_response, additional_response


XLEROBOT_2WHEELS_CALIBRATION_PATH = (
    HF_LEROBOT_CALIBRATION / ROBOTS / "xlerobot_2wheels" / "my_xlerobot_2wheels_lab.json"
)
SO_FOLLOWER_CALIBRATION_DIR = HF_LEROBOT_CALIBRATION / ROBOTS / "so_follower"
SO_FOLLOWER_RIGHT_ARM_CALIBRATION_PATH = SO_FOLLOWER_CALIBRATION_DIR / "right_arm.json"
RIGHT_ARM_CALIBRATION_KEY_MAP = (
    ("right_arm_shoulder_pan", "shoulder_pan"),
    ("right_arm_shoulder_lift", "shoulder_lift"),
    ("right_arm_elbow_flex", "elbow_flex"),
    ("right_arm_wrist_flex", "wrist_flex"),
    ("right_arm_wrist_roll", "wrist_roll"),
    ("right_arm_gripper", "gripper"),
)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def _write_json(path: Path, payload: dict) -> None:
    _atomic_write_bytes(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
    )


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with VLA_CAMERA_DEBUG_IO_LOCK:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _encode_image_bytes(frame, *, extension: str, input_is_rgb: bool = False) -> bytes:
    image = frame.copy()
    if input_is_rgb and getattr(image, "ndim", 0) == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

    success, buffer = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError(f"cv2.imencode({extension!r}, frame) returned False.")
    return buffer.tobytes()


def _backend_label(backend: Cv2Backends | int | None) -> str:
    if backend is None:
        return "default"
    try:
        return Cv2Backends(backend).name
    except Exception:
        return str(backend)


def _vla_camera_candidates() -> list[dict[str, object]]:
    if sys.platform.startswith("win"):
        return [
            {"backend": Cv2Backends.DSHOW, "backend_name": "DSHOW", "fourcc": None},
            {"backend": Cv2Backends.DSHOW, "backend_name": "DSHOW", "fourcc": "MJPG"},
            {"backend": Cv2Backends.DSHOW, "backend_name": "DSHOW", "fourcc": "YUY2"},
        ]
    return [{"backend": Cv2Backends.ANY, "backend_name": "ANY", "fourcc": None}]


def _frame_stats(frame) -> dict:
    return {
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "mean": round(float(frame.mean()), 3),
        "std": round(float(frame.std()), 3),
        "min": int(frame.min()),
        "max": int(frame.max()),
    }


def _extract_right_arm_so_follower_calibration(source_payload: dict) -> dict[str, dict[str, int]]:
    mapped_payload: dict[str, dict[str, int]] = {}
    missing_keys: list[str] = []
    required_fields = ("id", "drive_mode", "homing_offset", "range_min", "range_max")

    for source_key, target_key in RIGHT_ARM_CALIBRATION_KEY_MAP:
        raw_item = source_payload.get(source_key)
        if not isinstance(raw_item, dict):
            missing_keys.append(source_key)
            continue

        missing_fields = [field for field in required_fields if field not in raw_item]
        if missing_fields:
            raise RuntimeError(
                f"Calibration entry '{source_key}' is missing required fields: {', '.join(missing_fields)}"
            )

        mapped_payload[target_key] = {field: int(raw_item[field]) for field in required_fields}

    if missing_keys:
        raise RuntimeError(
            "Missing right-arm calibration keys in xlerobot_2wheels calibration: "
            + ", ".join(missing_keys)
        )

    return mapped_payload


def _format_calibration_summary(mapped_payload: dict[str, dict[str, int]]) -> str:
    return ", ".join(
        f"{joint}={values['range_min']}..{values['range_max']}"
        for joint, values in mapped_payload.items()
    )


def _sync_right_arm_calibration_for_so_follower() -> dict[str, str]:
    if not XLEROBOT_2WHEELS_CALIBRATION_PATH.is_file():
        raise RuntimeError(
            "Missing xlerobot_2wheels calibration file: "
            f"{XLEROBOT_2WHEELS_CALIBRATION_PATH}"
        )

    try:
        source_payload = json.loads(XLEROBOT_2WHEELS_CALIBRATION_PATH.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read xlerobot_2wheels calibration file: {XLEROBOT_2WHEELS_CALIBRATION_PATH}"
        ) from exc

    mapped_payload = _extract_right_arm_so_follower_calibration(source_payload)
    SO_FOLLOWER_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(SO_FOLLOWER_RIGHT_ARM_CALIBRATION_PATH, mapped_payload)

    return {
        "source_path": str(XLEROBOT_2WHEELS_CALIBRATION_PATH),
        "target_path": str(SO_FOLLOWER_RIGHT_ARM_CALIBRATION_PATH),
        "status": "overwritten",
        "summary": _format_calibration_summary(mapped_payload),
    }


def _is_blank_frame(frame) -> tuple[bool, str]:
    mean_value = float(frame.mean())
    std_value = float(frame.std())
    min_value = int(frame.min())
    max_value = int(frame.max())

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


class DebugRobotCamera(RobotCamera):
    """RobotCamera with per-capture dumps for raw and LLM input images."""

    def __init__(self, usb_port):
        self.usb_port = usb_port
        self.capture = None
        self.backend_name = "default"
        self.fourcc = "default"
        self.validation_stats = "not validated"
        self._connect_with_validation()

    def _candidate_configs(self):
        if sys.platform.startswith("win"):
            return [
                {"backend": cv2.CAP_DSHOW, "backend_name": "DSHOW", "fourcc": None},
                {"backend": cv2.CAP_DSHOW, "backend_name": "DSHOW", "fourcc": "MJPG"},
                {"backend": cv2.CAP_DSHOW, "backend_name": "DSHOW", "fourcc": "YUY2"},
            ]
        return [{"backend": None, "backend_name": "default", "fourcc": None}]

    def _open_capture(self, *, backend, fourcc):
        if backend is None:
            capture = cv2.VideoCapture(self.usb_port)
        else:
            capture = cv2.VideoCapture(self.usb_port, backend)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        if fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        return capture

    def _warmup_and_read(self, capture):
        frame = None
        for _ in range(CAMERA_WARMUP_FRAMES):
            capture.grab()
        for _ in range(CAMERA_READ_ATTEMPTS):
            ok, frame = capture.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.05)
        raise RuntimeError(f"Failed to read frame from camera {self.usb_port!r}.")

    def _connect_with_validation(self):
        issues: list[str] = []
        for candidate in self._candidate_configs():
            capture = self._open_capture(
                backend=candidate["backend"],
                fourcc=candidate["fourcc"],
            )
            try:
                if not capture.isOpened():
                    issues.append(
                        f"{candidate['backend_name']}/{candidate['fourcc'] or 'default'}: open failed"
                    )
                    continue
                frame = self._warmup_and_read(capture)
                is_blank, stats_message = _is_blank_frame(frame)
                if is_blank:
                    issues.append(
                        f"{candidate['backend_name']}/{candidate['fourcc'] or 'default'}: {stats_message}"
                    )
                    continue
                self.capture = capture
                self.backend_name = candidate["backend_name"]
                self.fourcc = candidate["fourcc"] or "default"
                self.validation_stats = stats_message
                print(
                    f"[Camera] Using camera {self.usb_port} with "
                    f"backend={self.backend_name}, fourcc={self.fourcc}: {stats_message}"
                )
                return
            finally:
                if self.capture is not capture:
                    capture.release()

        issue_text = "\n".join(issues) if issues else "no candidate camera config succeeded"
        raise RuntimeError(f"Failed to open a valid frame from camera {self.usb_port!r}:\n{issue_text}")

    def _read_valid_frame(self):
        last_issue = "unknown"
        for attempt_index in range(1, CAMERA_READ_ATTEMPTS + 1):
            if self.capture is None or not self.capture.isOpened():
                self.reopen()
            self.capture.grab()
            ok, frame = self.capture.read()
            if ok and frame is not None:
                is_blank, stats_message = _is_blank_frame(frame)
                if not is_blank:
                    self.validation_stats = stats_message
                    return frame, stats_message
                last_issue = stats_message
            else:
                last_issue = "cv2.read() returned no frame"

            if attempt_index < CAMERA_READ_ATTEMPTS:
                self.reopen()
                time.sleep(0.1)

        raise RuntimeError(
            f"Camera {self.usb_port!r} returned only invalid frames after {CAMERA_READ_ATTEMPTS} attempts: "
            f"{last_issue}"
        )

    def release(self):
        if self.capture is not None:
            self.capture.release()

    def reopen(self):
        self.release()
        capture = self._open_capture(
            backend=cv2.CAP_DSHOW if self.backend_name == "DSHOW" else None,
            fourcc=None if self.fourcc == "default" else self.fourcc,
        )
        if not capture.isOpened():
            capture.release()
            self._connect_with_validation()
            return
        self.capture = capture
        for _ in range(CAMERA_WARMUP_FRAMES):
            self.capture.grab()

    def capture_image(self, camera_fov=120, center_angle=0, navigation_mode="normal"):
        frame, stats_message = self._read_valid_frame()

        saved_at = datetime.now().astimezone().isoformat(timespec="seconds")
        raw_success, raw_buffer = cv2.imencode(".png", frame)
        if not raw_success:
            raise RuntimeError("cv2.imencode('.png', raw frame) returned False.")
        _atomic_write_bytes(LATEST_RAW_CAMERA_IMAGE_PATH, raw_buffer.tobytes())
        _write_json(
            LATEST_RAW_CAMERA_JSON_PATH,
            {
                "saved_at": saved_at,
                "source_method": "DebugRobotCamera.capture_image.raw",
                "camera_id": self.usb_port,
                "camera_type": "RobotCamera",
                "selected_backend": self.backend_name,
                "selected_fourcc": self.fourcc,
                "validation_stats": self.validation_stats,
                "current_frame_stats": _frame_stats(frame),
                "blank_stats": stats_message,
                "image_path": str(LATEST_RAW_CAMERA_IMAGE_PATH.resolve()),
                "save_status": "ok",
                "save_error": None,
            },
        )

        overlay_frame = basic_augmentation(
            frame.copy(),
            h_fov=camera_fov,
            center_angle=center_angle,
            navigation_mode=navigation_mode,
        )
        overlay_success, overlay_buffer = cv2.imencode(".jpg", overlay_frame)
        if not overlay_success:
            raise RuntimeError("cv2.imencode('.jpg', overlay frame) returned False.")

        jpeg_bytes = overlay_buffer.tobytes()
        _atomic_write_bytes(LATEST_LLM_INPUT_IMAGE_PATH, jpeg_bytes)
        _write_json(
            LATEST_LLM_INPUT_JSON_PATH,
            {
                "saved_at": saved_at,
                "source_method": "DebugRobotCamera.capture_image.overlay",
                "mime_type": "image/jpeg",
                "byte_length": len(jpeg_bytes),
                "camera_fov": camera_fov,
                "center_angle": center_angle,
                "navigation_mode": navigation_mode,
                "image_path": str(LATEST_LLM_INPUT_IMAGE_PATH.resolve()),
                "note": "Saved model input image with RoboCrew grid overlay.",
            },
        )
        return jpeg_bytes


def _new_vla_camera_debug_run_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")


def _update_vla_camera_debug_session(
    run_dir: Path,
    camera_key: str,
    payload: dict[str, object],
) -> None:
    session_path = run_dir / "session.json"
    if not session_path.is_file():
        return

    try:
        session_payload = json.loads(session_path.read_text(encoding="utf-8"))
    except Exception:
        return

    cameras_payload = session_payload.get("cameras")
    if not isinstance(cameras_payload, dict):
        return

    camera_payload = cameras_payload.get(camera_key)
    if not isinstance(camera_payload, dict):
        return

    camera_payload.update(payload)
    _write_json(session_path, session_payload)


def _update_vla_action_debug_session(
    run_dir: Path,
    payload: dict[str, object],
) -> None:
    session_path = run_dir / "session.json"
    if not session_path.is_file():
        return

    try:
        session_payload = json.loads(session_path.read_text(encoding="utf-8"))
    except Exception:
        return

    action_payload = session_payload.get("action_debug")
    if not isinstance(action_payload, dict):
        action_payload = {}
        session_payload["action_debug"] = action_payload

    action_payload.update(payload)
    _write_json(session_path, session_payload)


def _prepare_vla_camera_debug_session(
    camera_configs: dict[str, OpenCVCameraConfig],
    camera_settings: dict[str, dict],
) -> dict[str, object] | None:
    global LAST_VLA_CAMERA_DEBUG_SESSION

    if not VLA_CAMERA_DEBUG_ENABLED:
        LAST_VLA_CAMERA_DEBUG_SESSION = None
        return None

    run_id = _new_vla_camera_debug_run_id()
    run_dir = VLA_CAMERA_DEBUG_DIR / run_id
    snapshots_dir = run_dir / "snapshots"
    events_path = run_dir / "events.jsonl"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    session_payload: dict[str, object] = {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "enabled": True,
        "interval_seconds": VLA_CAMERA_DEBUG_INTERVAL_S,
        "run_dir": str(run_dir.resolve()),
        "events_path": str(events_path.resolve()),
        "policy_name": POLICY_NAME,
        "policy_server": POLICY_SERVER_ADDRESS,
        "keep_grasp_after_vla": KEEP_GRASP_AFTER_VLA,
        "reset_arms_after_grab": RESET_ARMS_AFTER_GRAB,
        "vla_execution_time_s": VLA_EXECUTION_TIME_S,
        "vla_start_pose": VLA_START_POSE_NAME,
        "action_debug": {
            "enabled": VLA_ACTION_DEBUG_ENABLED,
            "events_path": str((run_dir / "action_events.jsonl").resolve()),
            "summary_path": str((run_dir / "action_summary.json").resolve()),
            "status": "pending" if VLA_ACTION_DEBUG_ENABLED else "disabled",
        },
        "cameras": {},
    }

    cameras_payload: dict[str, object] = session_payload["cameras"]  # type: ignore[assignment]
    for camera_key, config in camera_configs.items():
        camera_id = camera_settings[camera_key]["index_or_path"]
        setattr(config, "_debug_enabled", True)
        setattr(config, "_debug_run_id", run_id)
        setattr(config, "_debug_camera_key", camera_key)
        setattr(config, "_debug_camera_id", camera_id)
        setattr(config, "_debug_session_dir", str(run_dir))
        setattr(config, "_debug_events_path", str(events_path))
        setattr(config, "_debug_snapshots_dir", str(snapshots_dir))
        setattr(config, "_debug_interval_s", float(VLA_CAMERA_DEBUG_INTERVAL_S))

        cameras_payload[camera_key] = {
            "camera_id": camera_id,
            "latest_image_path": str((run_dir / f"latest_{camera_key}.jpg").resolve()),
            "snapshot_pattern": str((snapshots_dir / f"NNN_{camera_key}.jpg").resolve()),
            "selected_backend": None,
            "selected_fourcc": None,
            "validation_stats": None,
            "status": "pending",
        }

    _write_json(run_dir / "session.json", session_payload)
    LAST_VLA_CAMERA_DEBUG_SESSION = session_payload
    return session_payload


def _normalize_action_debug_payload(payload: dict[str, object] | None) -> dict[str, float | None]:
    normalized = {joint: None for joint in VLA_ACTION_DEBUG_JOINT_KEYS}
    if not isinstance(payload, dict):
        return normalized

    for joint in VLA_ACTION_DEBUG_JOINT_KEYS:
        value = payload.get(joint)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            normalized[joint] = round(float(value), 4)

    return normalized


def _read_action_debug_present_positions(robot) -> dict[str, float | None] | None:
    bus = getattr(robot, "bus", None)
    if bus is None:
        return None

    try:
        raw_positions = bus.sync_read("Present_Position")
    except Exception:
        return None

    payload: dict[str, float | None] = {}
    for joint in VLA_ACTION_DEBUG_JOINT_KEYS:
        value = raw_positions.get(joint.removesuffix(".pos"))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            payload[joint] = round(float(value), 4)
        else:
            payload[joint] = None

    return payload


def _compute_action_goal_delta(
    performed_action: dict[str, float | None],
    present_position_before: dict[str, float | None] | None,
) -> dict[str, float | None]:
    delta = {joint: None for joint in VLA_ACTION_DEBUG_JOINT_KEYS}
    if present_position_before is None:
        return delta

    for joint in VLA_ACTION_DEBUG_JOINT_KEYS:
        goal = performed_action.get(joint)
        present = present_position_before.get(joint)
        if goal is None or present is None:
            continue
        delta[joint] = round(goal - present, 4)

    return delta


def _build_action_joint_summary(
    action_events: list[dict[str, object]],
) -> dict[str, dict[str, float | None]]:
    joint_summary: dict[str, dict[str, float | None]] = {}

    for joint in VLA_ACTION_DEBUG_JOINT_KEYS:
        raw_values: list[float] = []
        performed_values: list[float] = []
        abs_goal_deltas: list[float] = []

        for event in action_events:
            raw_payload = event.get("raw_action")
            if isinstance(raw_payload, dict):
                raw_value = raw_payload.get(joint)
                if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
                    raw_values.append(float(raw_value))

            performed_payload = event.get("performed_action")
            if isinstance(performed_payload, dict):
                performed_value = performed_payload.get(joint)
                if isinstance(performed_value, (int, float)) and not isinstance(performed_value, bool):
                    performed_values.append(float(performed_value))

            goal_delta_payload = event.get("goal_delta")
            if isinstance(goal_delta_payload, dict):
                delta_value = goal_delta_payload.get(joint)
                if isinstance(delta_value, (int, float)) and not isinstance(delta_value, bool):
                    abs_goal_deltas.append(abs(float(delta_value)))

        joint_summary[joint] = {
            "raw_min": round(min(raw_values), 4) if raw_values else None,
            "raw_max": round(max(raw_values), 4) if raw_values else None,
            "performed_min": round(min(performed_values), 4) if performed_values else None,
            "performed_max": round(max(performed_values), 4) if performed_values else None,
            "max_abs_goal_delta": round(max(abs_goal_deltas), 4) if abs_goal_deltas else None,
        }

    return joint_summary


class DebugVLAOpenCVCamera(OpenCVCamera):
    """OpenCV camera wrapper that saves VLA manipulation frames for debugging."""

    def __init__(self, config: OpenCVCameraConfig):
        super().__init__(config)
        self._debug_enabled = bool(getattr(config, "_debug_enabled", False))
        self._debug_run_id = str(getattr(config, "_debug_run_id", ""))
        self._debug_camera_key = str(getattr(config, "_debug_camera_key", "unknown"))
        self._debug_camera_id = getattr(config, "_debug_camera_id", getattr(config, "index_or_path", "unknown"))
        self._debug_session_dir = Path(str(getattr(config, "_debug_session_dir", VLA_CAMERA_DEBUG_DIR)))
        self._debug_events_path = Path(str(getattr(config, "_debug_events_path", self._debug_session_dir / "events.jsonl")))
        self._debug_snapshots_dir = Path(
            str(getattr(config, "_debug_snapshots_dir", self._debug_session_dir / "snapshots"))
        )
        self._debug_interval_s = float(getattr(config, "_debug_interval_s", VLA_CAMERA_DEBUG_INTERVAL_S))
        self._debug_latest_image_path = self._debug_session_dir / f"latest_{self._debug_camera_key}.jpg"
        self._debug_frame_index = 0
        self._debug_last_saved_monotonic: float | None = None
        self._debug_save_warning_emitted = False
        self._selected_backend_label = _backend_label(self.backend)
        self._selected_fourcc_label = self.config.fourcc or "default"
        self._validation_stats = "not validated"

    def _cleanup_after_failed_connect(self) -> None:
        try:
            if self.thread is not None or self.videocapture is not None:
                self.disconnect()
        except Exception:
            try:
                if self.videocapture is not None:
                    self.videocapture.release()
            except Exception:
                pass
            self.videocapture = None
            self.thread = None
            self.stop_event = None
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    def _current_candidate_summary(self) -> str:
        return f"backend={self._selected_backend_label}, fourcc={self._selected_fourcc_label}"

    def _record_session_result(self, **payload: object) -> None:
        if not self._debug_enabled:
            return
        _update_vla_camera_debug_session(self._debug_session_dir, self._debug_camera_key, dict(payload))

    def connect(self, warmup: bool = True) -> None:
        issues: list[str] = []
        for candidate in _vla_camera_candidates():
            backend = candidate["backend"]
            fourcc = candidate["fourcc"]
            backend_name = str(candidate["backend_name"])
            fourcc_label = str(fourcc or "default")

            self.config.backend = backend
            self.backend = backend
            self.config.fourcc = fourcc
            self._selected_backend_label = backend_name
            self._selected_fourcc_label = fourcc_label

            try:
                super().connect(warmup=warmup)
                frame = OpenCVCamera.read_latest(self, max_age_ms=max(1000, self.warmup_s * 1000 + 500))
                is_blank, stats_message = _is_blank_frame(frame)
                if is_blank:
                    raise RuntimeError(stats_message)

                self._validation_stats = stats_message
                self._record_session_result(
                    selected_backend=self._selected_backend_label,
                    selected_fourcc=self._selected_fourcc_label,
                    validation_stats=self._validation_stats,
                    status="connected",
                    connection_error=None,
                )
                print(
                    f"[VLA Camera] {self._debug_camera_key}({self._debug_camera_id}) "
                    f"using {self._current_candidate_summary()}: {self._validation_stats}"
                )
                return
            except Exception as exc:
                issues.append(f"{backend_name}/{fourcc_label}: {exc}")
                self._cleanup_after_failed_connect()

        error_message = (
            f"VLA camera validation failed for {self._debug_camera_key}({self._debug_camera_id}): "
            + " | ".join(issues)
        )
        self._record_session_result(
            selected_backend=None,
            selected_fourcc=None,
            validation_stats=None,
            status="failed",
            connection_error=error_message,
        )
        raise RuntimeError(error_message)

    def _read_checked_frame(self, max_age_ms: int = 500):
        reconnect_attempted = False
        frame = OpenCVCamera.read_latest(self, max_age_ms=max_age_ms)
        is_blank, stats_message = _is_blank_frame(frame)
        if not is_blank:
            self._validation_stats = stats_message
            return frame, False, reconnect_attempted

        reconnect_attempted = True
        try:
            self._cleanup_after_failed_connect()
            super().connect(warmup=True)
            frame = OpenCVCamera.read_latest(self, max_age_ms=max(max_age_ms, 1000))
            is_blank, stats_message = _is_blank_frame(frame)
            if is_blank:
                raise RuntimeError(
                    f"VLA camera {self._debug_camera_key}({self._debug_camera_id}) remained blank after reconnect "
                    f"with {self._current_candidate_summary()}: {stats_message}"
                )
        except Exception as exc:
            self._record_session_result(
                selected_backend=self._selected_backend_label,
                selected_fourcc=self._selected_fourcc_label,
                validation_stats=None,
                status="failed",
                connection_error=str(exc),
            )
            raise

        self._validation_stats = stats_message
        self._record_session_result(
            selected_backend=self._selected_backend_label,
            selected_fourcc=self._selected_fourcc_label,
            validation_stats=self._validation_stats,
            status="connected",
            connection_error=None,
        )
        return frame, True, reconnect_attempted

    def _should_save_debug_frame(self, now_monotonic: float) -> bool:
        if not self._debug_enabled:
            return False
        if self._debug_last_saved_monotonic is None:
            return True
        return (now_monotonic - self._debug_last_saved_monotonic) >= self._debug_interval_s

    def _save_debug_frame(self, frame, *, blank_detected: bool, reconnect_attempted: bool) -> None:
        if not self._debug_enabled:
            return

        self._debug_session_dir.mkdir(parents=True, exist_ok=True)
        self._debug_snapshots_dir.mkdir(parents=True, exist_ok=True)

        input_is_rgb = self.color_mode == ColorMode.RGB
        jpeg_bytes = _encode_image_bytes(frame, extension=".jpg", input_is_rgb=input_is_rgb)
        snapshot_path = self._debug_snapshots_dir / f"{self._debug_frame_index:03d}_{self._debug_camera_key}.jpg"
        saved_at = datetime.now().astimezone().isoformat(timespec="milliseconds")

        _atomic_write_bytes(self._debug_latest_image_path, jpeg_bytes)
        _atomic_write_bytes(snapshot_path, jpeg_bytes)
        _append_jsonl(
            self._debug_events_path,
            {
                "saved_at": saved_at,
                "run_id": self._debug_run_id,
                "camera_key": self._debug_camera_key,
                "camera_id": self._debug_camera_id,
                "frame_index": self._debug_frame_index,
                "shape": list(frame.shape),
                "mean": round(float(frame.mean()), 3),
                "std": round(float(frame.std()), 3),
                "min": int(frame.min()),
                "max": int(frame.max()),
                "selected_backend": self._selected_backend_label,
                "selected_fourcc": self._selected_fourcc_label,
                "validation_stats": self._validation_stats,
                "blank_detected": blank_detected,
                "reconnect_attempted": reconnect_attempted,
                "latest_image_path": str(self._debug_latest_image_path.resolve()),
                "snapshot_path": str(snapshot_path.resolve()),
            },
        )
        self._debug_frame_index += 1
        self._debug_last_saved_monotonic = time.monotonic()

    def read_latest(self, max_age_ms: int = 500):
        frame, blank_detected, reconnect_attempted = self._read_checked_frame(max_age_ms=max_age_ms)

        if self._should_save_debug_frame(time.monotonic()):
            try:
                self._save_debug_frame(
                    frame,
                    blank_detected=blank_detected,
                    reconnect_attempted=reconnect_attempted,
                )
            except Exception as exc:
                if not self._debug_save_warning_emitted:
                    print(
                        f"[VLA Debug] Failed to save {self._debug_camera_key} frame "
                        f"for run {self._debug_run_id}: {exc}"
                    )
                    self._debug_save_warning_emitted = True

        return frame


def _patch_vla_opencv_camera() -> dict[str, type[OpenCVCamera]]:
    original_classes = {
        "package": lerobot_opencv_package.OpenCVCamera,
        "module": opencv_camera_module.OpenCVCamera,
    }
    lerobot_opencv_package.OpenCVCamera = DebugVLAOpenCVCamera
    opencv_camera_module.OpenCVCamera = DebugVLAOpenCVCamera
    return original_classes


def _restore_vla_opencv_camera(original_classes: dict[str, type[OpenCVCamera]]) -> None:
    lerobot_opencv_package.OpenCVCamera = original_classes["package"]
    opencv_camera_module.OpenCVCamera = original_classes["module"]


class DebugRobotClient(RobotClient):
    """RobotClient variant that logs policy actions during VLA manipulation."""

    def __init__(self, config: RobotClientConfig, *, debug_run_dir: Path | None = None):
        self._debug_run_dir = debug_run_dir
        self._action_debug_enabled = VLA_ACTION_DEBUG_ENABLED and debug_run_dir is not None
        self._action_events_path = debug_run_dir / "action_events.jsonl" if debug_run_dir is not None else None
        self._action_summary_path = debug_run_dir / "action_summary.json" if debug_run_dir is not None else None
        self._action_events: list[dict[str, object]] = []
        self._action_debug_warning_emitted = False
        super().__init__(config)

        if self._action_debug_enabled and self._debug_run_dir is not None:
            _update_vla_action_debug_session(
                self._debug_run_dir,
                {
                    "status": "active",
                    "action_count": 0,
                },
            )

    def _record_action_event(
        self,
        *,
        policy_timestep: int,
        raw_action: dict[str, float | None],
        performed_action: dict[str, float | None],
        present_position_before: dict[str, float | None] | None,
        goal_delta: dict[str, float | None],
        queue_size_after_pop: int,
        action_timestamp: float,
    ) -> None:
        if not self._action_debug_enabled or self._action_events_path is None:
            return

        event_payload: dict[str, object] = {
            "saved_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "policy_timestep": int(policy_timestep),
            "action_timestamp": round(float(action_timestamp), 6),
            "queue_size_after_pop": int(queue_size_after_pop),
            "raw_action": raw_action,
            "performed_action": performed_action,
            "present_position_before": present_position_before,
            "goal_delta": goal_delta,
        }

        try:
            _append_jsonl(self._action_events_path, event_payload)
            self._action_events.append(event_payload)
            if self._debug_run_dir is not None:
                _update_vla_action_debug_session(
                    self._debug_run_dir,
                    {
                        "status": "active",
                        "action_count": len(self._action_events),
                    },
                )
        except Exception as exc:
            if not self._action_debug_warning_emitted:
                print(f"[VLA Action Debug] Failed to save action event: {exc}")
                self._action_debug_warning_emitted = True

    def _write_action_summary(self) -> None:
        if not self._action_debug_enabled or self._action_summary_path is None or self._debug_run_dir is None:
            return

        final_present_position = _read_action_debug_present_positions(self.robot)
        initial_present_position = None
        if self._action_events:
            first_event = self._action_events[0]
            first_present_position = first_event.get("present_position_before")
            if isinstance(first_present_position, dict):
                initial_present_position = first_present_position

        summary_payload: dict[str, object] = {
            "run_id": self._debug_run_dir.name,
            "action_events_path": str(self._action_events_path.resolve()),
            "action_count": len(self._action_events),
            "initial_present_position": initial_present_position,
            "final_present_position": final_present_position,
            "extension_joint_keys": list(VLA_ACTION_EXTENSION_JOINT_KEYS),
            "joint_summary": _build_action_joint_summary(self._action_events),
        }
        _write_json(self._action_summary_path, summary_payload)
        _update_vla_action_debug_session(
            self._debug_run_dir,
            {
                "status": "completed",
                "action_count": len(self._action_events),
                "summary_written": True,
            },
        )

    def control_loop_action(self, verbose: bool = False) -> dict[str, object]:
        """Read queued policy actions, send them, and log joint-level debug telemetry."""

        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action = self.action_queue.get_nowait()
            current_queue_size = self.action_queue.qsize()
        get_end = time.perf_counter() - get_start

        raw_action = self._action_tensor_to_action_dict(timed_action.get_action())
        normalized_raw_action = _normalize_action_debug_payload(raw_action)
        present_position_before = _read_action_debug_present_positions(self.robot)
        performed_action = self.robot.send_action(raw_action)
        normalized_performed_action = _normalize_action_debug_payload(performed_action)
        goal_delta = _compute_action_goal_delta(normalized_performed_action, present_position_before)

        self._record_action_event(
            policy_timestep=int(timed_action.get_timestep()),
            raw_action=normalized_raw_action,
            performed_action=normalized_performed_action,
            present_position_before=present_position_before,
            goal_delta=goal_delta,
            queue_size_after_pop=current_queue_size,
            action_timestamp=float(timed_action.get_timestamp()),
        )

        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return performed_action

    def stop(self):
        try:
            self._write_action_summary()
        except Exception as exc:
            if self._debug_run_dir is not None:
                _update_vla_action_debug_session(
                    self._debug_run_dir,
                    {
                        "status": "failed",
                        "summary_written": False,
                        "error": str(exc),
                    },
                )
            if not self._action_debug_warning_emitted:
                print(f"[VLA Action Debug] Failed to write action summary: {exc}")
                self._action_debug_warning_emitted = True
        finally:
            super().stop()


class TwoWheelServoControler(servo_controls_module.ServoControler):
    """2-wheel differential-drive variant of RoboCrew's XLeRobot controller."""

    def __init__(
        self,
        right_arm_wheel_usb: str | None = None,
        left_arm_head_usb: str | None = None,
        *,
        speed: int = servo_controls_module.DEFAULT_SPEED,
    ) -> None:
        self.right_arm_wheel_usb = right_arm_wheel_usb
        self.left_arm_head_usb = left_arm_head_usb
        self.speed = speed
        self.action_map = TWO_WHEEL_ACTION_MAP
        self._wheel_ids = tuple(TWO_WHEEL_ACTION_MAP["forward"].keys())
        self._head_ids = tuple(servo_controls_module.HEAD_SERVO_MAP.values())
        self._right_arm_ids = tuple(servo_controls_module.ARM_SERVO_MAPS["right"].values())
        self._left_arm_ids = tuple(servo_controls_module.ARM_SERVO_MAPS["left"].values())
        self._arm_positions_right = {
            name: 0.0 for name in servo_controls_module.ARM_SERVO_MAPS["right"].keys()
        }
        self._arm_positions_left = {
            name: 0.0 for name in servo_controls_module.ARM_SERVO_MAPS["left"].keys()
        }
        self._arm_positions = {name: 0.0 for name in servo_controls_module.ARM_SERVO_MAPS["right"].keys()}

        right_arm_calibration = servo_controls_module._load_arm_calibration(
            "right_arm.json",
            self._right_arm_ids,
            right_arm_wheel_usb,
        )
        right_arm_calibration.update(
            {
                BASE_LEFT_WHEEL_ID: servo_controls_module.MotorCalibration(
                    id=BASE_LEFT_WHEEL_ID,
                    drive_mode=1,
                    homing_offset=0,
                    range_min=0,
                    range_max=4095,
                ),
                BASE_RIGHT_WHEEL_ID: servo_controls_module.MotorCalibration(
                    id=BASE_RIGHT_WHEEL_ID,
                    drive_mode=0,
                    homing_offset=0,
                    range_min=0,
                    range_max=4095,
                ),
            }
        )
        left_arm_calibration = servo_controls_module._load_arm_calibration(
            "left_arm.json",
            self._left_arm_ids,
            left_arm_head_usb,
        )

        if right_arm_wheel_usb:
            arm_motors = {
                aid: servo_controls_module.Motor(
                    aid,
                    "sts3215",
                    servo_controls_module.POSITION_NORM_MODE,
                )
                for aid in self._right_arm_ids
            }
            self.wheel_bus = servo_controls_module.FeetechMotorsBus(
                port=right_arm_wheel_usb,
                motors={
                    **arm_motors,
                    BASE_LEFT_WHEEL_ID: servo_controls_module.Motor(
                        BASE_LEFT_WHEEL_ID,
                        "sts3215",
                        servo_controls_module.MotorNormMode.RANGE_M100_100,
                    ),
                    BASE_RIGHT_WHEEL_ID: servo_controls_module.Motor(
                        BASE_RIGHT_WHEEL_ID,
                        "sts3215",
                        servo_controls_module.MotorNormMode.RANGE_M100_100,
                    ),
                },
                calibration=right_arm_calibration,
            )
            self.wheel_bus.connect()
            self.apply_wheel_modes()
            self.apply_arm_modes()

        head_calibration = {
            **left_arm_calibration,
            servo_controls_module.HEAD_SERVO_MAP["yaw"]: servo_controls_module.MotorCalibration(
                id=servo_controls_module.HEAD_SERVO_MAP["yaw"],
                drive_mode=0,
                homing_offset=0,
                range_min=0,
                range_max=4095,
            ),
            servo_controls_module.HEAD_SERVO_MAP["pitch"]: servo_controls_module.MotorCalibration(
                id=servo_controls_module.HEAD_SERVO_MAP["pitch"],
                drive_mode=0,
                homing_offset=0,
                range_min=0,
                range_max=4095,
            ),
        }

        if left_arm_head_usb:
            left_arm_motors = {
                aid: servo_controls_module.Motor(
                    aid,
                    "sts3215",
                    servo_controls_module.POSITION_NORM_MODE,
                )
                for aid in self._left_arm_ids
            }
            self.head_bus = servo_controls_module.FeetechMotorsBus(
                port=left_arm_head_usb,
                motors={
                    **left_arm_motors,
                    servo_controls_module.HEAD_SERVO_MAP["yaw"]: servo_controls_module.Motor(
                        servo_controls_module.HEAD_SERVO_MAP["yaw"],
                        "sts3215",
                        servo_controls_module.HEAD_NORM_MODE,
                    ),
                    servo_controls_module.HEAD_SERVO_MAP["pitch"]: servo_controls_module.Motor(
                        servo_controls_module.HEAD_SERVO_MAP["pitch"],
                        "sts3215",
                        servo_controls_module.HEAD_NORM_MODE,
                    ),
                },
                calibration=head_calibration,
            )
            self.head_bus.connect()
            self.apply_head_modes()
            self.apply_arm_modes()
            self._head_positions = {
                servo_controls_module.HEAD_SERVO_MAP["yaw"]: 0.0,
                servo_controls_module.HEAD_SERVO_MAP["pitch"]: 0.0,
            }


def _shutdown_robot_client(client: RobotClient) -> None:
    client.stop()


def _disconnect_wheel_bus_for_vla(servo_controler: TwoWheelServoControler) -> None:
    wheel_bus = getattr(servo_controler, "wheel_bus", None)
    if wheel_bus is None:
        return
    try:
        wheel_bus.disconnect()
    except Exception:
        pass


def _reconnect_wheel_bus_after_vla(servo_controler: TwoWheelServoControler) -> None:
    wheel_bus = getattr(servo_controler, "wheel_bus", None)
    if wheel_bus is None:
        return
    wheel_bus.connect()
    servo_controler.apply_wheel_modes()
    servo_controler.apply_arm_modes()


def _saved_position_path(servo_controler: TwoWheelServoControler, position_name: str) -> Path:
    position_file = getattr(servo_controler, "_arm_position_file", None)
    if callable(position_file):
        try:
            return Path(position_file(position_name))
        except Exception:
            pass

    base_dir = Path(
        getattr(servo_controls_module, "DEFAULT_ARM_POSITION_DIR", "~/.cache/robocrew/positions/")
    ).expanduser()
    file_name = position_name if position_name.endswith(".json") else f"{position_name}.json"
    return base_dir / file_name


def _validate_saved_position_for_arm(
    servo_controler: TwoWheelServoControler,
    position_name: str,
    arm_side: str,
) -> str | None:
    position_path = _saved_position_path(servo_controler, position_name)
    if not position_path.is_file():
        return (
            f"Required saved pose '{position_name}' for the {arm_side} arm was not found at "
            f"{position_path}. Record this pose before running the notebook policy."
        )

    try:
        raw_data = json.loads(position_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"Failed to read saved pose '{position_name}' at {position_path}: {exc}"

    if not isinstance(raw_data, dict):
        return f"Saved pose '{position_name}' at {position_path} must contain a JSON object."

    saved_side = raw_data.get("arm_side")
    positions = raw_data.get("positions")
    if saved_side is None and "left" in raw_data and "right" in raw_data:
        saved_side = "both"
        positions = raw_data

    if saved_side != arm_side:
        return (
            f"Saved pose '{position_name}' is for '{saved_side}', but this right-arm policy requires "
            f"an arm_side='{arm_side}' pose. Re-save the pose for the right arm only."
        )

    if not isinstance(positions, dict):
        return f"Saved pose '{position_name}' at {position_path} has no valid positions object."

    required_joints = tuple(servo_controls_module.ARM_SERVO_MAPS[arm_side].keys())
    missing_joints = [joint for joint in required_joints if joint not in positions]
    if missing_joints:
        return (
            f"Saved pose '{position_name}' at {position_path} is missing joints: "
            + ", ".join(missing_joints)
        )

    return None


def _disconnect_servo_controler_for_exit(servo_controler: TwoWheelServoControler) -> None:
    if not (KEEP_GRASP_AFTER_VLA and TASK_COMPLETION_STATE.get("completed")):
        servo_controler.disconnect()
        return

    print("[Debug] Preserving right-arm torque after successful notebook grab.")
    wheel_bus = getattr(servo_controler, "wheel_bus", None)
    if wheel_bus is not None:
        try:
            servo_controler._wheels_stop()
        except Exception:
            pass
        try:
            wheel_bus.disconnect(disable_torque=False)
        except Exception:
            pass

    head_bus = getattr(servo_controler, "head_bus", None)
    if head_bus is not None:
        try:
            head_bus.disconnect()
        except Exception:
            pass


RIGHT_ARM_CALIBRATION_SYNC_INFO = _sync_right_arm_calibration_for_so_follower()


def create_vla_single_arm_manipulation_for_2wheels(
    *,
    tool_name: str,
    tool_description: str,
    task_prompt: str,
    server_address: str,
    policy_name: str,
    policy_type: str,
    arm_port: str,
    servo_controler: TwoWheelServoControler,
    camera_config: dict[str, dict],
    main_camera_object: RobotCamera,
    completion_state: dict[str, bool] | None = None,
    execution_time: int = VLA_EXECUTION_TIME_S,
    policy_device: str = "cpu",
    fps: int = 30,
    actions_per_chunk: int = 50,
):
    configured_cameras = {}
    default_vla_candidate = _vla_camera_candidates()[0]
    for cam_name, cam_settings in camera_config.items():
        configured_cameras[cam_name] = OpenCVCameraConfig(
            index_or_path=cam_settings["index_or_path"],
            width=cam_settings.get("width", 640),
            height=cam_settings.get("height", 480),
            fps=cam_settings.get("fps", 30),
            backend=default_vla_candidate["backend"],
            fourcc=default_vla_candidate["fourcc"],
        )

    robot_config = SOFollowerConfig(
        port=arm_port,
        cameras=configured_cameras,
    )
    robot_config.type = "so101_follower"
    robot_config.id = "right_arm"
    robot_config.calibration_dir = SO_FOLLOWER_CALIBRATION_DIR
    robot_config.disable_torque_on_disconnect = not KEEP_GRASP_AFTER_VLA

    cfg = RobotClientConfig(
        robot=robot_config,
        task=task_prompt,
        server_address=server_address,
        policy_type=policy_type,
        pretrained_name_or_path=policy_name,
        policy_device=policy_device,
        actions_per_chunk=actions_per_chunk,
        chunk_size_threshold=0.5,
        fps=fps,
    )

    @tool
    def manipulation_tool() -> str:
        """Tool description to override."""
        print("Manipulation tool activated")
        if completion_state is not None:
            completion_state["completed"] = False

        pose_error = _validate_saved_position_for_arm(servo_controler, VLA_START_POSE_NAME, "right")
        if pose_error is not None:
            print(f"[VLA Debug] {pose_error}")
            return pose_error

        try:
            servo_controler.set_saved_position(VLA_START_POSE_NAME, arm_side="right")
        except Exception as exc:
            failure_reason = f"Failed to move right arm to saved pose '{VLA_START_POSE_NAME}': {exc}"
            print(f"[VLA Debug] {failure_reason}")
            return failure_reason

        servo_controler.turn_head_to_vla_position()
        time.sleep(0.3)

        main_camera_object.release()
        time.sleep(0.5)
        _disconnect_wheel_bus_for_vla(servo_controler)

        debug_session = _prepare_vla_camera_debug_session(configured_cameras, camera_config)
        original_vla_camera_classes = _patch_vla_opencv_camera() if debug_session is not None else None
        client = None
        failure_reason = None
        try:
            if debug_session is not None:
                print(f"[VLA Debug] Run directory: {debug_session['run_dir']}")
                if VLA_ACTION_DEBUG_ENABLED:
                    run_dir = Path(str(debug_session["run_dir"]))
                    print(f"[VLA Action Debug] Events: {run_dir / 'action_events.jsonl'}")
                    print(f"[VLA Action Debug] Summary: {run_dir / 'action_summary.json'}")
            client = DebugRobotClient(
                cfg,
                debug_run_dir=Path(str(debug_session["run_dir"])) if debug_session is not None else None,
            )
            if not client.start():
                failure_reason = "Failed to connect to robot server."
            else:
                threading.Thread(target=client.receive_actions, daemon=True).start()
                threading.Timer(execution_time, _shutdown_robot_client, args=(client,)).start()
                client.control_loop(task=task_prompt)
        except Exception as exc:
            failure_reason = f"Manipulation failed: {exc}"
            print(f"[VLA Debug] {failure_reason}")
        finally:
            if client is not None:
                try:
                    client.stop()
                except Exception:
                    pass

            if original_vla_camera_classes is not None:
                _restore_vla_opencv_camera(original_vla_camera_classes)
            _reconnect_wheel_bus_after_vla(servo_controler)
            time.sleep(0.5)
            main_camera_object.reopen()
            time.sleep(0.3)
            servo_controler.turn_head_to_vla_position(50)
            if RESET_ARMS_AFTER_GRAB:
                servo_controler.set_saved_position("default", arm_side="both")
            else:
                print("[VLA Debug] Keeping final right-arm grasp pose; not resetting arms to default.")
            if debug_session is not None:
                print(f"[VLA Debug] Saved session: {debug_session['run_dir']}")

        if failure_reason is not None:
            return failure_reason
        if completion_state is not None:
            completion_state["completed"] = True
        return PICKUP_SUCCESS_MESSAGE

    manipulation_tool.name = tool_name
    manipulation_tool.description = tool_description
    return manipulation_tool


main_camera = DebugRobotCamera(MAIN_CAMERA_ID)
servo_controler = TwoWheelServoControler(RIGHT_ARM_WHEEL_USB, LEFT_ARM_HEAD_USB)

move_forward = create_move_forward(servo_controler)
move_backward = create_move_backward(servo_controler)
turn_left = create_turn_left(servo_controler)
turn_right = create_turn_right(servo_controler)
look_around = create_look_around(servo_controler, main_camera)
go_to_precision_mode = create_go_to_precision_mode(servo_controler)

pick_up_notebook = create_vla_single_arm_manipulation_for_2wheels(
    tool_name=PICKUP_TOOL_NAME,
    tool_description="Manipulation tool to grab a notebook from the table.",
    task_prompt=PICKUP_TASK_PROMPT,
    server_address=POLICY_SERVER_ADDRESS,
    policy_name=POLICY_NAME,
    policy_type=POLICY_TYPE,
    arm_port=RIGHT_ARM_WHEEL_USB,
    servo_controler=servo_controler,
    camera_config={
        "main": {"index_or_path": MAIN_CAMERA_ID},
        "right_arm": {"index_or_path": RIGHT_ARM_CAMERA_ID},
    },
    main_camera_object=main_camera,
    completion_state=TASK_COMPLETION_STATE,
    execution_time=VLA_EXECUTION_TIME_S,
    policy_device=POLICY_DEVICE,
)

agent = NotebookGrabAgent(
    model=MODEL_NAME,
    system_prompt=SYSTEM_PROMPT,
    tools=[
        move_forward,
        move_backward,
        turn_left,
        turn_right,
        look_around,
        go_to_precision_mode,
        pick_up_notebook,
    ],
    history_len=8,
    main_camera=main_camera,
    camera_fov=90,
    servo_controler=servo_controler,
    completion_state=TASK_COMPLETION_STATE,
    completion_tool_name=PICKUP_TOOL_NAME,
)

agent.task = TASK


def main() -> None:
    print(f"[Debug] Model: {MODEL_NAME}")
    print(f"[Debug] Task: {TASK}")
    print(f"[Debug] Main camera id: {MAIN_CAMERA_ID}")
    print(f"[Debug] Right arm camera id: {RIGHT_ARM_CAMERA_ID}")
    print(f"[Debug] Left arm + head port: {LEFT_ARM_HEAD_USB}")
    print(f"[Debug] Right arm + base port: {RIGHT_ARM_WHEEL_USB}")
    print(f"[Debug] Base wheel ids: {BASE_LEFT_WHEEL_ID}, {BASE_RIGHT_WHEEL_ID}")
    print(f"[Debug] Policy server: {POLICY_SERVER_ADDRESS}")
    print(f"[Debug] Policy name: {POLICY_NAME}")
    print(f"[Debug] Policy type: {POLICY_TYPE}")
    print(f"[Debug] Policy device: {POLICY_DEVICE}")
    print(f"[Debug] Keep grasp after VLA: {KEEP_GRASP_AFTER_VLA}")
    print(f"[Debug] Reset arms after grab: {RESET_ARMS_AFTER_GRAB}")
    print(f"[Debug] VLA execution time: {VLA_EXECUTION_TIME_S}s")
    print(f"[Debug] VLA right-arm start pose: {VLA_START_POSE_NAME}")
    print(f"[Debug] Calibration source: {RIGHT_ARM_CALIBRATION_SYNC_INFO['source_path']}")
    print(f"[Debug] Calibration target: {RIGHT_ARM_CALIBRATION_SYNC_INFO['target_path']}")
    print(f"[Debug] Calibration sync: {RIGHT_ARM_CALIBRATION_SYNC_INFO['status']}")
    print(f"[Debug] Right arm calibration ranges: {RIGHT_ARM_CALIBRATION_SYNC_INFO['summary']}")
    print(f"[Debug] Camera backend: {main_camera.backend_name}")
    print(f"[Debug] Camera fourcc: {main_camera.fourcc}")
    print(f"[Debug] Camera validation: {main_camera.validation_stats}")
    print(f"[Debug] Raw camera dump: {LATEST_RAW_CAMERA_IMAGE_PATH}")
    print(f"[Debug] Raw camera metadata: {LATEST_RAW_CAMERA_JSON_PATH}")
    print(f"[Debug] LLM input dump: {LATEST_LLM_INPUT_IMAGE_PATH}")
    print(f"[Debug] LLM input metadata: {LATEST_LLM_INPUT_JSON_PATH}")
    print(f"[Debug] VLA camera debug enabled: {VLA_CAMERA_DEBUG_ENABLED}")
    print(f"[Debug] VLA camera debug base dir: {VLA_CAMERA_DEBUG_DIR}")
    print(f"[Debug] VLA action debug enabled: {VLA_ACTION_DEBUG_ENABLED}")
    print(f"[Debug] VLA debug camera ids: main={MAIN_CAMERA_ID}, right_arm={RIGHT_ARM_CAMERA_ID}")
    print(
        "[Debug] VLA camera candidates: "
        + ", ".join(
            f"{candidate['backend_name']}/{candidate['fourcc'] or 'default'}" for candidate in _vla_camera_candidates()
        )
    )

    try:
        # Start with the head tilted toward the tabletop so the LLM can search for a notebook.
        servo_controler.turn_head_to_vla_position(50)
        time.sleep(0.3)
        agent.go()
    finally:
        try:
            _disconnect_servo_controler_for_exit(servo_controler)
        except Exception:
            pass
        try:
            main_camera.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()
