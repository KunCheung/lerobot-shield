from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import cv2
import numpy as np



REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
EXPECTED_LEROBOT_INIT_PATH = (SRC_PATH / "lerobot" / "__init__.py").resolve()
DOTENV_PATH = Path(__file__).with_name(".env")
MEMORY_DB_PATH = Path(os.getenv("TEMP", str(Path.cwd()))) / "robocrew_robot_memory.db"
TMP_IMAGES_DIR = Path(__file__).with_name("tmp_images")
LATEST_RAW_CAMERA_IMAGE_PATH = TMP_IMAGES_DIR / "latest_raw_camera.png"
LATEST_LLM_INPUT_IMAGE_PATH = TMP_IMAGES_DIR / "latest_llm_input.jpg"

MODEL_NAME = "openai:kimi-k2.5"
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
TASK = "Approach a human."
CAMERA_ID = 1
ROBOT_ID = "my_xlerobot_2wheels_lab"
PORT1 = "COM5"  # left arm + head
PORT2 = "COM4"  # right arm + 2-wheel base
LINEAR_SPEED_MPS = 0.10
ANGULAR_SPEED_DPS = 30.0
MAX_FORWARD_STEP_M = 0.20


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


os.environ["MOONSHOT_API_KEY"] = os.getenv("MOONSHOT_API_KEY")
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
import robocrew.core.LLMAgent as robocrew_llm_module
from robocrew.core.utils import basic_augmentation
from robocrew.robots.XLeRobot.tools import create_turn_left, create_turn_right
from langchain.chat_models import init_chat_model as lc_init_chat_model
from langchain_core.tools import tool

try:
    from lerobot.scripts.lerobot_find_cameras import is_blank_frame as _is_blank_frame
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


def _format_backend_label(config: OpenCVCameraConfig) -> str:
    return getattr(config.backend, "name", str(config.backend))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def _encode_frame_bytes(frame: Any, *, extension: str, input_is_rgb: bool) -> bytes:
    if cv2 is None:
        raise RuntimeError("opencv-python is required to save RoboCrew camera frames.")

    image = np.asarray(frame).copy() if np is not None else frame.copy()
    if input_is_rgb and getattr(image, "ndim", 0) == 3:
        channels = image.shape[2]
        if channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

    success, buffer = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError(f"cv2.imencode({extension!r}, frame) returned False.")
    return buffer.tobytes()


def _build_validated_opencv_camera(camera_id: int) -> tuple[OpenCVCamera, dict[str, Any]]:
    cv_config = OpenCVCameraConfig(
        index_or_path=camera_id,
        color_mode=ColorMode.RGB,
        backend=Cv2Backends.DSHOW,
    )
    instance = OpenCVCamera(cv_config)
    backend_label = _format_backend_label(cv_config)
    fourcc_label = cv_config.fourcc or "default"

    print(
        f"[Camera] Opening OpenCV camera {camera_id} with "
        f"backend={backend_label}, fourcc={fourcc_label}"
    )
    instance.connect(warmup=True)
    validation_frame = instance.read()
    is_blank, stats_message = _is_blank_frame(validation_frame)
    if is_blank:
        instance.disconnect()
        raise RuntimeError(
            f"Fixed camera configuration {backend_label}/{fourcc_label} failed validation: {stats_message}"
        )

    meta = {
        "type": "OpenCV",
        "id": camera_id,
        "selected_backend": backend_label,
        "selected_fourcc": fourcc_label,
        "validation_stats": stats_message,
    }
    print(f"[Camera] Using backend={backend_label}, fourcc={fourcc_label}: {stats_message}")
    return instance, meta


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

    def _read_validated_frame(self) -> Any:
        frame = self._camera.read()
        is_blank, stats_message = _is_blank_frame(frame)
        if is_blank:
            raise RuntimeError(f"ValidatedMainCamera.read: {stats_message}")
        return frame

    def _save_latest_raw_frame(self, frame: Any) -> None:
        raw_bytes = _encode_frame_bytes(
            frame,
            extension=".png",
            input_is_rgb=getattr(self._camera, "color_mode", None) == ColorMode.RGB,
        )
        _atomic_write_bytes(LATEST_RAW_CAMERA_IMAGE_PATH, raw_bytes)

    def _save_latest_overlay_frame(self, jpeg_bytes: bytes) -> None:
        _atomic_write_bytes(LATEST_LLM_INPUT_IMAGE_PATH, jpeg_bytes)

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

        frame = self._read_validated_frame()
        self._save_latest_raw_frame(frame)
        overlay_frame = self._build_overlay_frame(
            frame,
            camera_fov=camera_fov,
            center_angle=center_angle,
            navigation_mode=navigation_mode,
        )
        jpeg_bytes = _encode_frame_bytes(
            overlay_frame,
            extension=".jpg",
            input_is_rgb=False,
        )
        self._save_latest_overlay_frame(jpeg_bytes)
        return jpeg_bytes

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


def _print_llm_tool_calls(response: Any) -> None:
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls:
        return

    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name", "<unknown>")
            tool_args = tool_call.get("args", {})
        else:
            tool_name = getattr(tool_call, "name", "<unknown>")
            tool_args = getattr(tool_call, "args", {})
        print(f"[LLM Tool] {tool_name} args={tool_args!r}")


class DebugChatModel:
    """Wrap a LangChain chat model and print only text plus tool-call arguments."""

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
        response = self._wrapped.invoke(*args, **kwargs)
        _print_llm_text_output(response)
        _print_llm_tool_calls(response)
        return response

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        response = await self._wrapped.ainvoke(*args, **kwargs)
        _print_llm_text_output(response)
        _print_llm_tool_calls(response)
        return response

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


def create_safe_move_forward(servo_controller: TwoWheelsServoAdapter):
    @tool
    def move_forward(distance_meters: float) -> str:
        """Drives the robot forward or backward in short safe steps and reassesses after each move."""

        requested_distance = float(distance_meters)
        if requested_distance >= 0:
            actual_distance = servo_controller.go_forward(requested_distance)
            direction = "forward"
        else:
            actual_distance = servo_controller.go_backward(-requested_distance)
            direction = "backward"

        return (
            f"Moved {direction} {actual_distance:.2f} meters "
            f"(requested {abs(requested_distance):.2f} meters)."
        )

    return move_forward


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
            max_distance_per_step_m=MAX_FORWARD_STEP_M,
        )

        tools = [
            create_safe_move_forward(servo_controler),
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
        print(f"[Debug] Latest raw camera image: {LATEST_RAW_CAMERA_IMAGE_PATH}")
        print(f"[Debug] Latest model input image: {LATEST_LLM_INPUT_IMAGE_PATH}")
        print(f"[Debug] Max forward step per tool call: {MAX_FORWARD_STEP_M:.2f}m")

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
