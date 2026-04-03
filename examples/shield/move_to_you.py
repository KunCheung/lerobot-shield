from __future__ import annotations

import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
DOTENV_PATH = Path(__file__).with_name(".env")
MEMORY_DB_PATH = Path(os.getenv("TEMP", str(Path.cwd()))) / "robocrew_robot_memory.db"

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

from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels, XLerobot2WheelsConfig
from lerobot.utils.errors import DeviceNotConnectedError
from robocrew.core.camera import RobotCamera
import robocrew.core.LLMAgent as robocrew_llm_module
from robocrew.robots.XLeRobot.tools import create_move_forward, create_turn_left, create_turn_right
from langchain.chat_models import init_chat_model as lc_init_chat_model


def _patched_init_chat_model(model: str, *args, **kwargs):
    if model in {"kimi-k2.5", "openai:kimi-k2.5"}:
        kwargs.setdefault("base_url", MOONSHOT_BASE_URL)
        kwargs.setdefault("api_key", moonshot_api_key)
        if ":" not in model:
            model = "openai:kimi-k2.5"
    return lc_init_chat_model(model, *args, **kwargs)


robocrew_llm_module.init_chat_model = _patched_init_chat_model
LLMAgent = robocrew_llm_module.LLMAgent


class TwoWheelsServoAdapter:
    """Minimal RoboCrew-compatible wrapper around XLerobot2Wheels."""

    def __init__(
        self,
        robot: XLerobot2Wheels,
        *,
        right_arm_wheel_usb: str,
        linear_speed_mps: float,
        angular_speed_dps: float,
    ) -> None:
        self.robot = robot
        self.right_arm_wheel_usb = right_arm_wheel_usb
        # Keep this falsy so LLMAgent skips head and arm initialization.
        self.left_arm_head_usb = None
        self.linear_speed_mps = linear_speed_mps
        self.angular_speed_dps = angular_speed_dps

    def _run_for_duration(self, *, x_vel: float = 0.0, theta_vel: float = 0.0, duration_s: float) -> None:
        if duration_s <= 0:
            return

        try:
            self.robot.send_action({"x.vel": x_vel, "theta.vel": theta_vel})
            time.sleep(duration_s)
        finally:
            self.robot.stop_base()

    def go_forward(self, meters: float) -> None:
        distance = abs(float(meters))
        self._run_for_duration(x_vel=self.linear_speed_mps, duration_s=distance / self.linear_speed_mps)

    def go_backward(self, meters: float) -> None:
        distance = abs(float(meters))
        self._run_for_duration(x_vel=-self.linear_speed_mps, duration_s=distance / self.linear_speed_mps)

    def turn_left(self, degrees: float) -> None:
        angle = abs(float(degrees))
        self._run_for_duration(theta_vel=self.angular_speed_dps, duration_s=angle / self.angular_speed_dps)

    def turn_right(self, degrees: float) -> None:
        angle = abs(float(degrees))
        self._run_for_duration(theta_vel=-self.angular_speed_dps, duration_s=angle / self.angular_speed_dps)

    def disconnect(self) -> None:
        try:
            self.robot.stop_base()
        except Exception:
            pass

        try:
            if self.robot.is_connected:
                self.robot.disconnect()
        except DeviceNotConnectedError:
            pass


def build_robot() -> XLerobot2Wheels:
    robot_config = XLerobot2WheelsConfig(
        id=ROBOT_ID,
        port1=PORT1,
        port2=PORT2,
    )
    robot = XLerobot2Wheels(robot_config)
    robot.connect()
    return robot


def main() -> None:
    main_camera = RobotCamera(CAMERA_ID)
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

    agent = LLMAgent(
        model=MODEL_NAME,
        tools=tools,
        main_camera=main_camera,
        servo_controler=servo_controler,
    )
    agent.task = TASK
    agent.go()


if __name__ == "__main__":
    main()
