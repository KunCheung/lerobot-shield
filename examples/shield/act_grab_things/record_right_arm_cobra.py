from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SHIELD_DIR = SCRIPT_DIR.parent
REPO_ROOT = SHIELD_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS


RIGHT_ARM_WHEEL_USB = "COM4"
DEFAULT_POSITION_NAME = "cobra"
POSITION_DIR = Path("~/.cache/robocrew/positions/").expanduser()
VERIFY_RECALL_SETTLE_S = 1.0

RIGHT_ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
RIGHT_ARM_JOINT_KEYS = tuple(f"{joint}.pos" for joint in RIGHT_ARM_JOINTS)

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


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


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
        f"{joint}={values['range_min']}..{values['range_max']}" for joint, values in mapped_payload.items()
    )


def _sync_right_arm_calibration_for_so_follower() -> dict[str, str]:
    if not XLEROBOT_2WHEELS_CALIBRATION_PATH.is_file():
        raise RuntimeError(f"Missing xlerobot_2wheels calibration file: {XLEROBOT_2WHEELS_CALIBRATION_PATH}")

    try:
        source_payload = json.loads(XLEROBOT_2WHEELS_CALIBRATION_PATH.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read xlerobot_2wheels calibration file: {XLEROBOT_2WHEELS_CALIBRATION_PATH}"
        ) from exc

    mapped_payload = _extract_right_arm_so_follower_calibration(source_payload)
    _atomic_write_json(SO_FOLLOWER_RIGHT_ARM_CALIBRATION_PATH, mapped_payload)

    return {
        "source_path": str(XLEROBOT_2WHEELS_CALIBRATION_PATH),
        "target_path": str(SO_FOLLOWER_RIGHT_ARM_CALIBRATION_PATH),
        "summary": _format_calibration_summary(mapped_payload),
    }


def _make_right_arm_robot(port: str, *, disable_torque_on_disconnect: bool):
    robot_config = SOFollowerConfig(port=port, cameras={})
    robot_config.type = "so101_follower"
    robot_config.id = "right_arm"
    robot_config.calibration_dir = SO_FOLLOWER_CALIBRATION_DIR
    robot_config.disable_torque_on_disconnect = disable_torque_on_disconnect
    return make_robot_from_config(robot_config)


def _position_path(position_name: str) -> Path:
    file_name = position_name if position_name.endswith(".json") else f"{position_name}.json"
    return POSITION_DIR / file_name


def _read_right_arm_positions(robot) -> dict[str, float]:
    observation = robot.get_observation()
    positions: dict[str, float] = {}
    missing_keys: list[str] = []

    for joint, key in zip(RIGHT_ARM_JOINTS, RIGHT_ARM_JOINT_KEYS, strict=True):
        value = observation.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            missing_keys.append(key)
            continue
        positions[joint] = round(float(value), 4)

    if missing_keys:
        raise RuntimeError("Missing right-arm observation keys: " + ", ".join(missing_keys))

    return positions


def _save_right_arm_pose(position_name: str, positions: dict[str, float]) -> Path:
    missing_joints = [joint for joint in RIGHT_ARM_JOINTS if joint not in positions]
    if missing_joints:
        raise RuntimeError("Cannot save pose; missing joints: " + ", ".join(missing_joints))

    payload = {
        "arm_side": "right",
        "positions": {joint: positions[joint] for joint in RIGHT_ARM_JOINTS},
    }
    save_path = _position_path(position_name)
    _atomic_write_json(save_path, payload)
    return save_path


def _print_positions(label: str, positions: dict[str, float]) -> None:
    print(label)
    for joint in RIGHT_ARM_JOINTS:
        print(f"  {joint}: {positions[joint]}")


def _verify_recall(robot, positions: dict[str, float]) -> None:
    action = {f"{joint}.pos": value for joint, value in positions.items()}
    print("[Verify] Enabling torque and sending the saved right-arm pose once...")
    robot.bus.enable_torque()
    robot.send_action(action)
    time.sleep(VERIFY_RECALL_SETTLE_S)
    present = _read_right_arm_positions(robot)
    _print_positions("[Verify] Present position after recall:", present)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record the right-arm cobra start pose for the two-wheel notebook grab example."
    )
    parser.add_argument(
        "--port",
        default=RIGHT_ARM_WHEEL_USB,
        help=f"Right-arm serial port. Defaults to {RIGHT_ARM_WHEEL_USB}.",
    )
    parser.add_argument(
        "--position-name",
        default=DEFAULT_POSITION_NAME,
        help=f"Saved pose name under {POSITION_DIR}. Defaults to {DEFAULT_POSITION_NAME}.",
    )
    parser.add_argument(
        "--keep-torque",
        action="store_true",
        help="Do not release right-arm torque before reading the current pose.",
    )
    parser.add_argument(
        "--verify-recall",
        action="store_true",
        help="After saving, send the saved pose once to verify recall. This may move the arm.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    calibration_sync_info = _sync_right_arm_calibration_for_so_follower()

    print(f"[Record] Right-arm port: {args.port}")
    print(f"[Record] Calibration source: {calibration_sync_info['source_path']}")
    print(f"[Record] Calibration target: {calibration_sync_info['target_path']}")
    print(f"[Record] Calibration summary: {calibration_sync_info['summary']}")

    robot = _make_right_arm_robot(args.port, disable_torque_on_disconnect=True)
    try:
        robot.connect(calibrate=False)
        initial_positions = _read_right_arm_positions(robot)
        _print_positions("[Record] Current right-arm position:", initial_positions)

        if args.keep_torque:
            print("[Record] Keeping torque enabled before capture.")
        else:
            print("[Record] Releasing right-arm torque for manual positioning.")
            robot.bus.disable_torque()
            input("Move the right arm to the cobra pose, then press Enter to save... ")

        captured_positions = _read_right_arm_positions(robot)
        _print_positions("[Record] Captured cobra pose:", captured_positions)
        save_path = _save_right_arm_pose(args.position_name, captured_positions)
        print(f"Saved right-arm pose: {save_path}")

        if args.verify_recall:
            _verify_recall(robot, captured_positions)
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
