from __future__ import annotations

import json
import sys
import time
from datetime import datetime
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
TMP_IMAGES_DIR = SHIELD_DIR / "tmp_images"
DEBUG_BASE_DIR = TMP_IMAGES_DIR / "right_arm_extension_debug"
SETTLE_S = 0.8
MID_READ_DELAY_S = 0.25
MOVEMENT_THRESHOLD_DEG = 3.0
COMMAND_THRESHOLD_DEG = 10.0
OVERLOAD_LOAD_THRESHOLD = 900
OVERLOAD_CURRENT_THRESHOLD = 400

RIGHT_ARM_JOINT_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
FOCUS_JOINT_KEYS = (
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
)
REGISTER_DEBUG_MOTORS = (
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
)
REGISTER_DEBUG_FIELDS = (
    "Goal_Position",
    "Present_Position",
    "Present_Load",
    "Present_Current",
    "Status",
    "Torque_Enable",
)

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

PHASE_SPECS: list[dict[str, object]] = [
    {
        "name": "shoulder_lift_single_joint",
        "steps": [
            {"shoulder_lift.pos": -30.0},
            {"shoulder_lift.pos": -45.0},
            {"shoulder_lift.pos": -60.0},
            {"shoulder_lift.pos": -75.0},
            {"shoulder_lift.pos": -90.0},
            {"shoulder_lift.pos": -60.0},
            {"shoulder_lift.pos": -30.0},
        ],
    },
    {
        "name": "shoulder_lift_reverse_probe",
        "steps": [
            {"shoulder_lift.pos": 10.0},
            {"shoulder_lift.pos": 20.0},
            {"shoulder_lift.pos": 30.0},
            {"shoulder_lift.pos": 40.0},
            {"shoulder_lift.pos": 20.0},
            {"shoulder_lift.pos": 0.0},
        ],
    },
    {
        "name": "elbow_flex_single_joint",
        "steps": [
            {"elbow_flex.pos": 15.0},
            {"elbow_flex.pos": 30.0},
            {"elbow_flex.pos": 45.0},
            {"elbow_flex.pos": 60.0},
            {"elbow_flex.pos": 75.0},
            {"elbow_flex.pos": 45.0},
            {"elbow_flex.pos": 15.0},
        ],
    },
    {
        "name": "elbow_flex_reverse_probe",
        "steps": [
            {"elbow_flex.pos": -10.0},
            {"elbow_flex.pos": -20.0},
            {"elbow_flex.pos": -30.0},
            {"elbow_flex.pos": -40.0},
            {"elbow_flex.pos": -20.0},
            {"elbow_flex.pos": 0.0},
        ],
    },
    {
        "name": "wrist_flex_single_joint",
        "steps": [
            {"wrist_flex.pos": 35.0},
            {"wrist_flex.pos": 45.0},
            {"wrist_flex.pos": 55.0},
            {"wrist_flex.pos": 65.0},
            {"wrist_flex.pos": 75.0},
            {"wrist_flex.pos": 85.0},
            {"wrist_flex.pos": 55.0},
            {"wrist_flex.pos": 35.0},
        ],
    },
    {
        "name": "policy_like_composite",
        "steps": [
            {
                "shoulder_pan.pos": -7.4878,
                "shoulder_lift.pos": -60.4593,
                "elbow_flex.pos": 70.4154,
                "wrist_flex.pos": 29.1860,
                "wrist_roll.pos": -16.5156,
                "gripper.pos": 8.9562,
            },
            {
                "shoulder_pan.pos": -23.7729,
                "shoulder_lift.pos": -54.3075,
                "elbow_flex.pos": 20.9970,
                "wrist_flex.pos": 59.4885,
                "wrist_roll.pos": -16.3240,
                "gripper.pos": 3.3203,
            },
            {
                "shoulder_pan.pos": -26.6336,
                "shoulder_lift.pos": -42.8192,
                "elbow_flex.pos": 15.1755,
                "wrist_flex.pos": 54.3316,
                "wrist_roll.pos": -16.1086,
                "gripper.pos": 4.8484,
            },
        ],
    },
]


def _new_run_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def _write_json(path: Path, payload: dict) -> None:
    _atomic_write_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


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
    SO_FOLLOWER_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(SO_FOLLOWER_RIGHT_ARM_CALIBRATION_PATH, mapped_payload)

    return {
        "source_path": str(XLEROBOT_2WHEELS_CALIBRATION_PATH),
        "target_path": str(SO_FOLLOWER_RIGHT_ARM_CALIBRATION_PATH),
        "status": "overwritten",
        "summary": _format_calibration_summary(mapped_payload),
    }


def _normalize_positions(payload: dict | None) -> dict[str, float | None]:
    normalized = {joint: None for joint in RIGHT_ARM_JOINT_KEYS}
    if not isinstance(payload, dict):
        return normalized

    for joint in RIGHT_ARM_JOINT_KEYS:
        value = payload.get(joint)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            normalized[joint] = round(float(value), 4)

    return normalized


def _read_present_positions(robot) -> dict[str, float | None]:
    observation = robot.get_observation()
    return _normalize_positions(observation)


def _safe_read_raw_register(robot, motor: str, register: str) -> int | None | dict[str, str]:
    try:
        value = robot.bus.read(register, motor, normalize=False)
    except Exception as exc:
        return {"error": str(exc)}

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    return {"error": f"Unsupported value type: {type(value).__name__}"}


def _read_raw_register_snapshot(robot) -> dict[str, dict[str, int | None | dict[str, str]]]:
    snapshot: dict[str, dict[str, int | None | dict[str, str]]] = {}
    for motor in REGISTER_DEBUG_MOTORS:
        snapshot[motor] = {}
        for register in REGISTER_DEBUG_FIELDS:
            snapshot[motor][register] = _safe_read_raw_register(robot, motor, register)
    return snapshot


def _is_reverse_probe_phase(phase_name: str) -> bool:
    return phase_name.endswith("_reverse_probe")


def _snapshot_overload_reasons(
    snapshot: dict[str, dict[str, int | None | dict[str, str]]] | object,
) -> list[str]:
    if not isinstance(snapshot, dict):
        return []

    reasons: list[str] = []
    for motor in REGISTER_DEBUG_MOTORS:
        motor_payload = snapshot.get(motor)
        if not isinstance(motor_payload, dict):
            continue

        for register, raw_value in motor_payload.items():
            if isinstance(raw_value, dict):
                error_message = raw_value.get("error")
                if isinstance(error_message, str) and "overload error" in error_message.lower():
                    reasons.append(f"{motor}.{register}: {error_message}")

        present_load = motor_payload.get("Present_Load")
        if isinstance(present_load, int) and abs(present_load) >= OVERLOAD_LOAD_THRESHOLD:
            reasons.append(f"{motor}.Present_Load={present_load}")

        present_current = motor_payload.get("Present_Current")
        if isinstance(present_current, int) and present_current >= OVERLOAD_CURRENT_THRESHOLD:
            reasons.append(f"{motor}.Present_Current={present_current}")

    return reasons


def _summarize_overload(
    registers_mid: dict[str, dict[str, int | None | dict[str, str]]],
    registers_after: dict[str, dict[str, int | None | dict[str, str]]],
) -> dict[str, object]:
    mid_reasons = _snapshot_overload_reasons(registers_mid)
    if mid_reasons:
        return {
            "detected": True,
            "stage": "mid",
            "reason": mid_reasons[0],
            "reasons": mid_reasons,
        }

    after_reasons = _snapshot_overload_reasons(registers_after)
    if after_reasons:
        return {
            "detected": True,
            "stage": "after",
            "reason": after_reasons[0],
            "reasons": after_reasons,
        }

    return {"detected": False, "stage": None, "reason": None, "reasons": []}


def _compute_error_after(
    goal: dict[str, float],
    present_after: dict[str, float | None],
) -> dict[str, float | None]:
    error_payload: dict[str, float | None] = {}
    for joint, target in goal.items():
        present = present_after.get(joint)
        if present is None:
            error_payload[joint] = None
        else:
            error_payload[joint] = round(float(target) - float(present), 4)
    return error_payload


def _compute_movement(
    present_before: dict[str, float | None],
    present_after: dict[str, float | None],
) -> dict[str, float | None]:
    movement_payload: dict[str, float | None] = {}
    for joint in RIGHT_ARM_JOINT_KEYS:
        before = present_before.get(joint)
        after = present_after.get(joint)
        if before is None or after is None:
            movement_payload[joint] = None
        else:
            movement_payload[joint] = round(float(after) - float(before), 4)
    return movement_payload


def _joint_stats_for_steps(steps: list[dict[str, object]]) -> dict[str, dict[str, float | None]]:
    stats: dict[str, dict[str, float | None]] = {}
    for joint in RIGHT_ARM_JOINT_KEYS:
        errors: list[float] = []
        movements: list[float] = []
        for step in steps:
            error_after = step.get("error_after")
            if isinstance(error_after, dict):
                error_value = error_after.get(joint)
                if isinstance(error_value, (int, float)) and not isinstance(error_value, bool):
                    errors.append(abs(float(error_value)))

            movement = step.get("movement")
            if isinstance(movement, dict):
                movement_value = movement.get(joint)
                if isinstance(movement_value, (int, float)) and not isinstance(movement_value, bool):
                    movements.append(abs(float(movement_value)))

        stats[joint] = {
            "mean_abs_error_after": round(sum(errors) / len(errors), 4) if errors else None,
            "max_abs_error_after": round(max(errors), 4) if errors else None,
            "mean_abs_movement": round(sum(movements) / len(movements), 4) if movements else None,
            "max_abs_movement": round(max(movements), 4) if movements else None,
        }
    return stats


def _extract_raw_register_int(snapshot: object, motor: str, register: str) -> int | None:
    if not isinstance(snapshot, dict):
        return None
    motor_payload = snapshot.get(motor)
    if not isinstance(motor_payload, dict):
        return None

    value = motor_payload.get(register)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return None


def _build_raw_register_diagnosis(
    raw_events: list[dict[str, object]],
) -> dict[str, dict[str, int | str | None]]:
    diagnosis: dict[str, dict[str, int | str | None]] = {}

    for motor in REGISTER_DEBUG_MOTORS:
        commanded_steps = 0
        goal_position_changed_steps = 0
        present_position_changed_steps = 0
        load_active_steps = 0
        current_active_steps = 0

        for event in raw_events:
            before_snapshot = event.get("registers_before")
            after_snapshot = event.get("registers_after")
            goal_before = _extract_raw_register_int(before_snapshot, motor, "Goal_Position")
            goal_after = _extract_raw_register_int(after_snapshot, motor, "Goal_Position")
            present_before = _extract_raw_register_int(before_snapshot, motor, "Present_Position")
            present_after = _extract_raw_register_int(after_snapshot, motor, "Present_Position")
            present_load = _extract_raw_register_int(after_snapshot, motor, "Present_Load")
            present_current = _extract_raw_register_int(after_snapshot, motor, "Present_Current")

            if goal_before is None or goal_after is None or present_before is None or present_after is None:
                continue

            if goal_before == goal_after:
                continue

            commanded_steps += 1
            if abs(goal_after - goal_before) >= 8:
                goal_position_changed_steps += 1
            if abs(present_after - present_before) >= 8:
                present_position_changed_steps += 1
            if present_load is not None and abs(present_load) >= 50:
                load_active_steps += 1
            if present_current is not None and abs(present_current) >= 50:
                current_active_steps += 1

        likely_cause: str | None = None
        if commanded_steps > 0 and goal_position_changed_steps == 0:
            likely_cause = "goal_position_not_updating"
        elif goal_position_changed_steps > 0 and present_position_changed_steps == 0:
            likely_cause = "goal_updates_but_motor_does_not_move"
        elif commanded_steps > 0 and present_position_changed_steps > 0:
            likely_cause = "goal_and_present_both_move"

        diagnosis[motor] = {
            "commanded_steps": commanded_steps,
            "goal_position_changed_steps": goal_position_changed_steps,
            "present_position_changed_steps": present_position_changed_steps,
            "load_active_steps": load_active_steps,
            "current_active_steps": current_active_steps,
            "likely_cause": likely_cause,
        }

    return diagnosis


def _build_joint_diagnosis(phase_results: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    all_steps: list[dict[str, object]] = []
    phase3_steps: list[dict[str, object]] = []
    for phase_name, payload in phase_results.items():
        if _is_reverse_probe_phase(phase_name):
            continue
        steps = payload.get("steps", [])
        if isinstance(steps, list):
            all_steps.extend(steps)
            if phase_name == "policy_like_composite":
                phase3_steps = steps

    diagnosis: dict[str, dict[str, object]] = {}
    for joint in FOCUS_JOINT_KEYS:
        commanded_steps = 0
        blocked_steps = 0
        following_steps = 0
        for step in all_steps:
            goal = step.get("goal")
            present_before = step.get("present_before")
            movement = step.get("movement")
            error_after = step.get("error_after")

            if not isinstance(goal, dict) or joint not in goal:
                continue
            if not isinstance(present_before, dict) or not isinstance(movement, dict) or not isinstance(error_after, dict):
                continue

            before_value = present_before.get(joint)
            move_value = movement.get(joint)
            error_value = error_after.get(joint)
            goal_value = goal.get(joint)
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (before_value, goal_value)):
                continue

            if abs(float(goal_value) - float(before_value)) < COMMAND_THRESHOLD_DEG:
                continue

            commanded_steps += 1
            if isinstance(move_value, (int, float)) and abs(float(move_value)) < MOVEMENT_THRESHOLD_DEG:
                blocked_steps += 1
            if isinstance(error_value, (int, float)) and abs(float(error_value)) < MOVEMENT_THRESHOLD_DEG:
                following_steps += 1

        status = "inconclusive"
        if blocked_steps >= 2:
            status = "not_following_goal"
        elif commanded_steps > 0 and following_steps > commanded_steps / 2:
            status = "following_goal"

        diagnosis[joint] = {
            "status": status,
            "commanded_steps": commanded_steps,
            "blocked_steps": blocked_steps,
            "following_steps": following_steps,
        }

    elbow_payload = diagnosis.get("elbow_flex.pos", {})
    if elbow_payload.get("status") == "following_goal":
        for joint in FOCUS_JOINT_KEYS:
            if joint == "elbow_flex.pos":
                continue
            phase3_commanded_steps = 0
            phase3_blocked_steps = 0
            for step in phase3_steps:
                goal = step.get("goal")
                present_before = step.get("present_before")
                movement = step.get("movement")
                if not isinstance(goal, dict) or joint not in goal:
                    continue
                if not isinstance(present_before, dict) or not isinstance(movement, dict):
                    continue
                before_value = present_before.get(joint)
                goal_value = goal.get(joint)
                move_value = movement.get(joint)
                if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (before_value, goal_value)):
                    continue
                if abs(float(goal_value) - float(before_value)) < COMMAND_THRESHOLD_DEG:
                    continue
                phase3_commanded_steps += 1
                if isinstance(move_value, (int, float)) and abs(float(move_value)) < MOVEMENT_THRESHOLD_DEG:
                    phase3_blocked_steps += 1

            if phase3_commanded_steps > 0 and phase3_blocked_steps >= 1:
                diagnosis[joint]["phase3_marker"] = "selective_joint_follow_failure"

    return diagnosis


def _analyze_phase_motion(
    phase_results: dict[str, dict[str, object]],
    phase_name: str,
    joint: str,
) -> dict[str, int | bool]:
    payload = phase_results.get(phase_name, {})
    steps = payload.get("steps", [])
    if not isinstance(steps, list):
        steps = []

    commanded_steps = 0
    moving_steps = 0
    for step in steps:
        goal = step.get("goal")
        present_before = step.get("present_before")
        movement = step.get("movement")
        if not isinstance(goal, dict) or joint not in goal:
            continue
        if not isinstance(present_before, dict) or not isinstance(movement, dict):
            continue

        before_value = present_before.get(joint)
        goal_value = goal.get(joint)
        move_value = movement.get(joint)
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (before_value, goal_value)):
            continue

        if abs(float(goal_value) - float(before_value)) < COMMAND_THRESHOLD_DEG:
            continue

        commanded_steps += 1
        if isinstance(move_value, (int, float)) and abs(float(move_value)) >= MOVEMENT_THRESHOLD_DEG:
            moving_steps += 1

    return {
        "commanded_steps": commanded_steps,
        "moving_steps": moving_steps,
        "aborted_on_overload": bool(payload.get("aborted_on_overload")),
    }


def _build_direction_probe_diagnosis(
    phase_results: dict[str, dict[str, object]],
) -> dict[str, dict[str, int | bool | str]]:
    diagnosis: dict[str, dict[str, int | bool | str]] = {}
    phase_specs = (
        ("shoulder_lift", "shoulder_lift.pos", "shoulder_lift_single_joint", "shoulder_lift_reverse_probe"),
        ("elbow_flex", "elbow_flex.pos", "elbow_flex_single_joint", "elbow_flex_reverse_probe"),
    )

    for motor, joint, forward_phase, reverse_phase in phase_specs:
        forward = _analyze_phase_motion(phase_results, forward_phase, joint)
        reverse = _analyze_phase_motion(phase_results, reverse_phase, joint)
        forward_overloaded = bool(forward["aborted_on_overload"])
        reverse_overloaded = bool(reverse["aborted_on_overload"])
        reverse_commanded_steps = int(reverse["commanded_steps"])
        reverse_moving_steps = int(reverse["moving_steps"])

        status = "both_directions_blocked"
        if reverse_overloaded and reverse_moving_steps > 0:
            status = "usable_range_near_current_pose_is_narrow"
        elif reverse_overloaded:
            status = "reverse_direction_also_overloads"
        elif forward_overloaded and reverse_moving_steps >= max(2, reverse_commanded_steps // 2 or 1):
            status = "forward_direction_likely_wrong"
        elif reverse_moving_steps > 0:
            status = "usable_range_near_current_pose_is_narrow"

        diagnosis[motor] = {
            "status": status,
            "forward_overloaded": forward_overloaded,
            "reverse_overloaded": reverse_overloaded,
            "reverse_commanded_steps": reverse_commanded_steps,
            "reverse_moving_steps": reverse_moving_steps,
        }

    return diagnosis


def _run_step(
    robot,
    *,
    run_id: str,
    phase_name: str,
    step_index: int,
    goal: dict[str, float],
    events_path: Path,
    raw_register_events_path: Path,
) -> dict[str, object]:
    present_before = _read_present_positions(robot)
    registers_before = _read_raw_register_snapshot(robot)
    robot.send_action(goal)
    time.sleep(MID_READ_DELAY_S)
    present_mid = _read_present_positions(robot)
    registers_mid = _read_raw_register_snapshot(robot)
    time.sleep(max(0.0, SETTLE_S - MID_READ_DELAY_S))
    present_after = _read_present_positions(robot)
    registers_after = _read_raw_register_snapshot(robot)
    overload_summary = _summarize_overload(registers_mid, registers_after)

    event_payload: dict[str, object] = {
        "saved_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "run_id": run_id,
        "phase": phase_name,
        "step_index": step_index,
        "goal": {joint: round(float(value), 4) for joint, value in goal.items()},
        "present_before": present_before,
        "present_mid": present_mid,
        "present_after": present_after,
        "error_after": _compute_error_after(goal, present_after),
        "movement": _compute_movement(present_before, present_after),
        "settle_s": SETTLE_S,
        "overload_detected": overload_summary["detected"],
        "abort_stage": overload_summary["stage"],
        "abort_reason": overload_summary["reason"],
        "abort_reasons": overload_summary["reasons"],
    }
    _append_jsonl(events_path, event_payload)

    raw_register_payload: dict[str, object] = {
        "saved_at": event_payload["saved_at"],
        "run_id": run_id,
        "phase": phase_name,
        "step_index": step_index,
        "goal": event_payload["goal"],
        "registers_before": registers_before,
        "registers_mid": registers_mid,
        "registers_after": registers_after,
        "overload_detected": overload_summary["detected"],
        "abort_stage": overload_summary["stage"],
        "abort_reason": overload_summary["reason"],
        "abort_reasons": overload_summary["reasons"],
    }
    _append_jsonl(raw_register_events_path, raw_register_payload)
    return event_payload


def main() -> None:
    run_id = _new_run_id()
    run_dir = DEBUG_BASE_DIR / run_id
    events_path = run_dir / "command_events.jsonl"
    raw_register_events_path = run_dir / "raw_register_events.jsonl"
    summary_path = run_dir / "summary.json"
    calibration_sync_info = _sync_right_arm_calibration_for_so_follower()

    session_payload = {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "port": RIGHT_ARM_WHEEL_USB,
        "calibration_source": calibration_sync_info["source_path"],
        "calibration_target": calibration_sync_info["target_path"],
        "calibration_sync": calibration_sync_info["status"],
        "calibration_summary": calibration_sync_info["summary"],
        "command_events_path": str(events_path.resolve()),
        "raw_register_events_path": str(raw_register_events_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "phases": [
            {"name": phase["name"], "step_count": len(phase["steps"])}  # type: ignore[index]
            for phase in PHASE_SPECS
        ],
    }
    _write_json(run_dir / "session.json", session_payload)

    robot_config = SOFollowerConfig(port=RIGHT_ARM_WHEEL_USB, cameras={})
    robot_config.type = "so101_follower"
    robot_config.id = "right_arm"
    robot_config.calibration_dir = SO_FOLLOWER_CALIBRATION_DIR

    print(f"[Debug] Run directory: {run_dir}")
    print(f"[Debug] Port: {RIGHT_ARM_WHEEL_USB}")
    print(f"[Debug] Calibration source: {calibration_sync_info['source_path']}")
    print(f"[Debug] Calibration target: {calibration_sync_info['target_path']}")
    print(f"[Debug] Calibration summary: {calibration_sync_info['summary']}")

    robot = make_robot_from_config(robot_config)
    phase_results: dict[str, dict[str, object]] = {}
    initial_present_position: dict[str, float | None] | None = None
    final_present_position: dict[str, float | None] | None = None

    try:
        robot.connect(calibrate=False)
        initial_present_position = _read_present_positions(robot)
        print("[Debug] Initial present position:")
        for joint in RIGHT_ARM_JOINT_KEYS:
            print(f"  {joint}: {initial_present_position[joint]}")

        for phase_spec in PHASE_SPECS:
            phase_name = str(phase_spec["name"])
            steps = phase_spec["steps"]
            print(f"[Debug] Running phase: {phase_name}")
            phase_events: list[dict[str, object]] = []
            phase_status = "completed"
            abort_reason = None
            for step_index, goal in enumerate(steps):
                event_payload = _run_step(
                    robot,
                    run_id=run_id,
                    phase_name=phase_name,
                    step_index=step_index,
                    goal=goal,  # type: ignore[arg-type]
                    events_path=events_path,
                    raw_register_events_path=raw_register_events_path,
                )
                phase_events.append(event_payload)
                print(
                    f"  Step {step_index}: goal={event_payload['goal']} "
                    f"present_after={event_payload['present_after']}"
                )
                if event_payload.get("overload_detected"):
                    phase_status = "aborted_on_overload"
                    abort_reason = event_payload.get("abort_reason")
                    print(
                        f"  [Safety] Aborting phase '{phase_name}' after step {step_index} "
                        f"due to overload: {abort_reason}"
                    )
                    break

            phase_results[phase_name] = {
                "step_count": len(phase_events),
                "status": phase_status,
                "aborted_on_overload": phase_status == "aborted_on_overload",
                "abort_reason": abort_reason,
                "targeted_joints": sorted({joint for event in phase_events for joint in event["goal"]}),  # type: ignore[index]
                "joint_stats": _joint_stats_for_steps(phase_events),
                "steps": phase_events,
            }

        final_present_position = _read_present_positions(robot)
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass

    summary_payload = {
        "run_id": run_id,
        "initial_present_position": initial_present_position,
        "final_present_position": final_present_position,
        "phase_results": phase_results,
        "joint_diagnosis": _build_joint_diagnosis(phase_results),
        "direction_probe_diagnosis": _build_direction_probe_diagnosis(phase_results),
        "raw_register_diagnosis": _build_raw_register_diagnosis(
            [
                json.loads(line)
                for line in raw_register_events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        ),
    }
    _write_json(summary_path, summary_payload)

    print(f"[Debug] Command events: {events_path}")
    print(f"[Debug] Raw register events: {raw_register_events_path}")
    print(f"[Debug] Summary: {summary_path}")
    print("[Debug] Joint diagnosis:")
    for joint, payload in summary_payload["joint_diagnosis"].items():
        print(f"  {joint}: {payload}")
    print("[Debug] Direction probe diagnosis:")
    for motor, payload in summary_payload["direction_probe_diagnosis"].items():
        print(f"  {motor}: {payload}")
    print("[Debug] Raw register diagnosis:")
    for motor, payload in summary_payload["raw_register_diagnosis"].items():
        print(f"  {motor}: {payload}")


if __name__ == "__main__":
    main()
