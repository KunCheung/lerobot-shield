from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import common


WINDOW_NAME = "SmolVLA PickPlace Live"
DEFAULT_LIVE_DEBUG_ROOT = common.SCRIPT_DIR / "debug_live"
DEFAULT_MAX_STEPS = 500
DEFAULT_MAX_JOINT_DELTA = 3.0
DEFAULT_MAX_GRIPPER_DELTA = 3.0
DEFAULT_MAX_RAW_JOINT_DELTA = 80.0
DEFAULT_MAX_RAW_GRIPPER_DELTA = 40.0
DEFAULT_SETTLE_S = 0.20
DEFAULT_OPEN_GRIPPER_VALUE = 0.0
DEFAULT_CLOSE_GRIPPER_VALUE = 30.0
DEFAULT_OPEN_UNTIL_STEP = 40
DEFAULT_FORCE_CLOSE_AFTER_STEP = 60
DEFAULT_CLOSE_DWELL_STEPS = 30
GRIPPER_KEY = "right_arm_gripper.pos"


@dataclass(frozen=True)
class ClampResult:
    clamped_action: dict[str, float]
    clamped_delta: dict[str, float]
    was_clamped: bool
    clamped_joints: list[str]


@dataclass(frozen=True)
class GripperAssistDecision:
    phase: str
    policy_gripper: float
    assisted_gripper: float
    override_policy: bool


def parse_positive_float(raw_value: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a number, got '{raw_value}'.") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("Expected a positive number.")
    return value


def parse_non_negative_float(raw_value: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a number, got '{raw_value}'.") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative number.")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Conservative live runner for the SmolVLA SO-101 pick-and-place checkpoint. "
            "This script can move the right arm only after --confirm-live is passed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default=common.DEFAULT_MODEL_ID, help="SmolVLA model repo id or local path.")
    parser.add_argument("--task", default=common.DEFAULT_TASK, help="Natural-language task passed to the policy.")
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        metavar="CAMERA=INDEX",
        help=(
            "Override physical camera sources. Accepts camera1/camera2 or "
            "observation.images.camera1/camera2. "
            f"If omitted, defaults are used: {common.DEFAULT_CAMERA_MAPPING_TEXT}."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=common.parse_positive_int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum live steps to send. Keep this small for early hardware tests.",
    )
    parser.add_argument(
        "--max-joint-delta",
        type=parse_positive_float,
        default=DEFAULT_MAX_JOINT_DELTA,
        help="Maximum absolute target change per non-gripper joint for each sent command.",
    )
    parser.add_argument(
        "--max-gripper-delta",
        type=parse_positive_float,
        default=DEFAULT_MAX_GRIPPER_DELTA,
        help="Maximum absolute target change for the gripper for each sent command.",
    )
    parser.add_argument(
        "--settle-s",
        type=parse_non_negative_float,
        default=DEFAULT_SETTLE_S,
        help="Sleep after each sent command to let the arm settle.",
    )
    parser.add_argument("--no-window", action="store_true", help="Do not open the OpenCV preview window.")
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required safety acknowledgement. Without this flag the script refuses to move the robot.",
    )
    parser.add_argument(
        "--step-confirm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require pressing ENTER before each clamped command is sent.",
    )
    parser.add_argument(
        "--use-cached-chunk",
        action="store_true",
        help=(
            "Use SmolVLA's cached action chunk instead of resetting before every step. "
            "Leave off for the first hardware tests."
        ),
    )
    return parser


def apply_fixed_defaults(args: argparse.Namespace) -> argparse.Namespace:
    args.device = None
    args.robot_id = common.DEFAULT_ROBOT_ID
    args.port1 = common.DEFAULT_PORT1
    args.port2 = common.DEFAULT_PORT2
    args.max_raw_joint_delta = DEFAULT_MAX_RAW_JOINT_DELTA
    args.max_raw_gripper_delta = DEFAULT_MAX_RAW_GRIPPER_DELTA
    args.debug_root = DEFAULT_LIVE_DEBUG_ROOT
    args.debug_write_every = common.DEFAULT_DEBUG_WRITE_EVERY
    args.terminal_debug_every = common.DEFAULT_TERMINAL_DEBUG_EVERY
    args.local_files_only = False
    args.cache_dir = None
    args.gripper_assist = False
    args.open_gripper_value = DEFAULT_OPEN_GRIPPER_VALUE
    args.close_gripper_value = DEFAULT_CLOSE_GRIPPER_VALUE
    args.open_until_step = DEFAULT_OPEN_UNTIL_STEP
    args.force_close_after_step = DEFAULT_FORCE_CLOSE_AFTER_STEP
    args.close_dwell_steps = DEFAULT_CLOSE_DWELL_STEPS
    return args


def parse_args() -> argparse.Namespace:
    return apply_fixed_defaults(build_arg_parser().parse_args())


def ensure_live_confirmed(args: argparse.Namespace) -> None:
    if args.confirm_live:
        return
    raise RuntimeError(
        "Refusing to move the robot without --confirm-live. "
        "For the first hardware test, run with --confirm-live --max-steps 1 --step-confirm."
    )


def ensure_finite_values(label: str, values: dict[str, float]) -> None:
    bad_values = {name: value for name, value in values.items() if not math.isfinite(value)}
    if bad_values:
        raise RuntimeError(f"{label} contains non-finite values: {bad_values}")


def validate_raw_delta(args: argparse.Namespace, predicted_delta: dict[str, float]) -> None:
    too_large: dict[str, dict[str, float]] = {}
    for name, delta in predicted_delta.items():
        limit = args.max_raw_gripper_delta if name == GRIPPER_KEY else args.max_raw_joint_delta
        if abs(delta) > limit:
            too_large[name] = {"delta": delta, "limit": limit}
    if too_large:
        raise RuntimeError(
            "Raw policy delta is beyond the live safety limit; refusing to send any command. "
            f"Details: {too_large}"
        )


def clamp_policy_action(
    args: argparse.Namespace,
    joint_snapshot: common.JointDebugSnapshot,
) -> ClampResult:
    ensure_finite_values("current_state", joint_snapshot.current_state)
    ensure_finite_values("predicted_action", joint_snapshot.predicted_action)
    ensure_finite_values("predicted_delta", joint_snapshot.predicted_delta)
    validate_raw_delta(args, joint_snapshot.predicted_delta)

    clamped_action: dict[str, float] = {}
    clamped_delta: dict[str, float] = {}
    clamped_joints: list[str] = []
    for name in common.RIGHT_ARM_6D_STATE_NAMES:
        current = joint_snapshot.current_state[name]
        raw_delta = joint_snapshot.predicted_delta[name]
        limit = args.max_gripper_delta if name == GRIPPER_KEY else args.max_joint_delta
        safe_delta = min(max(raw_delta, -limit), limit)
        clamped_delta[name] = float(safe_delta)
        clamped_action[name] = float(current + safe_delta)
        if abs(safe_delta - raw_delta) > 1e-6:
            clamped_joints.append(name)

    return ClampResult(
        clamped_action=clamped_action,
        clamped_delta=clamped_delta,
        was_clamped=bool(clamped_joints),
        clamped_joints=clamped_joints,
    )


def validate_gripper_args(args: argparse.Namespace) -> None:
    if args.gripper_assist and args.force_close_after_step <= args.open_until_step:
        raise RuntimeError(
            "--force-close-after-step must be greater than --open-until-step so the arm has time to approach "
            "before the forced close phase."
        )


def apply_gripper_assist(
    args: argparse.Namespace,
    step_index: int,
    clamp_result: ClampResult,
) -> tuple[ClampResult, GripperAssistDecision]:
    policy_gripper = float(clamp_result.clamped_action[GRIPPER_KEY])
    if not args.gripper_assist:
        return clamp_result, GripperAssistDecision(
            phase="policy",
            policy_gripper=policy_gripper,
            assisted_gripper=policy_gripper,
            override_policy=False,
        )

    if step_index <= args.open_until_step:
        phase = "forced-open"
        assisted_gripper = float(args.open_gripper_value)
    elif step_index >= args.force_close_after_step:
        dwell_until = args.force_close_after_step + args.close_dwell_steps - 1
        phase = "forced-close-dwell" if step_index <= dwell_until else "forced-close-keep"
        assisted_gripper = float(args.close_gripper_value)
    else:
        phase = "policy-between-open-close"
        assisted_gripper = policy_gripper

    override_policy = abs(assisted_gripper - policy_gripper) > 1e-6
    if not override_policy:
        return clamp_result, GripperAssistDecision(
            phase=phase,
            policy_gripper=policy_gripper,
            assisted_gripper=assisted_gripper,
            override_policy=False,
        )

    assisted_action = dict(clamp_result.clamped_action)
    assisted_delta = dict(clamp_result.clamped_delta)
    assisted_action[GRIPPER_KEY] = assisted_gripper
    assisted_delta[GRIPPER_KEY] = assisted_gripper - (
        clamp_result.clamped_action[GRIPPER_KEY] - clamp_result.clamped_delta[GRIPPER_KEY]
    )
    assisted_joints = list(clamp_result.clamped_joints)
    if GRIPPER_KEY not in assisted_joints:
        assisted_joints.append(GRIPPER_KEY)

    return ClampResult(
        clamped_action=assisted_action,
        clamped_delta=assisted_delta,
        was_clamped=True,
        clamped_joints=assisted_joints,
    ), GripperAssistDecision(
        phase=phase,
        policy_gripper=policy_gripper,
        assisted_gripper=assisted_gripper,
        override_policy=True,
    )


def write_live_session_config(
    run_dir: Path,
    args: argparse.Namespace,
    config: common.PreTrainedConfig,
    camera_assignments: list[common.CameraAssignment],
    ignored_checkpoint_cameras: set[str],
) -> None:
    common.write_json(
        run_dir / "session.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "live",
            "model_id": args.model_id,
            "task": args.task,
            "device": str(config.device),
            "robot_id": args.robot_id,
            "port1": args.port1,
            "port2": args.port2,
            "state_names": list(common.RIGHT_ARM_6D_STATE_NAMES),
            "action_names": list(common.RIGHT_ARM_6D_STATE_NAMES),
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
            "safety": {
                "confirm_live": args.confirm_live,
                "step_confirm": args.step_confirm,
                "use_cached_chunk": args.use_cached_chunk,
                "max_joint_delta": args.max_joint_delta,
                "max_gripper_delta": args.max_gripper_delta,
                "max_raw_joint_delta": args.max_raw_joint_delta,
                "max_raw_gripper_delta": args.max_raw_gripper_delta,
                "settle_s": args.settle_s,
                "gripper_assist": args.gripper_assist,
                "open_gripper_value": args.open_gripper_value,
                "close_gripper_value": args.close_gripper_value,
                "open_until_step": args.open_until_step,
                "force_close_after_step": args.force_close_after_step,
                "close_dwell_steps": args.close_dwell_steps,
            },
        },
    )


def print_live_startup_summary(
    args: argparse.Namespace,
    config: common.PreTrainedConfig,
    run_dir: Path,
    camera_assignments: list[common.CameraAssignment],
    ignored_checkpoint_cameras: set[str],
) -> None:
    print(f"[Model] model_id: {args.model_id}")
    print(f"[Model] type: {config.type}")
    print(f"[Model] device: {config.device}")
    print("[Model] layout: right_arm_6d state_dim=6 action_dim=6")
    print(f"[Task] {args.task}")
    print(f"[Robot] id={args.robot_id} port1={args.port1} port2={args.port2}")
    print(f"[Camera] active mapping: {common.format_active_camera_mapping(camera_assignments)}")
    if ignored_checkpoint_cameras:
        print(f"[Camera] ignored checkpoint cameras: {', '.join(sorted(ignored_checkpoint_cameras))}")
    print(f"[Debug] run_dir: {run_dir}")
    print("[Mode] LIVE: this script sends clamped right-arm commands after confirmation.")
    print(
        "[Safety] "
        f"max_joint_delta={args.max_joint_delta} "
        f"max_gripper_delta={args.max_gripper_delta} "
        f"step_confirm={args.step_confirm} "
        f"use_cached_chunk={args.use_cached_chunk}"
    )
    if args.gripper_assist:
        print(
            "[GripperAssist] "
            f"open={args.open_gripper_value} until_step={args.open_until_step}; "
            f"close={args.close_gripper_value} from_step={args.force_close_after_step}; "
            f"dwell_steps={args.close_dwell_steps}"
        )
    if not args.use_cached_chunk:
        print("[Safety] Policy queue is reset before every live step.")
    if args.no_window:
        print("[Mode] Preview window disabled (--no-window).")
    if str(config.device) == "cpu":
        print("[Hint] CPU inference may take tens of seconds for each fresh action chunk.")


def build_live_runtime(args: argparse.Namespace) -> common.RuntimeArtifacts:
    run_dir = common.make_run_dir(args.debug_root)
    config = common.build_policy_config(args)
    raw_camera_assignments = args.camera if args.camera is not None else []
    camera_assignments, ignored_checkpoint_cameras = common.parse_camera_assignments(
        raw_camera_assignments,
        config.image_features,
    )
    write_live_session_config(run_dir, args, config, camera_assignments, ignored_checkpoint_cameras)
    print_live_startup_summary(args, config, run_dir, camera_assignments, ignored_checkpoint_cameras)

    policy, preprocessor, postprocessor = common.load_policy_artifacts(args, config)
    robot = common.build_robot_instance(args)
    cameras = [common.build_camera(assignment) for assignment in camera_assignments]
    initial_frames_rgb = common.read_camera_frames(cameras)
    dataset_features = common.build_dataset_features(cameras, initial_frames_rgb)

    return common.RuntimeArtifacts(
        args=args,
        run_dir=run_dir,
        device=common.torch.device(policy.config.device),
        camera_assignments=camera_assignments,
        dataset_features=dataset_features,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot=robot,
        cameras=cameras,
        ignored_checkpoint_cameras=ignored_checkpoint_cameras,
    )


def format_clamped_joints(clamp_result: ClampResult) -> str:
    if not clamp_result.clamped_joints:
        return "none"
    return ",".join(common.JOINT_SHORT_LABELS[name] for name in clamp_result.clamped_joints)


def build_live_debug_text(
    *,
    step_index: int,
    args: argparse.Namespace,
    runtime: common.RuntimeArtifacts,
    observation: common.RuntimeObservation,
    action_step: common.ActionSelection,
    joint_snapshot: common.JointDebugSnapshot,
    clamp_result: ClampResult,
    gripper_assist: GripperAssistDecision,
    camera_runtime_status: str,
    send_status: str,
    sent_action: dict[str, float] | None,
) -> str:
    lines = [
        f"step={step_index}",
        "mode=live",
        f"send_status={send_status}",
        f"model_id={args.model_id}",
        f"task={args.task}",
        "layout=right_arm_6d",
        f"camera_mapping={common.format_active_camera_mapping(runtime.camera_assignments)}",
        f"ignored_checkpoint_cameras={','.join(sorted(runtime.ignored_checkpoint_cameras)) or 'none'}",
        f"camera_frames={camera_runtime_status}",
        f"queue_status={common.format_queue_status(action_step.queue_depth_before, action_step.queue_depth_after)}",
        f"observation_ms={observation.observation_ms:.1f}",
        f"camera_ms={observation.camera_ms:.1f}",
        f"preprocess_ms={action_step.preprocess_ms:.1f}",
        f"model_ms={action_step.model_ms:.1f}",
        f"postprocess_ms={action_step.postprocess_ms:.1f}",
        f"action_vector={common.format_vector(action_step.action_vector)}",
        f"was_clamped={clamp_result.was_clamped}",
        f"clamped_joints={format_clamped_joints(clamp_result)}",
        f"gripper_assist_phase={gripper_assist.phase}",
        f"gripper_assist_override={gripper_assist.override_policy}",
        f"policy_gripper={gripper_assist.policy_gripper:.3f}",
        f"assisted_gripper={gripper_assist.assisted_gripper:.3f}",
        common.compact_joint_line("current", joint_snapshot.current_state),
        common.compact_joint_line("predicted", joint_snapshot.predicted_action),
        common.compact_joint_line("raw_delta", joint_snapshot.predicted_delta),
        common.compact_joint_line("clamped", clamp_result.clamped_action),
        common.compact_joint_line("clamped_delta", clamp_result.clamped_delta),
    ]
    if sent_action is not None:
        lines.append(common.compact_joint_line("sent", sent_action))
    return "\n".join(lines) + "\n"


def build_live_status_lines(
    *,
    args: argparse.Namespace,
    runtime: common.RuntimeArtifacts,
    observation: common.RuntimeObservation,
    action_step: common.ActionSelection,
    joint_snapshot: common.JointDebugSnapshot,
    clamp_result: ClampResult,
    gripper_assist: GripperAssistDecision,
    camera_runtime_status: str,
    send_status: str,
) -> list[str]:
    return [
        "LIVE right-arm only",
        f"send: {send_status}",
        f"task: {args.task}",
        f"cameras: {common.format_active_camera_mapping(runtime.camera_assignments)}",
        f"camera_frames: {camera_runtime_status}",
        (
            f"{common.format_queue_status(action_step.queue_depth_before, action_step.queue_depth_after)} "
            f"obs={observation.observation_ms:.1f}ms cam={observation.camera_ms:.1f}ms"
        ),
        (
            f"prep={action_step.preprocess_ms:.1f}ms "
            f"model={action_step.model_ms:.1f}ms "
            f"post={action_step.postprocess_ms:.1f}ms"
        ),
        f"clamped={clamp_result.was_clamped} joints={format_clamped_joints(clamp_result)}",
        (
            f"grip: {gripper_assist.phase} "
            f"policy={gripper_assist.policy_gripper:.1f} assist={gripper_assist.assisted_gripper:.1f}"
        ),
        common.compact_joint_line("cur", joint_snapshot.current_state),
        common.compact_joint_line("raw", joint_snapshot.predicted_delta),
        common.compact_joint_line("cmd", clamp_result.clamped_delta),
        "ENTER sends, q cancels" if args.step_confirm else "q or ESC in window quits",
    ]


def make_live_action_payload(
    *,
    send_status: str,
    action_step: common.ActionSelection,
    joint_snapshot: common.JointDebugSnapshot,
    clamp_result: ClampResult,
    gripper_assist: GripperAssistDecision,
    sent_action: dict[str, float] | None,
    observation: common.RuntimeObservation,
    camera_runtime_status: str,
) -> dict[str, Any]:
    return {
        "saved_at": datetime.now().isoformat(timespec="milliseconds"),
        "mode": "live",
        "send_status": send_status,
        "action_vector": action_step.action_vector,
        "current_state": joint_snapshot.current_state,
        "predicted_action": joint_snapshot.predicted_action,
        "predicted_delta": joint_snapshot.predicted_delta,
        "clamped_action": clamp_result.clamped_action,
        "clamped_delta": clamp_result.clamped_delta,
        "sent_action": sent_action,
        "was_clamped": clamp_result.was_clamped,
        "clamped_joints": clamp_result.clamped_joints,
        "policy_gripper": gripper_assist.policy_gripper,
        "assisted_gripper": gripper_assist.assisted_gripper,
        "gripper_assist_phase": gripper_assist.phase,
        "gripper_assist_override": gripper_assist.override_policy,
        "camera_frames": camera_runtime_status,
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


def write_live_runtime_debug(
    run_dir: Path,
    *,
    step_index: int,
    preview_frame_bgr: Any,
    debug_text: str,
    action_payload: dict[str, Any],
    append_history: bool,
) -> None:
    common.cv2.imwrite(str(run_dir / "latest_preview.jpg"), preview_frame_bgr)
    (run_dir / "latest_runtime.txt").write_text(debug_text, encoding="utf-8")
    common.write_json(run_dir / "latest_action.json", action_payload)
    if append_history:
        common.append_jsonl(run_dir / "actions.jsonl", {"step": step_index, **action_payload})


def prompt_before_send(step_index: int, clamp_result: ClampResult) -> bool:
    print(
        f"[Confirm] Step {step_index}: ENTER sends the clamped right-arm command "
        f"(clamped={clamp_result.was_clamped}, joints={format_clamped_joints(clamp_result)}). "
        "Type q then ENTER to cancel."
    )
    answer = input("> ").strip().lower()
    return answer not in {"q", "quit", "n", "no", "cancel", "stop"}


def show_preview(preview_frame: Any) -> bool:
    common.cv2.imshow(WINDOW_NAME, preview_frame)
    key = common.cv2.waitKey(1) & 0xFF
    return key not in {ord("q"), 27}


def maybe_reset_policy_for_live_step(args: argparse.Namespace, policy: Any) -> None:
    if not args.use_cached_chunk:
        policy.reset()


def main() -> None:
    args = parse_args()
    ensure_live_confirmed(args)
    validate_gripper_args(args)
    common.load_runtime_dependencies()

    runtime: common.RuntimeArtifacts | None = None
    robot: common.XLerobot2Wheels | None = None
    cameras: list[common.ValidatedLocalCamera] = []

    try:
        runtime = build_live_runtime(args)
        robot = runtime.robot
        cameras = runtime.cameras

        print("[Robot] Connecting for LIVE right-arm run...")
        robot.connect()

        action_names = runtime.dataset_features[common.ACTION]["names"]
        for step_index in range(1, args.max_steps + 1):
            maybe_reset_policy_for_live_step(args, runtime.policy)

            observation = common.collect_runtime_observation(robot, cameras)
            inference_frame = common.build_inference_frame(
                observation=observation.raw_observation,
                ds_features=runtime.dataset_features,
                device=runtime.device,
                task=args.task,
            )
            action_step = common.select_policy_action(
                runtime.policy,
                runtime.preprocessor,
                runtime.postprocessor,
                inference_frame,
                runtime.dataset_features,
            )
            if len(action_step.action_vector) != len(action_names):
                raise RuntimeError(
                    f"Action mismatch: tensor has {len(action_step.action_vector)} dims "
                    f"but action names has {len(action_names)}."
                )

            joint_snapshot = common.build_joint_debug_snapshot(
                observation.raw_observation,
                action_step.named_action,
            )
            clamp_result = clamp_policy_action(args, joint_snapshot)
            clamp_result, gripper_assist = apply_gripper_assist(args, step_index, clamp_result)
            camera_runtime_status = common.format_camera_runtime_status(cameras)
            send_status = "pending-confirmation" if args.step_confirm else "pending-send"
            sent_action: dict[str, float] | None = None

            status_lines = build_live_status_lines(
                args=args,
                runtime=runtime,
                observation=observation,
                action_step=action_step,
                joint_snapshot=joint_snapshot,
                clamp_result=clamp_result,
                gripper_assist=gripper_assist,
                camera_runtime_status=camera_runtime_status,
                send_status=send_status,
            )
            preview_frame = common.build_preview_frame(observation.frames_rgb, status_lines)
            debug_text = build_live_debug_text(
                step_index=step_index,
                args=args,
                runtime=runtime,
                observation=observation,
                action_step=action_step,
                joint_snapshot=joint_snapshot,
                clamp_result=clamp_result,
                gripper_assist=gripper_assist,
                camera_runtime_status=camera_runtime_status,
                send_status=send_status,
                sent_action=sent_action,
            )
            action_payload = make_live_action_payload(
                send_status=send_status,
                action_step=action_step,
                joint_snapshot=joint_snapshot,
                clamp_result=clamp_result,
                gripper_assist=gripper_assist,
                sent_action=sent_action,
                observation=observation,
                camera_runtime_status=camera_runtime_status,
            )

            if step_index == 1 or step_index % args.terminal_debug_every == 0:
                print(debug_text.rstrip())

            if step_index == 1 or step_index % args.debug_write_every == 0:
                write_live_runtime_debug(
                    runtime.run_dir,
                    step_index=step_index,
                    preview_frame_bgr=preview_frame,
                    debug_text=debug_text,
                    action_payload=action_payload,
                    append_history=False,
                )

            if not args.no_window and not show_preview(preview_frame):
                send_status = "canceled-window"
                print("[Exit] User requested stop from preview window before sending.")
                break

            if args.step_confirm and not prompt_before_send(step_index, clamp_result):
                send_status = "canceled-confirmation"
                debug_text = build_live_debug_text(
                    step_index=step_index,
                    args=args,
                    runtime=runtime,
                    observation=observation,
                    action_step=action_step,
                    joint_snapshot=joint_snapshot,
                    clamp_result=clamp_result,
                    gripper_assist=gripper_assist,
                    camera_runtime_status=camera_runtime_status,
                    send_status=send_status,
                    sent_action=sent_action,
                )
                action_payload = make_live_action_payload(
                    send_status=send_status,
                    action_step=action_step,
                    joint_snapshot=joint_snapshot,
                    clamp_result=clamp_result,
                    gripper_assist=gripper_assist,
                    sent_action=sent_action,
                    observation=observation,
                    camera_runtime_status=camera_runtime_status,
                )
                write_live_runtime_debug(
                    runtime.run_dir,
                    step_index=step_index,
                    preview_frame_bgr=preview_frame,
                    debug_text=debug_text,
                    action_payload=action_payload,
                    append_history=True,
                )
                print("[Exit] User canceled before sending.")
                break

            sent_action = robot.send_action(clamp_result.clamped_action)
            send_status = "sent"
            debug_text = build_live_debug_text(
                step_index=step_index,
                args=args,
                runtime=runtime,
                observation=observation,
                action_step=action_step,
                joint_snapshot=joint_snapshot,
                clamp_result=clamp_result,
                gripper_assist=gripper_assist,
                camera_runtime_status=camera_runtime_status,
                send_status=send_status,
                sent_action=sent_action,
            )
            action_payload = make_live_action_payload(
                send_status=send_status,
                action_step=action_step,
                joint_snapshot=joint_snapshot,
                clamp_result=clamp_result,
                gripper_assist=gripper_assist,
                sent_action=sent_action,
                observation=observation,
                camera_runtime_status=camera_runtime_status,
            )
            if step_index == 1 or step_index % args.terminal_debug_every == 0:
                print(debug_text.rstrip())
            if step_index == 1 or step_index % args.debug_write_every == 0:
                status_lines = build_live_status_lines(
                    args=args,
                    runtime=runtime,
                    observation=observation,
                    action_step=action_step,
                    joint_snapshot=joint_snapshot,
                    clamp_result=clamp_result,
                    gripper_assist=gripper_assist,
                    camera_runtime_status=camera_runtime_status,
                    send_status=send_status,
                )
                preview_frame = common.build_preview_frame(observation.frames_rgb, status_lines)
                write_live_runtime_debug(
                    runtime.run_dir,
                    step_index=step_index,
                    preview_frame_bgr=preview_frame,
                    debug_text=debug_text,
                    action_payload=action_payload,
                    append_history=True,
                )

            print(common.compact_joint_line("[Sent]", sent_action))
            if args.settle_s > 0:
                time.sleep(args.settle_s)

            if not args.no_window and not show_preview(preview_frame):
                print("[Exit] User requested stop from preview window.")
                break

        print("[Exit] Live run finished.")

    finally:
        try:
            common.cv2.destroyAllWindows()
        except Exception:
            pass
        for camera in reversed(cameras):
            camera.disconnect()
        if robot is not None:
            try:
                if robot.is_connected:
                    robot.disconnect()
            except common.DeviceNotConnectedError:
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



