from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from follow_logic import (
    CENTER_TOLERANCE_RATIO,
    FORWARD_STEP_FAR_M,
    FORWARD_STEP_FUZZY_M,
    FORWARD_STEP_NEAR_M,
    FUZZY_MIN_CONSECUTIVE_HITS,
    MAX_SEARCH_TURNS,
    SEARCH_SWEEP_DEG,
    SEARCH_TURN_DEG,
    STOP_HEIGHT_RATIO,
    ActionDecision,
    PersonDetection,
    TrackerState,
    decide_next_action,
    format_action_label,
    select_target_detection,
)
from people_detection import PeopleDetectorStack
from runtime_common import (
    DEFAULT_ANGULAR_SPEED_DPS,
    DEFAULT_CAMERA_ID,
    DEFAULT_LINEAR_SPEED_MPS,
    DEFAULT_PORT1,
    DEFAULT_PORT2,
    DEFAULT_ROBOT_ID,
    ValidatedLocalCamera,
    assert_local_lerobot_source,
    build_camera,
    build_robot,
    build_servo_controller,
    ensure_directory,
    prompt_person_name,
    write_debug_image,
    write_debug_text,
)
from target_face_matcher import ReferenceGallery, TargetFaceMatch, TargetFaceMatcher


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE_ROOT = SCRIPT_DIR / "reference_person"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "models"
DEFAULT_DEBUG_DIR = SCRIPT_DIR / "debug"

WINDOW_NAME = "Target Person Follow Test"
POST_ACTION_PAUSE_S = 0.20
IDLE_LOOP_DELAY_S = 0.03
DEBUG_WRITE_EVERY_N_FRAMES = 10
RUNTIME_FACE_DETECT_SCORE_THRESHOLD = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Follow one enrolled target person using local CV-only detection plus face identity matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera-id", type=int, default=DEFAULT_CAMERA_ID)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--port1", default=DEFAULT_PORT1)
    parser.add_argument("--port2", default=DEFAULT_PORT2)
    parser.add_argument("--linear-speed-mps", type=float, default=DEFAULT_LINEAR_SPEED_MPS)
    parser.add_argument("--angular-speed-dps", type=float, default=DEFAULT_ANGULAR_SPEED_DPS)
    parser.add_argument("--live", action="store_true", help="Allow the robot base to move.")
    return parser.parse_args()


def write_runtime_debug(
    debug_dir: Path,
    *,
    overlay_bgr: np.ndarray,
    source_frame_bgr: np.ndarray,
    person_name: str,
    gallery: ReferenceGallery,
    state: TrackerState,
    target_match: TargetFaceMatch | None,
) -> None:
    write_debug_image(debug_dir / "last_frame.jpg", overlay_bgr)
    if target_match is not None:
        write_debug_image(debug_dir / "last_target_face.jpg", target_match.face.extract_roi(source_frame_bgr))
    write_debug_text(
        debug_dir / "last_match.txt",
        "\n".join(
            [
                f"target_name={person_name}",
                f"reference_dir={gallery.reference_dir}",
                f"reference_count={gallery.count}",
                f"matched={target_match is not None}",
                f"similarity={target_match.similarity:.4f}" if target_match is not None else "similarity=n/a",
                f"match_quality={target_match.match_quality}" if target_match is not None else "match_quality=n/a",
                f"reference_path={target_match.reference_path}" if target_match is not None else "reference_path=n/a",
                f"search_direction={state.preferred_search_direction()}",
                f"search_turns={state.search_turns}/{MAX_SEARCH_TURNS}",
                f"search_degrees={state.searched_degrees:.0f}/{SEARCH_SWEEP_DEG:.0f}",
            ]
        )
        + "\n",
    )


def draw_overlay(
    frame_bgr: np.ndarray,
    state: TrackerState,
    action: ActionDecision,
    *,
    target_match: TargetFaceMatch | None,
    target_name: str,
    reference_count: int,
    dry_run: bool,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    height, width = annotated.shape[:2]
    center_x = width // 2
    tolerance_px = int(width * CENTER_TOLERANCE_RATIO)

    cv2.line(annotated, (center_x, 0), (center_x, height), (0, 255, 255), 1)
    cv2.line(annotated, (center_x - tolerance_px, 0), (center_x - tolerance_px, height), (80, 80, 80), 1)
    cv2.line(annotated, (center_x + tolerance_px, 0), (center_x + tolerance_px, height), (80, 80, 80), 1)

    if target_match is not None:
        cv2.rectangle(
            annotated,
            (target_match.face.x, target_match.face.y),
            (target_match.face.x + target_match.face.w, target_match.face.y + target_match.face.h),
            (255, 170, 0),
            2,
        )

    detection = state.last_detection
    if detection is not None:
        top_left = (detection.x, detection.y)
        bottom_right = (detection.x + detection.w, detection.y + detection.h)
        cv2.rectangle(annotated, top_left, bottom_right, (0, 220, 0), 2)

        raw_center = (int(detection.center_x), int(detection.center_y))
        cv2.circle(annotated, raw_center, 4, (0, 220, 255), -1)

        if state.smoothed_center_x is not None and state.smoothed_center_y is not None:
            smoothed_center = (int(state.smoothed_center_x), int(state.smoothed_center_y))
            cv2.circle(annotated, smoothed_center, 5, (255, 255, 0), 2)

    lines = [
        f"DRY_RUN={dry_run}",
        f"target_name={target_name}",
        f"reference_count={reference_count}",
        f"target_match={'yes' if target_match is not None else 'no'}",
        f"similarity={target_match.similarity:.3f}" if target_match is not None else "similarity=n/a",
        f"match_quality={target_match.match_quality}" if target_match is not None else "match_quality=n/a",
        f"action: {format_action_label(action)}",
        f"band: {action.distance_band}",
        f"hits={state.consecutive_hits} misses={state.miss_count}",
        f"search_dir={state.preferred_search_direction()} "
        f"search={state.searched_degrees:.0f}/{SEARCH_SWEEP_DEG:.0f}deg "
        f"steps={state.search_turns}/{MAX_SEARCH_TURNS}",
        "h_ratio="
        + (
            f"{state.smoothed_height_ratio:.2f}"
            if state.smoothed_height_ratio is not None
            else "n/a"
        ),
    ]

    if detection is not None:
        offset_text = "offset_x=" + (
            f"{state.smoothed_center_x - (width / 2.0):.0f}px" if state.smoothed_center_x is not None else "n/a"
        )
        score_text = f"score={detection.score:.2f}"
        source_text = f"source={detection.source}"
        identity_text = (
            f"id_score={detection.identity_score:.3f}" if detection.identity_score is not None else "id_score=n/a"
        )
        lines.extend([offset_text, score_text, source_text, identity_text])
    else:
        lines.append("target: not found")

    lines.append("q or ESC to quit")

    y = 24
    for line in lines:
        cv2.putText(annotated, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24

    if action.reason:
        cv2.putText(
            annotated,
            action.reason,
            (12, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            action.reason,
            (12, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def perform_action(action: ActionDecision, *, servo_controller, dry_run: bool) -> None:
    if action.kind in {"wait", "stop", "exit"}:
        return

    print(f"[Decision] {format_action_label(action)} | {action.reason}")

    if dry_run or servo_controller is None:
        return

    if action.kind == "forward":
        servo_controller.go_forward(action.value)
    elif action.kind == "turn_left":
        servo_controller.turn_left(action.value)
    elif action.kind == "turn_right":
        servo_controller.turn_right(action.value)

    time.sleep(POST_ACTION_PAUSE_S)


def print_runtime_summary(
    camera: ValidatedLocalCamera,
    *,
    target_name: str,
    gallery: ReferenceGallery,
    dry_run: bool,
    args: argparse.Namespace,
) -> None:
    print(f"[Env] Python executable: {sys.executable}")
    print(f"[Debug] target_name: {target_name}")
    print(f"[Debug] reference_dir: {gallery.reference_dir}")
    print(f"[Debug] reference_count: {gallery.count}")
    print(f"[Debug] DRY_RUN: {dry_run}")
    print(f"[Debug] Camera ID: {args.camera_id}")
    print(f"[Debug] Raw camera backend/fourcc: {camera.selected_backend}/{camera.selected_fourcc}")
    print(f"[Debug] Raw camera validation stats: {camera.validation_stats}")
    print(f"[Debug] Center tolerance ratio: {CENTER_TOLERANCE_RATIO:.2f}")
    print(f"[Debug] Search turn deg: {SEARCH_TURN_DEG:.0f}")
    print(f"[Debug] Search sweep deg: {SEARCH_SWEEP_DEG:.0f}")
    print(f"[Debug] Stop height ratio: {STOP_HEIGHT_RATIO:.2f}")
    print(f"[Debug] Fuzzy hit threshold: {FUZZY_MIN_CONSECUTIVE_HITS}")
    print(f"[Debug] Runtime face detect threshold: {RUNTIME_FACE_DETECT_SCORE_THRESHOLD:.2f}")
    print(f"[Debug] Debug dir: {args.debug_dir}")
    if dry_run:
        print("[Debug] Dry-run mode is enabled, the robot base will not move.")
    else:
        print("[Debug] Live mode is enabled, the robot base can move.")


def main() -> None:
    args = parse_args()
    assert_local_lerobot_source("move_to_target_person_by_cv.py")
    person_name = prompt_person_name("Target person name to follow: ")
    debug_dir = ensure_directory(args.debug_dir)
    dry_run = not args.live

    camera = None
    robot = None
    servo_controller = None

    try:
        matcher = TargetFaceMatcher(
            args.model_dir,
            detect_score_threshold=RUNTIME_FACE_DETECT_SCORE_THRESHOLD,
        )
        gallery = matcher.load_reference_gallery(args.reference_root, person_name)
        detector_stack = PeopleDetectorStack.create()
        camera = build_camera(args.camera_id)
        if not dry_run:
            robot = build_robot(args.robot_id, args.port1, args.port2)
            servo_controller = build_servo_controller(
                robot,
                port2=args.port2,
                linear_speed_mps=args.linear_speed_mps,
                angular_speed_dps=args.angular_speed_dps,
                max_distance_per_step_m=max(FORWARD_STEP_FAR_M, FORWARD_STEP_NEAR_M, FORWARD_STEP_FUZZY_M),
            )

        print_runtime_summary(camera, target_name=person_name, gallery=gallery, dry_run=dry_run, args=args)

        tracker_state = TrackerState()
        frame_counter = 0

        while True:
            try:
                frame_bgr = camera.read_bgr()
            except RuntimeError as exc:
                print(f"[Camera] Read failed: {exc}")
                time.sleep(0.3)
                camera.reopen()
                continue

            frame_height, frame_width = frame_bgr.shape[:2]
            people_detections = detector_stack.detect_people(frame_bgr)
            target_match = matcher.match_target_face(frame_bgr, gallery)
            target_detection = select_target_detection(
                people_detections,
                target_match,
                frame_width=frame_width,
                frame_height=frame_height,
            )

            if target_detection is not None:
                tracker_state.record_detection(target_detection)
            else:
                tracker_state.record_miss()

            action = decide_next_action(tracker_state, frame_width)
            overlay = draw_overlay(
                frame_bgr,
                tracker_state,
                action,
                target_match=target_match,
                target_name=person_name,
                reference_count=gallery.count,
                dry_run=dry_run,
            )
            cv2.imshow(WINDOW_NAME, overlay)

            frame_counter += 1
            if frame_counter % DEBUG_WRITE_EVERY_N_FRAMES == 0:
                write_runtime_debug(
                    debug_dir,
                    overlay_bgr=overlay,
                    source_frame_bgr=frame_bgr,
                    person_name=person_name,
                    gallery=gallery,
                    state=tracker_state,
                    target_match=target_match,
                )

            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                print("[Exit] User requested stop from keyboard.")
                break

            if action.kind == "stop":
                print(f"[Done] {action.reason}")
                break

            if action.kind == "exit":
                print(f"[Exit] {action.reason}")
                break

            perform_action(action, servo_controller=servo_controller, dry_run=dry_run)
            if tracker_state.last_detection is None and action.kind in {"turn_left", "turn_right"}:
                tracker_state.record_search_step(action.kind, action.value)
            if action.kind == "wait":
                time.sleep(IDLE_LOOP_DELAY_S)
    finally:
        cv2.destroyAllWindows()
        if servo_controller is not None:
            servo_controller.disconnect()
        elif robot is not None:
            try:
                if robot.is_connected:
                    robot.disconnect()
            except Exception:
                pass
        if camera is not None:
            camera.disconnect()


if __name__ == "__main__":
    main()
