from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
EXPECTED_LEROBOT_INIT_PATH = (SRC_PATH / "lerobot" / "__init__.py").resolve()
DOTENV_PATH = Path(__file__).with_name(".env")

CAMERA_ID = 1
ROBOT_ID = "my_xlerobot_2wheels_lab"
PORT1 = "COM5"  # left arm + head
PORT2 = "COM4"  # right arm + 2-wheel base
LINEAR_SPEED_MPS = 0.10
ANGULAR_SPEED_DPS = 30.0

DRY_RUN = False
CENTER_TOLERANCE_RATIO = 0.12
SEARCH_TURN_DEG = 15.0
TURN_STEP_DEG = 10.0
FORWARD_STEP_FAR_M = 0.15
FORWARD_STEP_NEAR_M = 0.08
NEAR_HEIGHT_RATIO = 0.28
STOP_HEIGHT_RATIO = 0.58
MISS_LIMIT = 5
MAX_SEARCH_TURNS = 16
MIN_CONSECUTIVE_HITS = 2
SMOOTHING_WINDOW = 3
DETECTION_SCALE = 1.05
WINDOW_NAME = "Local Person Follow Test"
POST_ACTION_PAUSE_S = 0.20
IDLE_LOOP_DELAY_S = 0.03
UPPERBODY_SCALE_FACTOR = 1.03
UPPERBODY_MIN_NEIGHBORS = 2
UPPERBODY_MIN_SIZE = (40, 40)
FACE_SCALE_FACTOR = 1.05
FACE_MIN_NEIGHBORS = 3
FACE_MIN_SIZE = (28, 28)
FACE_TO_BODY_WIDTH_MULTIPLIER = 2.4
FACE_TO_BODY_HEIGHT_MULTIPLIER = 4.8
UPPERBODY_TO_BODY_WIDTH_MULTIPLIER = 1.15
UPPERBODY_TO_BODY_HEIGHT_MULTIPLIER = 2.25


if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import lerobot as local_lerobot

LOCAL_LEROBOT_INIT_PATH = Path(local_lerobot.__file__).resolve()
if LOCAL_LEROBOT_INIT_PATH != EXPECTED_LEROBOT_INIT_PATH:
    raise RuntimeError(
        "This script must run against the local repository source, but imported "
        f"lerobot from {LOCAL_LEROBOT_INIT_PATH}. Expected {EXPECTED_LEROBOT_INIT_PATH}. "
        "Please run: conda activate lerobot && cd C:\\projects\\lerobot && "
        "python -m pip install -e . && python examples/shield/2_move_to_you_local_test.py"
    )

if load_dotenv is not None and DOTENV_PATH.exists():
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


def _format_backend_label(config: OpenCVCameraConfig) -> str:
    return getattr(config.backend, "name", str(config.backend))


class ValidatedLocalCamera:
    def __init__(self, camera: OpenCVCamera, camera_meta: dict[str, Any]) -> None:
        self._camera = camera
        self.camera_id = camera_meta.get("id")
        self.selected_backend = camera_meta.get("selected_backend")
        self.selected_fourcc = camera_meta.get("selected_fourcc")
        self.validation_stats = camera_meta.get("validation_stats")

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


def build_camera(camera_id: int) -> ValidatedLocalCamera:
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

    meta = {
        "id": camera_id,
        "selected_backend": backend_label,
        "selected_fourcc": fourcc_label,
        "validation_stats": stats_message,
    }
    print(f"[Camera] Using backend={backend_label}, fourcc={fourcc_label}: {stats_message}")
    return ValidatedLocalCamera(camera, meta)


def build_robot() -> XLerobot2Wheels:
    robot_config = XLerobot2WheelsConfig(
        id=ROBOT_ID,
        port1=PORT1,
        port2=PORT2,
    )
    robot = XLerobot2Wheels(robot_config)
    robot.connect()
    return robot


def build_servo_controller(robot: XLerobot2Wheels) -> TwoWheelsServoAdapter:
    return TwoWheelsServoAdapter(
        robot,
        right_arm_wheel_usb=PORT2,
        linear_speed_mps=LINEAR_SPEED_MPS,
        angular_speed_dps=ANGULAR_SPEED_DPS,
        max_distance_per_step_m=max(FORWARD_STEP_FAR_M, FORWARD_STEP_NEAR_M),
    )


def build_people_detector() -> cv2.HOGDescriptor:
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return detector


def build_cascade_detector(filename: str) -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
    if detector.empty():
        raise RuntimeError(f"Failed to load cascade detector: {filename}")
    return detector


@dataclass
class PersonDetection:
    x: int
    y: int
    w: int
    h: int
    score: float
    frame_width: int
    frame_height: int
    source: str = "hog"

    @property
    def center_x(self) -> float:
        return self.x + (self.w / 2.0)

    @property
    def center_y(self) -> float:
        return self.y + (self.h / 2.0)

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def height_ratio(self) -> float:
        return self.h / float(self.frame_height)


@dataclass
class TrackerState:
    recent_center_x: deque[float] = field(default_factory=lambda: deque(maxlen=SMOOTHING_WINDOW))
    recent_center_y: deque[float] = field(default_factory=lambda: deque(maxlen=SMOOTHING_WINDOW))
    recent_height_ratio: deque[float] = field(default_factory=lambda: deque(maxlen=SMOOTHING_WINDOW))
    last_detection: PersonDetection | None = None
    consecutive_hits: int = 0
    miss_count: int = 0
    search_turns: int = 0

    def record_detection(self, detection: PersonDetection) -> None:
        self.last_detection = detection
        self.recent_center_x.append(detection.center_x)
        self.recent_center_y.append(detection.center_y)
        self.recent_height_ratio.append(detection.height_ratio)
        self.consecutive_hits += 1
        self.miss_count = 0
        self.search_turns = 0

    def record_miss(self) -> None:
        self.last_detection = None
        self.recent_center_x.clear()
        self.recent_center_y.clear()
        self.recent_height_ratio.clear()
        self.consecutive_hits = 0
        self.miss_count += 1

    @property
    def smoothed_center_x(self) -> float | None:
        if not self.recent_center_x:
            return None
        return float(sum(self.recent_center_x) / len(self.recent_center_x))

    @property
    def smoothed_center_y(self) -> float | None:
        if not self.recent_center_y:
            return None
        return float(sum(self.recent_center_y) / len(self.recent_center_y))

    @property
    def smoothed_height_ratio(self) -> float | None:
        if not self.recent_height_ratio:
            return None
        return float(sum(self.recent_height_ratio) / len(self.recent_height_ratio))


@dataclass
class ActionDecision:
    kind: str
    value: float = 0.0
    reason: str = ""
    distance_band: str = "unknown"


def _apply_nms(candidates: list[PersonDetection], *, score_threshold: float, nms_threshold: float) -> list[PersonDetection]:
    if not candidates:
        return []

    boxes = [[det.x, det.y, det.w, det.h] for det in candidates]
    scores = [max(det.score, 0.0) for det in candidates]
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=score_threshold, nms_threshold=nms_threshold)
    if len(indices) == 0:
        return candidates

    kept = np.asarray(indices).reshape(-1).tolist()
    return [candidates[index] for index in kept]


def _clip_box(x: float, y: float, w: float, h: float, frame_width: int, frame_height: int) -> tuple[int, int, int, int] | None:
    left = max(0, int(round(x)))
    top = max(0, int(round(y)))
    right = min(frame_width, int(round(x + w)))
    bottom = min(frame_height, int(round(y + h)))
    clipped_w = right - left
    clipped_h = bottom - top
    if clipped_w <= 0 or clipped_h <= 0:
        return None
    return left, top, clipped_w, clipped_h


def _detect_people_hog(
    frame_bgr: np.ndarray,
    frame_width: int,
    frame_height: int,
    detector: cv2.HOGDescriptor,
) -> list[PersonDetection]:
    rects, weights = detector.detectMultiScale(
        frame_bgr,
        winStride=(8, 8),
        padding=(8, 8),
        scale=DETECTION_SCALE,
    )

    if len(rects) == 0:
        return []

    weight_values = np.asarray(weights, dtype=float).reshape(-1)
    candidates = [
        PersonDetection(
            x=int(x),
            y=int(y),
            w=int(w),
            h=int(h),
            score=float(weight),
            frame_width=frame_width,
            frame_height=frame_height,
            source="hog",
        )
        for (x, y, w, h), weight in zip(rects, weight_values)
    ]

    candidates = _apply_nms(candidates, score_threshold=0.0, nms_threshold=0.35)
    candidates.sort(key=lambda det: (det.score, det.area), reverse=True)
    return candidates


def _detect_people_upperbody(
    frame_gray: np.ndarray,
    frame_width: int,
    frame_height: int,
    detector: cv2.CascadeClassifier,
) -> list[PersonDetection]:
    rects = detector.detectMultiScale(
        frame_gray,
        scaleFactor=UPPERBODY_SCALE_FACTOR,
        minNeighbors=UPPERBODY_MIN_NEIGHBORS,
        minSize=UPPERBODY_MIN_SIZE,
    )

    candidates: list[PersonDetection] = []
    for x, y, w, h in rects:
        clipped = _clip_box(
            x - (w * (UPPERBODY_TO_BODY_WIDTH_MULTIPLIER - 1.0) / 2.0),
            y,
            w * UPPERBODY_TO_BODY_WIDTH_MULTIPLIER,
            h * UPPERBODY_TO_BODY_HEIGHT_MULTIPLIER,
            frame_width,
            frame_height,
        )
        if clipped is None:
            continue
        left, top, clipped_w, clipped_h = clipped
        score = float((w * h) / max(frame_width * frame_height, 1))
        candidates.append(
            PersonDetection(
                x=left,
                y=top,
                w=clipped_w,
                h=clipped_h,
                score=score,
                frame_width=frame_width,
                frame_height=frame_height,
                source="upperbody",
            )
        )

    candidates = _apply_nms(candidates, score_threshold=0.0, nms_threshold=0.30)
    candidates.sort(key=lambda det: (det.score, det.area), reverse=True)
    return candidates


def _detect_people_face(
    frame_gray: np.ndarray,
    frame_width: int,
    frame_height: int,
    frontal_detector: cv2.CascadeClassifier,
    alt_detector: cv2.CascadeClassifier,
    profile_detector: cv2.CascadeClassifier,
) -> list[PersonDetection]:
    face_rects: list[tuple[int, int, int, int, str]] = []

    frontal = frontal_detector.detectMultiScale(
        frame_gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE,
    )
    face_rects.extend((int(x), int(y), int(w), int(h), "face") for x, y, w, h in frontal)

    alt = alt_detector.detectMultiScale(
        frame_gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE,
    )
    face_rects.extend((int(x), int(y), int(w), int(h), "face_alt") for x, y, w, h in alt)

    profile = profile_detector.detectMultiScale(
        frame_gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE,
    )
    face_rects.extend((int(x), int(y), int(w), int(h), "profile_face") for x, y, w, h in profile)

    flipped = cv2.flip(frame_gray, 1)
    flipped_profile = profile_detector.detectMultiScale(
        flipped,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE,
    )
    for x, y, w, h in flipped_profile:
        face_rects.append((frame_width - int(x + w), int(y), int(w), int(h), "profile_face"))

    candidates: list[PersonDetection] = []
    for x, y, w, h, source in face_rects:
        clipped = _clip_box(
            x - ((FACE_TO_BODY_WIDTH_MULTIPLIER - 1.0) * w / 2.0),
            y - (0.35 * h),
            w * FACE_TO_BODY_WIDTH_MULTIPLIER,
            h * FACE_TO_BODY_HEIGHT_MULTIPLIER,
            frame_width,
            frame_height,
        )
        if clipped is None:
            continue
        left, top, clipped_w, clipped_h = clipped
        score = float((w * h) / max(frame_width * frame_height, 1))
        candidates.append(
            PersonDetection(
                x=left,
                y=top,
                w=clipped_w,
                h=clipped_h,
                score=score,
                frame_width=frame_width,
                frame_height=frame_height,
                source=source,
            )
        )

    candidates = _apply_nms(candidates, score_threshold=0.0, nms_threshold=0.30)
    candidates.sort(key=lambda det: (det.score, det.area), reverse=True)
    return candidates


def detect_people(
    frame_bgr: np.ndarray,
    frame_width: int,
    frame_height: int,
    hog_detector: cv2.HOGDescriptor,
    upperbody_detector: cv2.CascadeClassifier,
    frontal_face_detector: cv2.CascadeClassifier,
    alt_face_detector: cv2.CascadeClassifier,
    profile_face_detector: cv2.CascadeClassifier,
) -> list[PersonDetection]:
    hog_candidates = _detect_people_hog(frame_bgr, frame_width, frame_height, hog_detector)
    if hog_candidates:
        return hog_candidates

    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    upperbody_candidates = _detect_people_upperbody(frame_gray, frame_width, frame_height, upperbody_detector)
    if upperbody_candidates:
        return upperbody_candidates

    return _detect_people_face(
        frame_gray,
        frame_width,
        frame_height,
        frontal_face_detector,
        alt_face_detector,
        profile_face_detector,
    )


def classify_distance_band(height_ratio: float | None) -> str:
    if height_ratio is None:
        return "unknown"
    if height_ratio >= STOP_HEIGHT_RATIO:
        return "near"
    if height_ratio >= NEAR_HEIGHT_RATIO:
        return "medium"
    return "far"


def decide_next_action(state: TrackerState, frame_width: int) -> ActionDecision:
    if state.last_detection is None:
        if state.miss_count >= MISS_LIMIT:
            if state.search_turns >= MAX_SEARCH_TURNS:
                return ActionDecision(
                    kind="exit",
                    reason=(
                        f"target missing for {state.miss_count} frames and "
                        f"search budget {MAX_SEARCH_TURNS} is exhausted"
                    ),
                )
            direction = "turn_left" if state.search_turns % 2 == 0 else "turn_right"
            return ActionDecision(
                kind=direction,
                value=SEARCH_TURN_DEG,
                reason=f"target missing for {state.miss_count} frames, start search turn {state.search_turns + 1}",
            )
        return ActionDecision(
            kind="wait",
            reason=f"target missing for {state.miss_count}/{MISS_LIMIT} frames",
        )

    smoothed_center_x = state.smoothed_center_x
    smoothed_height_ratio = state.smoothed_height_ratio
    distance_band = classify_distance_band(smoothed_height_ratio)

    if smoothed_center_x is None or smoothed_height_ratio is None:
        return ActionDecision(kind="wait", reason="not enough tracking history", distance_band=distance_band)

    if state.consecutive_hits < MIN_CONSECUTIVE_HITS:
        return ActionDecision(
            kind="wait",
            reason=f"stabilizing target {state.consecutive_hits}/{MIN_CONSECUTIVE_HITS}",
            distance_band=distance_band,
        )

    frame_center_x = frame_width / 2.0
    offset_x = smoothed_center_x - frame_center_x
    tolerance_px = frame_width * CENTER_TOLERANCE_RATIO
    if abs(offset_x) > tolerance_px:
        if offset_x < 0:
            return ActionDecision(
                kind="turn_left",
                value=TURN_STEP_DEG,
                reason=f"target offset {offset_x:.0f}px exceeds tolerance {tolerance_px:.0f}px",
                distance_band=distance_band,
            )
        return ActionDecision(
            kind="turn_right",
            value=TURN_STEP_DEG,
            reason=f"target offset {offset_x:.0f}px exceeds tolerance {tolerance_px:.0f}px",
            distance_band=distance_band,
        )

    if smoothed_height_ratio >= STOP_HEIGHT_RATIO:
        return ActionDecision(
            kind="stop",
            reason=f"height ratio {smoothed_height_ratio:.2f} reached stop threshold {STOP_HEIGHT_RATIO:.2f}",
            distance_band=distance_band,
        )

    if smoothed_height_ratio >= NEAR_HEIGHT_RATIO:
        return ActionDecision(
            kind="forward",
            value=FORWARD_STEP_NEAR_M,
            reason=f"height ratio {smoothed_height_ratio:.2f} is in medium range",
            distance_band=distance_band,
        )

    return ActionDecision(
        kind="forward",
        value=FORWARD_STEP_FAR_M,
        reason=f"height ratio {smoothed_height_ratio:.2f} is in far range",
        distance_band=distance_band,
    )


def format_action_label(action: ActionDecision) -> str:
    if action.kind == "forward":
        return f"forward {action.value:.2f}m"
    if action.kind in {"turn_left", "turn_right"}:
        direction = "left" if action.kind == "turn_left" else "right"
        return f"turn {direction} {action.value:.0f}deg"
    return action.kind


def draw_overlay(frame_bgr: np.ndarray, state: TrackerState, action: ActionDecision) -> np.ndarray:
    annotated = frame_bgr.copy()
    height, width = annotated.shape[:2]
    center_x = width // 2
    tolerance_px = int(width * CENTER_TOLERANCE_RATIO)

    cv2.line(annotated, (center_x, 0), (center_x, height), (0, 255, 255), 1)
    cv2.line(annotated, (center_x - tolerance_px, 0), (center_x - tolerance_px, height), (80, 80, 80), 1)
    cv2.line(annotated, (center_x + tolerance_px, 0), (center_x + tolerance_px, height), (80, 80, 80), 1)

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
        f"DRY_RUN={DRY_RUN}",
        f"action: {format_action_label(action)}",
        f"band: {action.distance_band}",
        f"hits={state.consecutive_hits} misses={state.miss_count} search={state.search_turns}/{MAX_SEARCH_TURNS}",
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
        lines.extend([offset_text, score_text, source_text])
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


def perform_action(action: ActionDecision, servo_controller: TwoWheelsServoAdapter | None) -> None:
    if action.kind in {"wait", "stop", "exit"}:
        return

    print(f"[Decision] {format_action_label(action)} | {action.reason}")

    if DRY_RUN or servo_controller is None:
        return

    if action.kind == "forward":
        servo_controller.go_forward(action.value)
    elif action.kind == "turn_left":
        servo_controller.turn_left(action.value)
    elif action.kind == "turn_right":
        servo_controller.turn_right(action.value)

    time.sleep(POST_ACTION_PAUSE_S)


def print_runtime_summary(camera: ValidatedLocalCamera) -> None:
    print(f"[Env] Python executable: {sys.executable}")
    print(f"[Env] lerobot module: {LOCAL_LEROBOT_INIT_PATH}")
    print(f"[Debug] DRY_RUN: {DRY_RUN}")
    print(f"[Debug] Camera ID: {CAMERA_ID}")
    print(f"[Debug] Raw camera backend/fourcc: {camera.selected_backend}/{camera.selected_fourcc}")
    print(f"[Debug] Raw camera validation stats: {camera.validation_stats}")
    print(f"[Debug] Center tolerance ratio: {CENTER_TOLERANCE_RATIO:.2f}")
    print(f"[Debug] Search turn deg: {SEARCH_TURN_DEG:.0f}")
    print(f"[Debug] Stop height ratio: {STOP_HEIGHT_RATIO:.2f}")
    if DRY_RUN:
        print("[Debug] Dry-run mode is enabled, the robot base will not move.")
    else:
        print("[Debug] Live mode is enabled, the robot base can move.")


def main() -> None:
    camera = None
    robot = None
    servo_controller = None

    try:
        camera = build_camera(CAMERA_ID)
        if not DRY_RUN:
            robot = build_robot()
            servo_controller = build_servo_controller(robot)

        print_runtime_summary(camera)

        detector = build_people_detector()
        upperbody_detector = build_cascade_detector("haarcascade_upperbody.xml")
        frontal_face_detector = build_cascade_detector("haarcascade_frontalface_default.xml")
        alt_face_detector = build_cascade_detector("haarcascade_frontalface_alt2.xml")
        profile_face_detector = build_cascade_detector("haarcascade_profileface.xml")
        tracker_state = TrackerState()

        while True:
            try:
                frame_bgr = camera.read_bgr()
            except RuntimeError as exc:
                print(f"[Camera] Read failed: {exc}")
                time.sleep(0.3)
                camera.reopen()
                continue

            frame_height, frame_width = frame_bgr.shape[:2]
            detections = detect_people(
                frame_bgr,
                frame_width,
                frame_height,
                detector,
                upperbody_detector,
                frontal_face_detector,
                alt_face_detector,
                profile_face_detector,
            )

            if detections:
                tracker_state.record_detection(detections[0])
            else:
                tracker_state.record_miss()

            action = decide_next_action(tracker_state, frame_width)
            overlay = draw_overlay(frame_bgr, tracker_state, action)
            cv2.imshow(WINDOW_NAME, overlay)

            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                print("[Exit] User requested stop from keyboard.")
                break

            if tracker_state.last_detection is None and action.kind in {"turn_left", "turn_right"}:
                tracker_state.search_turns += 1

            if action.kind == "stop":
                print(f"[Done] {action.reason}")
                break

            if action.kind == "exit":
                print(f"[Exit] {action.reason}")
                break

            perform_action(action, servo_controller)
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
            except DeviceNotConnectedError:
                pass
        if camera is not None:
            camera.disconnect()


if __name__ == "__main__":
    main()
