from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from runtime_common import expand_face_to_body_box
from target_face_matcher import TargetFaceMatch


CENTER_TOLERANCE_RATIO = 0.12
SEARCH_TURN_DEG = 15.0
SEARCH_SWEEP_DEG = 360.0
TURN_STEP_DEG = 10.0
FORWARD_STEP_FAR_M = 0.15
FORWARD_STEP_NEAR_M = 0.08
FORWARD_STEP_FUZZY_M = 0.06
NEAR_HEIGHT_RATIO = 0.28
STOP_HEIGHT_RATIO = 0.58
MISS_LIMIT = 5
MAX_SEARCH_TURNS = math.ceil(SEARCH_SWEEP_DEG / SEARCH_TURN_DEG)
MIN_CONSECUTIVE_HITS = 2
FUZZY_MIN_CONSECUTIVE_HITS = 4
SMOOTHING_WINDOW = 3
DEFAULT_SEARCH_DIRECTION = "left"


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
    identity_score: float | None = None
    identity_source: str | None = None

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

    def contains_point(self, point_x: float, point_y: float) -> bool:
        return self.x <= point_x <= (self.x + self.w) and self.y <= point_y <= (self.y + self.h)

    def with_identity(self, *, identity_score: float, identity_source: str) -> "PersonDetection":
        return PersonDetection(
            x=self.x,
            y=self.y,
            w=self.w,
            h=self.h,
            score=self.score,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            source=self.source,
            identity_score=float(identity_score),
            identity_source=identity_source,
        )


@dataclass
class TrackerState:
    recent_center_x: deque[float] = field(default_factory=lambda: deque(maxlen=SMOOTHING_WINDOW))
    recent_center_y: deque[float] = field(default_factory=lambda: deque(maxlen=SMOOTHING_WINDOW))
    recent_height_ratio: deque[float] = field(default_factory=lambda: deque(maxlen=SMOOTHING_WINDOW))
    last_detection: PersonDetection | None = None
    consecutive_hits: int = 0
    miss_count: int = 0
    search_turns: int = 0
    last_seen_offset_x: float | None = None
    search_direction: str | None = None
    searched_degrees: float = 0.0

    @staticmethod
    def _direction_from_offset(offset_x: float | None) -> str | None:
        if offset_x is None:
            return None
        if offset_x < 0:
            return "left"
        if offset_x > 0:
            return "right"
        return None

    def preferred_search_direction(self) -> str:
        hinted_direction = self._direction_from_offset(self.last_seen_offset_x)
        if hinted_direction is not None:
            return hinted_direction
        if self.search_direction in {"left", "right"}:
            return self.search_direction
        return DEFAULT_SEARCH_DIRECTION

    def record_search_step(self, action_kind: str, degrees: float) -> None:
        if action_kind == "turn_left":
            self.search_direction = "left"
        elif action_kind == "turn_right":
            self.search_direction = "right"
        else:
            return

        self.search_turns += 1
        self.searched_degrees += abs(float(degrees))

    def record_detection(self, detection: PersonDetection) -> None:
        self.last_detection = detection
        self.recent_center_x.append(detection.center_x)
        self.recent_center_y.append(detection.center_y)
        self.recent_height_ratio.append(detection.height_ratio)
        self.consecutive_hits += 1
        self.miss_count = 0
        self.search_turns = 0
        self.searched_degrees = 0.0

        offset_x = detection.center_x - (detection.frame_width / 2.0)
        self.last_seen_offset_x = float(offset_x)
        hinted_direction = self._direction_from_offset(self.last_seen_offset_x)
        if hinted_direction is not None:
            self.search_direction = hinted_direction

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


def _identity_source(match_quality: str, *, fallback: bool = False) -> str:
    suffix = "_fallback" if fallback else ""
    return f"face_id_{match_quality}{suffix}"


def _intersection_area(detection: PersonDetection, match: TargetFaceMatch) -> int:
    left = max(detection.x, match.face.x)
    top = max(detection.y, match.face.y)
    right = min(detection.x + detection.w, match.face.x + match.face.w)
    bottom = min(detection.y + detection.h, match.face.y + match.face.h)
    if right <= left or bottom <= top:
        return 0
    return int((right - left) * (bottom - top))


def _build_face_fallback_detection(
    match: TargetFaceMatch,
    frame_width: int,
    frame_height: int,
) -> PersonDetection | None:
    clipped = expand_face_to_body_box(
        match.face.x,
        match.face.y,
        match.face.w,
        match.face.h,
        frame_width,
        frame_height,
    )
    if clipped is None:
        return None

    left, top, clipped_w, clipped_h = clipped
    identity_source = _identity_source(match.match_quality, fallback=True)
    return PersonDetection(
        x=left,
        y=top,
        w=clipped_w,
        h=clipped_h,
        score=float(match.face.score),
        frame_width=frame_width,
        frame_height=frame_height,
        source=identity_source,
        identity_score=match.similarity,
        identity_source=identity_source,
    )


def select_target_detection(
    people_detections: list[PersonDetection],
    target_match: TargetFaceMatch | None,
    *,
    frame_width: int,
    frame_height: int,
) -> PersonDetection | None:
    if target_match is None:
        return None

    identity_source = _identity_source(target_match.match_quality)
    best_detection: PersonDetection | None = None
    best_overlap_score = -1.0

    for detection in people_detections:
        if not detection.contains_point(target_match.face.center_x, target_match.face.center_y):
            continue

        overlap_area = _intersection_area(detection, target_match)
        face_area = max(target_match.face.w * target_match.face.h, 1)
        overlap_score = (overlap_area / face_area) + (detection.score * 1e-3)
        if overlap_score <= best_overlap_score:
            continue

        best_overlap_score = overlap_score
        best_detection = detection.with_identity(
            identity_score=target_match.similarity,
            identity_source=identity_source,
        )

    if best_detection is not None:
        return best_detection

    return _build_face_fallback_detection(target_match, frame_width, frame_height)


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
            if state.searched_degrees >= SEARCH_SWEEP_DEG or state.search_turns >= MAX_SEARCH_TURNS:
                return ActionDecision(
                    kind="exit",
                    reason=(
                        f"target missing for {state.miss_count} frames and completed "
                        f"{state.searched_degrees:.0f}/{SEARCH_SWEEP_DEG:.0f}deg search sweep"
                    ),
                )

            direction_name = state.preferred_search_direction()
            direction = "turn_left" if direction_name == "left" else "turn_right"
            remaining_degrees = max(SEARCH_SWEEP_DEG - state.searched_degrees, 0.0)
            step_degrees = min(SEARCH_TURN_DEG, remaining_degrees)
            if step_degrees <= 0:
                return ActionDecision(
                    kind="exit",
                    reason=(
                        f"target missing for {state.miss_count} frames and completed "
                        f"{state.searched_degrees:.0f}/{SEARCH_SWEEP_DEG:.0f}deg search sweep"
                    ),
                )
            return ActionDecision(
                kind=direction,
                value=step_degrees,
                reason=(
                    f"target missing for {state.miss_count} frames, search {direction_name} "
                    f"{state.search_turns + 1}/{MAX_SEARCH_TURNS} "
                    f"({state.searched_degrees:.0f}/{SEARCH_SWEEP_DEG:.0f}deg)"
                ),
            )

        return ActionDecision(kind="wait", reason=f"target missing for {state.miss_count}/{MISS_LIMIT} frames")

    smoothed_center_x = state.smoothed_center_x
    smoothed_height_ratio = state.smoothed_height_ratio
    distance_band = classify_distance_band(smoothed_height_ratio)
    if smoothed_center_x is None or smoothed_height_ratio is None:
        return ActionDecision(kind="wait", reason="not enough tracking history", distance_band=distance_band)

    identity_source = state.last_detection.identity_source or ""
    fuzzy_identity = "fuzzy" in identity_source
    required_hits = FUZZY_MIN_CONSECUTIVE_HITS if fuzzy_identity else MIN_CONSECUTIVE_HITS
    if state.consecutive_hits < required_hits:
        return ActionDecision(
            kind="wait",
            reason=f"stabilizing target {state.consecutive_hits}/{required_hits}",
            distance_band=distance_band,
        )

    frame_center_x = frame_width / 2.0
    offset_x = smoothed_center_x - frame_center_x
    tolerance_px = frame_width * CENTER_TOLERANCE_RATIO
    if abs(offset_x) > tolerance_px:
        direction = "turn_left" if offset_x < 0 else "turn_right"
        return ActionDecision(
            kind=direction,
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

    if fuzzy_identity:
        return ActionDecision(
            kind="forward",
            value=FORWARD_STEP_FUZZY_M,
            reason=(
                f"small-face fuzzy match {state.last_detection.identity_score:.2f} accepted, "
                "advance cautiously"
            ),
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
