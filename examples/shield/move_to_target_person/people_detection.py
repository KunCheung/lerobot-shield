from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from follow_logic import PersonDetection
from runtime_common import clip_box, expand_face_to_body_box


DETECTION_SCALE = 1.05
UPPERBODY_SCALE_FACTOR = 1.03
UPPERBODY_MIN_NEIGHBORS = 2
UPPERBODY_MIN_SIZE = (40, 40)
FACE_SCALE_FACTOR = 1.05
FACE_MIN_NEIGHBORS = 3
FACE_MIN_SIZE = (28, 28)
UPPERBODY_TO_BODY_WIDTH_MULTIPLIER = 1.15
UPPERBODY_TO_BODY_HEIGHT_MULTIPLIER = 2.25


def _build_people_detector() -> cv2.HOGDescriptor:
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return detector


def _build_cascade_detector(filename: str) -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
    if detector.empty():
        raise RuntimeError(f"Failed to load cascade detector: {filename}")
    return detector


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


def _sort_candidates(candidates: list[PersonDetection]) -> list[PersonDetection]:
    candidates.sort(key=lambda det: (det.score, det.area), reverse=True)
    return candidates


def _make_detection(
    clipped: tuple[int, int, int, int] | None,
    *,
    score: float,
    frame_width: int,
    frame_height: int,
    source: str,
) -> PersonDetection | None:
    if clipped is None:
        return None
    left, top, clipped_w, clipped_h = clipped
    return PersonDetection(
        x=left,
        y=top,
        w=clipped_w,
        h=clipped_h,
        score=float(score),
        frame_width=frame_width,
        frame_height=frame_height,
        source=source,
    )


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
        for (x, y, w, h), weight in zip(rects, np.asarray(weights, dtype=float).reshape(-1))
    ]
    return _sort_candidates(_apply_nms(candidates, score_threshold=0.0, nms_threshold=0.35))


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
        clipped = clip_box(
            x - (w * (UPPERBODY_TO_BODY_WIDTH_MULTIPLIER - 1.0) / 2.0),
            y,
            w * UPPERBODY_TO_BODY_WIDTH_MULTIPLIER,
            h * UPPERBODY_TO_BODY_HEIGHT_MULTIPLIER,
            frame_width,
            frame_height,
        )
        detection = _make_detection(
            clipped,
            score=(w * h) / max(frame_width * frame_height, 1),
            frame_width=frame_width,
            frame_height=frame_height,
            source="upperbody",
        )
        if detection is not None:
            candidates.append(detection)

    return _sort_candidates(_apply_nms(candidates, score_threshold=0.0, nms_threshold=0.30))


def _collect_face_rects(
    frame_gray: np.ndarray,
    frame_width: int,
    frontal_detector: cv2.CascadeClassifier,
    alt_detector: cv2.CascadeClassifier,
    profile_detector: cv2.CascadeClassifier,
) -> list[tuple[int, int, int, int, str]]:
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

    return face_rects


def _detect_people_face(
    frame_gray: np.ndarray,
    frame_width: int,
    frame_height: int,
    frontal_detector: cv2.CascadeClassifier,
    alt_detector: cv2.CascadeClassifier,
    profile_detector: cv2.CascadeClassifier,
) -> list[PersonDetection]:
    candidates: list[PersonDetection] = []
    for x, y, w, h, source in _collect_face_rects(
        frame_gray,
        frame_width,
        frontal_detector,
        alt_detector,
        profile_detector,
    ):
        detection = _make_detection(
            expand_face_to_body_box(x, y, w, h, frame_width, frame_height),
            score=(w * h) / max(frame_width * frame_height, 1),
            frame_width=frame_width,
            frame_height=frame_height,
            source=source,
        )
        if detection is not None:
            candidates.append(detection)

    return _sort_candidates(_apply_nms(candidates, score_threshold=0.0, nms_threshold=0.30))


@dataclass
class PeopleDetectorStack:
    hog_detector: cv2.HOGDescriptor
    upperbody_detector: cv2.CascadeClassifier
    frontal_face_detector: cv2.CascadeClassifier
    alt_face_detector: cv2.CascadeClassifier
    profile_face_detector: cv2.CascadeClassifier

    @classmethod
    def create(cls) -> "PeopleDetectorStack":
        return cls(
            hog_detector=_build_people_detector(),
            upperbody_detector=_build_cascade_detector("haarcascade_upperbody.xml"),
            frontal_face_detector=_build_cascade_detector("haarcascade_frontalface_default.xml"),
            alt_face_detector=_build_cascade_detector("haarcascade_frontalface_alt2.xml"),
            profile_face_detector=_build_cascade_detector("haarcascade_profileface.xml"),
        )

    def detect_people(self, frame_bgr: np.ndarray) -> list[PersonDetection]:
        frame_height, frame_width = frame_bgr.shape[:2]

        hog_candidates = _detect_people_hog(frame_bgr, frame_width, frame_height, self.hog_detector)
        if hog_candidates:
            return hog_candidates

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        upperbody_candidates = _detect_people_upperbody(
            frame_gray,
            frame_width,
            frame_height,
            self.upperbody_detector,
        )
        if upperbody_candidates:
            return upperbody_candidates

        return _detect_people_face(
            frame_gray,
            frame_width,
            frame_height,
            self.frontal_face_detector,
            self.alt_face_detector,
            self.profile_face_detector,
        )
