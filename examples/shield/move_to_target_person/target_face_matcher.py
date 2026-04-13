from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from runtime_common import iter_image_files, list_available_person_names


DEFAULT_FACE_DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"
DEFAULT_FACE_RECOGNIZER_MODEL = "face_recognition_sface_2021dec.onnx"
FACE_MATCH_THRESHOLD = 0.363
FUZZY_FACE_MATCH_THRESHOLD = 0.28
FACE_DETECT_SCORE_THRESHOLD = 0.90
FACE_DETECT_NMS_THRESHOLD = 0.30
FACE_DETECT_TOP_K = 5000
FACE_DETECT_INPUT_SIZE = (320, 320)
SMALL_FACE_EDGE_PX = 80
MIN_FUZZY_FACE_EDGE_PX = 36


@dataclass(frozen=True)
class DetectedFace:
    raw: np.ndarray

    @property
    def x(self) -> int:
        return int(round(float(self.raw[0])))

    @property
    def y(self) -> int:
        return int(round(float(self.raw[1])))

    @property
    def w(self) -> int:
        return int(round(float(self.raw[2])))

    @property
    def h(self) -> int:
        return int(round(float(self.raw[3])))

    @property
    def score(self) -> float:
        return float(self.raw[14])

    @property
    def landmarks(self) -> np.ndarray:
        return np.asarray(self.raw[4:14], dtype=np.float32).reshape(5, 2)

    @property
    def center_x(self) -> float:
        return self.x + (self.w / 2.0)

    @property
    def center_y(self) -> float:
        return self.y + (self.h / 2.0)

    @property
    def min_edge(self) -> int:
        return min(self.w, self.h)

    def extract_roi(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        height, width = frame_bgr.shape[:2]
        left = max(0, self.x)
        top = max(0, self.y)
        right = min(width, self.x + self.w)
        bottom = min(height, self.y + self.h)
        if right <= left or bottom <= top:
            return None
        return frame_bgr[top:bottom, left:right].copy()


@dataclass(frozen=True)
class ReferenceGallery:
    person_name: str
    reference_dir: Path
    embeddings: list[np.ndarray]
    image_paths: list[Path]

    @property
    def count(self) -> int:
        return len(self.image_paths)


@dataclass(frozen=True)
class TargetFaceMatch:
    person_name: str
    face: DetectedFace
    similarity: float
    reference_index: int
    reference_path: Path
    match_quality: str
    similarity_threshold: float


class TargetFaceMatcher:
    def __init__(
        self,
        model_dir: Path,
        *,
        face_detector_model: str = DEFAULT_FACE_DETECTOR_MODEL,
        face_recognizer_model: str = DEFAULT_FACE_RECOGNIZER_MODEL,
        detect_score_threshold: float = FACE_DETECT_SCORE_THRESHOLD,
        detect_nms_threshold: float = FACE_DETECT_NMS_THRESHOLD,
        top_k: int = FACE_DETECT_TOP_K,
        cosine_threshold: float = FACE_MATCH_THRESHOLD,
        fuzzy_cosine_threshold: float = FUZZY_FACE_MATCH_THRESHOLD,
        small_face_edge_px: int = SMALL_FACE_EDGE_PX,
        minimum_fuzzy_face_edge_px: int = MIN_FUZZY_FACE_EDGE_PX,
    ) -> None:
        detector_path = Path(model_dir) / face_detector_model
        recognizer_path = Path(model_dir) / face_recognizer_model

        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("Current OpenCV build does not expose cv2.FaceDetectorYN.")
        if not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError("Current OpenCV build does not expose cv2.FaceRecognizerSF.")
        if not detector_path.exists():
            raise FileNotFoundError(f"Missing face detector model: {detector_path}")
        if not recognizer_path.exists():
            raise FileNotFoundError(f"Missing face recognizer model: {recognizer_path}")

        self.detector = cv2.FaceDetectorYN.create(
            str(detector_path),
            "",
            FACE_DETECT_INPUT_SIZE,
            float(detect_score_threshold),
            float(detect_nms_threshold),
            int(top_k),
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
        self.cosine_threshold = float(cosine_threshold)
        self.fuzzy_cosine_threshold = float(fuzzy_cosine_threshold)
        self.small_face_edge_px = int(small_face_edge_px)
        self.minimum_fuzzy_face_edge_px = int(minimum_fuzzy_face_edge_px)

    def _classify_match(self, face: DetectedFace, similarity: float) -> tuple[str, float] | None:
        if similarity >= self.cosine_threshold:
            return "strict", self.cosine_threshold

        is_small_face = face.min_edge < self.small_face_edge_px
        is_fuzzy_match = (
            is_small_face
            and face.min_edge >= self.minimum_fuzzy_face_edge_px
            and similarity >= self.fuzzy_cosine_threshold
        )
        if is_fuzzy_match:
            return "fuzzy", self.fuzzy_cosine_threshold

        return None

    def detect_faces(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        frame_height, frame_width = frame_bgr.shape[:2]
        self.detector.setInputSize((int(frame_width), int(frame_height)))
        _retval, detections = self.detector.detect(frame_bgr)
        if detections is None or len(detections) == 0:
            return []

        faces = [DetectedFace(np.asarray(row, dtype=np.float32).reshape(-1)) for row in detections]
        faces.sort(key=lambda face: face.score, reverse=True)
        return faces

    def extract_feature(self, frame_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        aligned_face = self.recognizer.alignCrop(frame_bgr, face.raw)
        feature = self.recognizer.feature(aligned_face)
        return np.asarray(feature, dtype=np.float32).reshape(-1)

    def compare_features(self, lhs_feature: np.ndarray, rhs_feature: np.ndarray) -> float:
        return float(self.recognizer.match(lhs_feature, rhs_feature, cv2.FaceRecognizerSF_FR_COSINE))

    def load_reference_gallery(self, reference_root: Path, person_name: str | None = None) -> ReferenceGallery:
        if person_name is None:
            reference_dir = Path(reference_root)
            person_name = reference_dir.name
            reference_root = reference_dir.parent
        else:
            reference_root = Path(reference_root)
            reference_dir = reference_root / person_name

        if not reference_dir.exists():
            available_people = list_available_person_names(reference_root)
            available_text = ", ".join(available_people) if available_people else "(none)"
            raise FileNotFoundError(
                f"Target person '{person_name}' does not exist under {reference_root}. "
                f"Available people: {available_text}"
            )

        image_paths = iter_image_files(reference_dir)
        if not image_paths:
            raise RuntimeError(f"No reference images found under {reference_dir}")

        embeddings: list[np.ndarray] = []
        validated_paths: list[Path] = []
        for image_path in image_paths:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise RuntimeError(f"Failed to read reference image: {image_path}")

            faces = self.detect_faces(image_bgr)
            if len(faces) != 1:
                raise RuntimeError(
                    f"Reference image {image_path.name} must contain exactly one detectable face, found {len(faces)}"
                )

            embeddings.append(self.extract_feature(image_bgr, faces[0]))
            validated_paths.append(image_path)

        return ReferenceGallery(
            person_name=person_name,
            reference_dir=reference_dir.resolve(),
            embeddings=embeddings,
            image_paths=validated_paths,
        )

    def match_target_face(self, frame_bgr: np.ndarray, gallery: ReferenceGallery) -> TargetFaceMatch | None:
        best_match: TargetFaceMatch | None = None
        for face in self.detect_faces(frame_bgr):
            probe_feature = self.extract_feature(frame_bgr, face)
            for reference_index, (reference_feature, reference_path) in enumerate(
                zip(gallery.embeddings, gallery.image_paths, strict=True)
            ):
                similarity = self.compare_features(probe_feature, reference_feature)
                match_metadata = self._classify_match(face, similarity)
                if match_metadata is None:
                    continue
                match_quality, similarity_threshold = match_metadata

                if best_match is None or similarity > best_match.similarity:
                    best_match = TargetFaceMatch(
                        person_name=gallery.person_name,
                        face=face,
                        similarity=float(similarity),
                        reference_index=reference_index,
                        reference_path=reference_path,
                        match_quality=match_quality,
                        similarity_threshold=float(similarity_threshold),
                    )

        return best_match
