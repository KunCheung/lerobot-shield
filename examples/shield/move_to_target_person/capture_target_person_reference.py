from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from runtime_common import (
    DEFAULT_CAMERA_ID,
    SCRIPT_DIR,
    archive_person_directory,
    assert_local_lerobot_source,
    build_camera,
    current_timestamp,
    ensure_directory,
    prompt_person_name,
    write_debug_image,
    write_debug_text,
)
from target_face_matcher import DetectedFace, TargetFaceMatcher


DEFAULT_REFERENCE_ROOT = SCRIPT_DIR / "reference_person"
DEFAULT_ARCHIVE_ROOT = DEFAULT_REFERENCE_ROOT / "archive"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "models"
DEFAULT_DEBUG_DIR = SCRIPT_DIR / "debug"
WINDOW_NAME = "Target Person Enrollment"
DEFAULT_MAX_IMAGES = 5
DEFAULT_MIN_IMAGES = 3
DEFAULT_MIN_FACE_SIZE_PX = 80
DEFAULT_MIN_FACE_SCORE = 0.90
DEFAULT_MIN_BLUR_SCORE = 80.0


@dataclass(frozen=True)
class EnrollmentSelection:
    face: DetectedFace | None
    blur_score: float
    status_message: str

    @property
    def can_save(self) -> bool:
        return self.face is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture target-person reference images using the current head camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera-id", type=int, default=DEFAULT_CAMERA_ID)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES)
    parser.add_argument("--min-images", type=int, default=DEFAULT_MIN_IMAGES)
    parser.add_argument("--min-face-size", type=int, default=DEFAULT_MIN_FACE_SIZE_PX)
    parser.add_argument("--min-face-score", type=float, default=DEFAULT_MIN_FACE_SCORE)
    parser.add_argument("--min-blur-score", type=float, default=DEFAULT_MIN_BLUR_SCORE)
    return parser.parse_args()


def compute_face_blur_score(frame_bgr: np.ndarray, face: DetectedFace) -> float:
    roi = face.extract_roi(frame_bgr)
    if roi is None or roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_enrollable_face(
    frame_bgr: np.ndarray,
    faces: list[DetectedFace],
    *,
    min_face_size_px: int,
    min_face_score: float,
    min_blur_score: float,
) -> EnrollmentSelection:
    if not faces:
        return EnrollmentSelection(face=None, blur_score=0.0, status_message="No face detected.")
    if len(faces) > 1:
        return EnrollmentSelection(
            face=None,
            blur_score=0.0,
            status_message=f"Detected {len(faces)} faces. Please keep only one person in view.",
        )

    face = faces[0]
    if face.score < min_face_score:
        return EnrollmentSelection(
            face=None,
            blur_score=0.0,
            status_message=f"Face confidence {face.score:.2f} is below {min_face_score:.2f}.",
        )
    if face.w < min_face_size_px or face.h < min_face_size_px:
        return EnrollmentSelection(
            face=None,
            blur_score=0.0,
            status_message=f"Face is too small ({face.w}x{face.h}). Need at least {min_face_size_px}px.",
        )

    blur_score = compute_face_blur_score(frame_bgr, face)
    if blur_score < min_blur_score:
        return EnrollmentSelection(
            face=None,
            blur_score=blur_score,
            status_message=f"Face is too blurry ({blur_score:.1f} < {min_blur_score:.1f}).",
        )

    return EnrollmentSelection(face=face, blur_score=blur_score, status_message="Ready to save.")


def draw_overlay(
    frame_bgr: np.ndarray,
    *,
    person_name: str,
    faces: list[DetectedFace],
    selected_face: DetectedFace | None,
    blur_score: float,
    saved_count: int,
    max_images: int,
    min_images: int,
    status_message: str,
) -> np.ndarray:
    annotated = frame_bgr.copy()

    for face in faces:
        color = (0, 220, 0) if selected_face is not None and face == selected_face else (0, 140, 255)
        cv2.rectangle(annotated, (face.x, face.y), (face.x + face.w, face.y + face.h), color, 2)
        cv2.putText(
            annotated,
            f"score={face.score:.2f}",
            (face.x, max(18, face.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    lines = [
        f"target_name={person_name}",
        f"saved={saved_count}/{max_images} min={min_images}",
        f"faces={len(faces)} blur={blur_score:.1f}",
        status_message,
        "s: save current frame",
        "q or ESC: quit",
    ]

    y = 24
    for line in lines:
        cv2.putText(annotated, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24

    return annotated


def save_reference_frame(person_dir: Path, frame_bgr: np.ndarray, image_index: int) -> Path:
    person_dir.mkdir(parents=True, exist_ok=True)
    image_path = person_dir / f"target_{image_index:03d}.jpg"
    cv2.imwrite(str(image_path), frame_bgr)
    return image_path


def write_capture_skip_debug(
    debug_dir: Path,
    *,
    person_name: str,
    selection: EnrollmentSelection,
) -> None:
    write_debug_text(
        debug_dir / "last_capture_status.txt",
        (
            f"person_name={person_name}\n"
            f"status={selection.status_message}\n"
            f"blur_score={selection.blur_score:.2f}\n"
        ),
    )


def write_capture_success_debug(
    debug_dir: Path,
    *,
    person_name: str,
    selection: EnrollmentSelection,
    frame_bgr: np.ndarray,
    image_path: Path,
    saved_count: int,
) -> None:
    if selection.face is None:
        return

    write_debug_image(debug_dir / "last_capture_frame.jpg", frame_bgr)
    write_debug_image(debug_dir / "last_capture_face.jpg", selection.face.extract_roi(frame_bgr))
    write_debug_text(
        debug_dir / "last_capture_status.txt",
        "\n".join(
            [
                f"person_name={person_name}",
                f"image_path={image_path}",
                f"saved_count={saved_count}",
                f"face_score={selection.face.score:.3f}",
                f"blur_score={selection.blur_score:.3f}",
                "status=saved",
            ]
        )
        + "\n",
    )


def main() -> None:
    args = parse_args()
    assert_local_lerobot_source("capture_target_person_reference.py")

    person_name = prompt_person_name("Person name to enroll: ")
    reference_root = ensure_directory(args.reference_root)
    archive_root = ensure_directory(args.archive_root)
    debug_dir = ensure_directory(args.debug_dir)
    person_dir = reference_root / person_name

    archived_to = archive_person_directory(person_dir, archive_root, timestamp=current_timestamp())
    if archived_to is not None:
        print(f"[Archive] Moved previous reference images to: {archived_to}")

    matcher = TargetFaceMatcher(args.model_dir, detect_score_threshold=args.min_face_score)
    camera = None
    saved_count = 0

    try:
        camera = build_camera(args.camera_id)
        print(f"[Target] person_name={person_name}")
        print(f"[Target] reference_dir={person_dir}")
        print(f"[Target] debug_dir={debug_dir}")

        while True:
            try:
                frame_bgr = camera.read_bgr()
            except RuntimeError as exc:
                print(f"[Camera] Read failed: {exc}")
                time.sleep(0.3)
                camera.reopen()
                continue

            faces = matcher.detect_faces(frame_bgr)
            selection = select_enrollable_face(
                frame_bgr,
                faces,
                min_face_size_px=args.min_face_size,
                min_face_score=args.min_face_score,
                min_blur_score=args.min_blur_score,
            )

            overlay = draw_overlay(
                frame_bgr,
                person_name=person_name,
                faces=faces,
                selected_face=selection.face,
                blur_score=selection.blur_score,
                saved_count=saved_count,
                max_images=args.max_images,
                min_images=args.min_images,
                status_message=selection.status_message,
            )
            cv2.imshow(WINDOW_NAME, overlay)

            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                if saved_count < args.min_images:
                    print(
                        f"[Exit] Stopped with only {saved_count} saved images. "
                        f"Recommended minimum is {args.min_images}."
                    )
                else:
                    print(f"[Exit] Enrollment finished with {saved_count} saved images.")
                break

            if key == ord("s"):
                if not selection.can_save:
                    write_capture_skip_debug(debug_dir, person_name=person_name, selection=selection)
                    print(f"[Skip] {selection.status_message}")
                    continue

                saved_count += 1
                image_path = save_reference_frame(person_dir, frame_bgr, saved_count)
                write_capture_success_debug(
                    debug_dir,
                    person_name=person_name,
                    selection=selection,
                    frame_bgr=frame_bgr,
                    image_path=image_path,
                    saved_count=saved_count,
                )
                print(f"[Saved] {image_path}")

                if saved_count >= args.max_images:
                    print(f"[Done] Captured {saved_count} reference images for {person_name}.")
                    break
    finally:
        cv2.destroyAllWindows()
        if camera is not None:
            camera.disconnect()


if __name__ == "__main__":
    main()
