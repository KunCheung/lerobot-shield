from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "shield" / "2_move_to_target_person"
LOCAL_TMP_ROOT = Path(__file__).resolve().parents[2] / "tmp_test_2_move_to_target_person"


def load_example_module(module_name: str):
    path = EXAMPLE_DIR / f"{module_name}.py"
    if str(EXAMPLE_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLE_DIR))

    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_local_tmp_path() -> Path:
    LOCAL_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_path = LOCAL_TMP_ROOT / f"case_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=False)
    return tmp_path


runtime_common = load_example_module("runtime_common")
target_face_matcher = load_example_module("target_face_matcher")
follow_logic = load_example_module("follow_logic")
people_detection = load_example_module("people_detection")
capture_script = load_example_module("capture_target_person_reference")
follow_script = load_example_module("move_to_target_person_by_cv")


class FakeRecognizer:
    def match(self, lhs_feature, rhs_feature, metric):
        _ = metric
        lhs = np.asarray(lhs_feature, dtype=np.float32).reshape(-1)
        rhs = np.asarray(rhs_feature, dtype=np.float32).reshape(-1)
        return float(np.dot(lhs, rhs))


def make_stub_matcher() -> target_face_matcher.TargetFaceMatcher:
    matcher = object.__new__(target_face_matcher.TargetFaceMatcher)
    matcher.recognizer = FakeRecognizer()
    matcher.cosine_threshold = target_face_matcher.FACE_MATCH_THRESHOLD
    matcher.fuzzy_cosine_threshold = target_face_matcher.FUZZY_FACE_MATCH_THRESHOLD
    matcher.small_face_edge_px = target_face_matcher.SMALL_FACE_EDGE_PX
    matcher.minimum_fuzzy_face_edge_px = target_face_matcher.MIN_FUZZY_FACE_EDGE_PX
    return matcher


def make_face(matcher_module, *, x: float = 0, y: float = 0, w: float = 100, h: float = 100, score: float = 0.99):
    raw = np.array(
        [
            x,
            y,
            w,
            h,
            x + 30,
            y + 35,
            x + 70,
            y + 35,
            x + 50,
            y + 55,
            x + 35,
            y + 75,
            x + 65,
            y + 75,
            score,
        ],
        dtype=np.float32,
    )
    return matcher_module.DetectedFace(raw=raw)


def test_validate_person_name_trims_whitespace() -> None:
    assert runtime_common.validate_person_name(" alice ") == "alice"


@pytest.mark.parametrize("bad_name", ["", "   ", "..", "a/b", "a\\b", "archive"])
def test_validate_person_name_rejects_invalid_values(bad_name: str) -> None:
    with pytest.raises(ValueError):
        runtime_common.validate_person_name(bad_name)


def test_archive_person_directory_only_moves_requested_person(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = make_local_tmp_path()
    reference_root = tmp_path / "reference_person"
    alice_dir = reference_root / "alice"
    bob_dir = reference_root / "bob"
    alice_dir.mkdir(parents=True)
    bob_dir.mkdir(parents=True)
    (alice_dir / "target_001.jpg").write_bytes(b"alice")
    (bob_dir / "target_001.jpg").write_bytes(b"bob")
    move_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        runtime_common.shutil,
        "move",
        lambda src, dst: move_calls.append((src, dst)),
    )

    archived = runtime_common.archive_person_directory(
        alice_dir,
        reference_root / "archive",
        timestamp="20260413_120000",
    )

    assert archived == reference_root / "archive" / "alice" / "20260413_120000"
    assert move_calls == [(str(alice_dir), str(archived))]
    assert (bob_dir / "target_001.jpg").exists()


def test_load_reference_gallery_missing_person_lists_available_people() -> None:
    matcher = make_stub_matcher()

    tmp_path = make_local_tmp_path()
    reference_root = tmp_path / "reference_person"
    (reference_root / "alice").mkdir(parents=True)
    (reference_root / "bob").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Available people: alice, bob"):
        matcher.load_reference_gallery(reference_root, "carol")


def test_load_reference_gallery_uses_only_selected_person_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    matcher = make_stub_matcher()

    tmp_path = make_local_tmp_path()
    reference_root = tmp_path / "reference_person"
    alice_dir = reference_root / "alice"
    bob_dir = reference_root / "bob"
    alice_dir.mkdir(parents=True)
    bob_dir.mkdir(parents=True)

    alice_image = np.full((20, 20, 3), 10, dtype=np.uint8)
    bob_image = np.full((20, 20, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(alice_dir / "target_001.png"), alice_image)
    cv2.imwrite(str(bob_dir / "target_001.png"), bob_image)

    monkeypatch.setattr(
        target_face_matcher.TargetFaceMatcher,
        "detect_faces",
        lambda self, frame_bgr: [make_face(target_face_matcher, score=0.95)],
    )
    monkeypatch.setattr(
        target_face_matcher.TargetFaceMatcher,
        "extract_feature",
        lambda self, frame_bgr, face: np.array([1.0, 0.0], dtype=np.float32)
        if int(frame_bgr.mean()) < 100
        else np.array([0.0, 1.0], dtype=np.float32),
    )

    gallery = matcher.load_reference_gallery(reference_root, "alice")

    assert gallery.person_name == "alice"
    assert gallery.count == 1
    assert all(path.parent.name == "alice" for path in gallery.image_paths)


def test_load_reference_gallery_rejects_reference_image_with_multiple_faces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matcher = make_stub_matcher()

    tmp_path = make_local_tmp_path()
    reference_dir = tmp_path / "reference_person" / "alice"
    reference_dir.mkdir(parents=True)
    cv2.imwrite(str(reference_dir / "target_001.png"), np.full((32, 32, 3), 30, dtype=np.uint8))

    monkeypatch.setattr(
        target_face_matcher.TargetFaceMatcher,
        "detect_faces",
        lambda self, frame_bgr: [
            make_face(target_face_matcher, x=0, score=0.98),
            make_face(target_face_matcher, x=120, score=0.97),
        ],
    )

    with pytest.raises(RuntimeError, match="exactly one detectable face"):
        matcher.load_reference_gallery(reference_dir)


def test_match_target_face_returns_best_face_above_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    matcher = make_stub_matcher()

    face_left = make_face(target_face_matcher, x=20, score=0.98)
    face_right = make_face(target_face_matcher, x=220, score=0.99)
    gallery = target_face_matcher.ReferenceGallery(
        person_name="alice",
        reference_dir=Path("reference_person/alice"),
        embeddings=[np.array([1.0, 0.0], dtype=np.float32)],
        image_paths=[Path("reference_person/alice/target_001.jpg")],
    )

    monkeypatch.setattr(
        target_face_matcher.TargetFaceMatcher,
        "detect_faces",
        lambda self, frame_bgr: [face_left, face_right],
    )
    monkeypatch.setattr(
        target_face_matcher.TargetFaceMatcher,
        "extract_feature",
        lambda self, frame_bgr, face: np.array([0.2, 0.1], dtype=np.float32)
        if face.x < 100
        else np.array([0.9, 0.1], dtype=np.float32),
    )

    match = matcher.match_target_face(np.zeros((200, 300, 3), dtype=np.uint8), gallery)

    assert match is not None
    assert match.face.x == face_right.x
    assert match.similarity == pytest.approx(0.9)
    assert match.match_quality == "strict"
    assert match.similarity_threshold == pytest.approx(target_face_matcher.FACE_MATCH_THRESHOLD)


def test_match_target_face_allows_fuzzy_match_for_small_face(monkeypatch: pytest.MonkeyPatch) -> None:
    matcher = make_stub_matcher()
    small_face = make_face(target_face_matcher, x=40, y=50, w=48, h=52, score=0.94)
    gallery = target_face_matcher.ReferenceGallery(
        person_name="alice",
        reference_dir=Path("reference_person/alice"),
        embeddings=[np.array([1.0, 0.0], dtype=np.float32)],
        image_paths=[Path("reference_person/alice/target_001.jpg")],
    )

    monkeypatch.setattr(
        target_face_matcher.TargetFaceMatcher,
        "detect_faces",
        lambda self, frame_bgr: [small_face],
    )
    monkeypatch.setattr(
        target_face_matcher.TargetFaceMatcher,
        "extract_feature",
        lambda self, frame_bgr, face: np.array([0.31, 0.0], dtype=np.float32),
    )

    match = matcher.match_target_face(np.zeros((240, 320, 3), dtype=np.uint8), gallery)

    assert match is not None
    assert match.match_quality == "fuzzy"
    assert match.similarity == pytest.approx(0.31)
    assert match.similarity_threshold == pytest.approx(target_face_matcher.FUZZY_FACE_MATCH_THRESHOLD)


def test_match_target_face_rejects_large_face_below_strict_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    matcher = make_stub_matcher()
    large_face = make_face(target_face_matcher, x=40, y=50, w=120, h=120, score=0.96)
    gallery = target_face_matcher.ReferenceGallery(
        person_name="alice",
        reference_dir=Path("reference_person/alice"),
        embeddings=[np.array([1.0, 0.0], dtype=np.float32)],
        image_paths=[Path("reference_person/alice/target_001.jpg")],
    )

    monkeypatch.setattr(
        target_face_matcher.TargetFaceMatcher,
        "detect_faces",
        lambda self, frame_bgr: [large_face],
    )
    monkeypatch.setattr(
        target_face_matcher.TargetFaceMatcher,
        "extract_feature",
        lambda self, frame_bgr, face: np.array([0.31, 0.0], dtype=np.float32),
    )

    match = matcher.match_target_face(np.zeros((240, 320, 3), dtype=np.uint8), gallery)

    assert match is None


def test_select_enrollable_face_rejects_multiple_faces() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    face1 = make_face(target_face_matcher, x=10, y=10)
    face2 = make_face(target_face_matcher, x=110, y=10)

    selection = capture_script.select_enrollable_face(
        frame,
        [face1, face2],
        min_face_size_px=80,
        min_face_score=0.90,
        min_blur_score=80.0,
    )

    assert selection.face is None
    assert selection.blur_score == 0.0
    assert not selection.can_save
    assert "only one person" in selection.status_message


def test_select_enrollable_face_rejects_small_face() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    small_face = make_face(target_face_matcher, x=30, y=30, w=40, h=40)

    selection = capture_script.select_enrollable_face(
        frame,
        [small_face],
        min_face_size_px=80,
        min_face_score=0.90,
        min_blur_score=80.0,
    )

    assert selection.face is None
    assert "too small" in selection.status_message


def test_select_enrollable_face_rejects_blurry_face() -> None:
    frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    face = make_face(target_face_matcher, x=20, y=20, w=120, h=120)

    selection = capture_script.select_enrollable_face(
        frame,
        [face],
        min_face_size_px=80,
        min_face_score=0.90,
        min_blur_score=80.0,
    )

    assert selection.face is None
    assert selection.blur_score < 80.0
    assert "blurry" in selection.status_message


def test_select_enrollable_face_returns_selection_object_when_ready() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    face = make_face(target_face_matcher, x=20, y=20, w=120, h=120)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(capture_script, "compute_face_blur_score", lambda frame_bgr, detected_face: 120.0)
    try:
        selection = capture_script.select_enrollable_face(
            frame,
            [face],
            min_face_size_px=80,
            min_face_score=0.90,
            min_blur_score=80.0,
        )
    finally:
        monkeypatch.undo()

    assert selection.face == face
    assert selection.blur_score == pytest.approx(120.0)
    assert selection.can_save
    assert selection.status_message == "Ready to save."


def test_select_target_detection_prefers_person_box_containing_face() -> None:
    person_left = follow_script.PersonDetection(0, 0, 120, 220, 0.4, 640, 480, source="hog")
    person_right = follow_script.PersonDetection(220, 0, 160, 260, 0.8, 640, 480, source="hog")
    match = target_face_matcher.TargetFaceMatch(
        person_name="alice",
        face=make_face(target_face_matcher, x=250, y=60, w=60, h=60),
        similarity=0.88,
        reference_index=0,
        reference_path=Path("reference_person/alice/target_001.jpg"),
        match_quality="strict",
        similarity_threshold=target_face_matcher.FACE_MATCH_THRESHOLD,
    )

    detection = follow_script.select_target_detection(
        [person_left, person_right],
        match,
        frame_width=640,
        frame_height=480,
    )

    assert detection is not None
    assert detection.x == person_right.x
    assert detection.identity_score == pytest.approx(0.88)
    assert detection.identity_source == "face_id_strict"


def test_select_target_detection_falls_back_to_face_expansion() -> None:
    match = target_face_matcher.TargetFaceMatch(
        person_name="alice",
        face=make_face(target_face_matcher, x=200, y=100, w=50, h=60),
        similarity=0.91,
        reference_index=0,
        reference_path=Path("reference_person/alice/target_001.jpg"),
        match_quality="strict",
        similarity_threshold=target_face_matcher.FACE_MATCH_THRESHOLD,
    )

    detection = follow_script.select_target_detection(
        [],
        match,
        frame_width=640,
        frame_height=480,
    )

    assert detection is not None
    assert detection.source == "face_id_strict_fallback"
    assert detection.h > match.face.h
    assert detection.identity_score == pytest.approx(0.91)
    assert detection.identity_source == "face_id_strict_fallback"


def test_people_detection_face_expansion_matches_target_fallback_geometry() -> None:
    match = target_face_matcher.TargetFaceMatch(
        person_name="alice",
        face=make_face(target_face_matcher, x=200, y=100, w=50, h=60),
        similarity=0.91,
        reference_index=0,
        reference_path=Path("reference_person/alice/target_001.jpg"),
        match_quality="strict",
        similarity_threshold=target_face_matcher.FACE_MATCH_THRESHOLD,
    )

    expanded = people_detection.expand_face_to_body_box(
        match.face.x,
        match.face.y,
        match.face.w,
        match.face.h,
        640,
        480,
    )
    detection = follow_script.select_target_detection([], match, frame_width=640, frame_height=480)

    assert expanded is not None
    assert detection is not None
    assert expanded == (detection.x, detection.y, detection.w, detection.h)


def test_decide_next_action_waits_longer_for_fuzzy_identity() -> None:
    state = follow_script.TrackerState()
    fuzzy_detection = follow_script.PersonDetection(
        x=270,
        y=100,
        w=100,
        h=80,
        score=0.9,
        frame_width=640,
        frame_height=480,
        source="hog",
        identity_score=0.31,
        identity_source="face_id_fuzzy",
    )

    for _ in range(follow_script.FUZZY_MIN_CONSECUTIVE_HITS - 1):
        state.record_detection(fuzzy_detection)

    action = follow_script.decide_next_action(state, frame_width=640)

    assert action.kind == "wait"
    assert f"{follow_script.FUZZY_MIN_CONSECUTIVE_HITS - 1}/{follow_script.FUZZY_MIN_CONSECUTIVE_HITS}" in action.reason


def test_decide_next_action_uses_cautious_forward_for_fuzzy_identity() -> None:
    state = follow_script.TrackerState()
    fuzzy_detection = follow_script.PersonDetection(
        x=270,
        y=100,
        w=100,
        h=80,
        score=0.9,
        frame_width=640,
        frame_height=480,
        source="hog",
        identity_score=0.31,
        identity_source="face_id_fuzzy",
    )

    for _ in range(follow_script.FUZZY_MIN_CONSECUTIVE_HITS):
        state.record_detection(fuzzy_detection)

    action = follow_script.decide_next_action(state, frame_width=640)

    assert action.kind == "forward"
    assert action.value == pytest.approx(follow_script.FORWARD_STEP_FUZZY_M)
    assert "fuzzy match" in action.reason


def test_decide_next_action_does_not_move_forward_when_target_is_missing() -> None:
    state = follow_script.TrackerState()
    state.record_miss()

    action = follow_script.decide_next_action(state, frame_width=640)

    assert action.kind != "forward"
