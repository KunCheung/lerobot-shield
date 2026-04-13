from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "shield" / "move_to_target_person"


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


follow_logic = load_example_module("follow_logic")


def make_detection(*, x: int, frame_width: int = 640, frame_height: int = 480):
    return follow_logic.PersonDetection(
        x=x,
        y=80,
        w=100,
        h=180,
        score=0.9,
        frame_width=frame_width,
        frame_height=frame_height,
        source="hog",
    )


def advance_to_search_mode(state) -> None:
    for _ in range(follow_logic.MISS_LIMIT):
        state.record_miss()


def test_search_budget_covers_full_360_degree_sweep() -> None:
    assert follow_logic.MAX_SEARCH_TURNS == math.ceil(
        follow_logic.SEARCH_SWEEP_DEG / follow_logic.SEARCH_TURN_DEG
    )
    assert follow_logic.MAX_SEARCH_TURNS == 24


def test_search_uses_last_seen_left_direction() -> None:
    state = follow_logic.TrackerState()
    state.record_detection(make_detection(x=40))
    advance_to_search_mode(state)

    first_action = follow_logic.decide_next_action(state, frame_width=640)
    assert first_action.kind == "turn_left"

    state.record_search_step(first_action.kind, first_action.value)
    second_action = follow_logic.decide_next_action(state, frame_width=640)
    assert second_action.kind == "turn_left"


def test_search_uses_last_seen_right_direction() -> None:
    state = follow_logic.TrackerState()
    state.record_detection(make_detection(x=500))
    advance_to_search_mode(state)

    first_action = follow_logic.decide_next_action(state, frame_width=640)
    assert first_action.kind == "turn_right"

    state.record_search_step(first_action.kind, first_action.value)
    second_action = follow_logic.decide_next_action(state, frame_width=640)
    assert second_action.kind == "turn_right"


def test_search_defaults_to_left_when_no_prior_direction_exists() -> None:
    state = follow_logic.TrackerState()
    advance_to_search_mode(state)

    action = follow_logic.decide_next_action(state, frame_width=640)

    assert action.kind == "turn_left"
    assert "search left" in action.reason


def test_search_exits_after_full_sweep_without_target() -> None:
    state = follow_logic.TrackerState()
    advance_to_search_mode(state)
    for _ in range(follow_logic.MAX_SEARCH_TURNS):
        action = follow_logic.decide_next_action(state, frame_width=640)
        assert action.kind == "turn_left"
        state.record_search_step(action.kind, action.value)

    final_action = follow_logic.decide_next_action(state, frame_width=640)

    assert final_action.kind == "exit"
    assert "360/360deg" in final_action.reason


def test_record_detection_resets_search_progress_after_reacquire() -> None:
    state = follow_logic.TrackerState()
    advance_to_search_mode(state)
    action = follow_logic.decide_next_action(state, frame_width=640)
    state.record_search_step(action.kind, action.value)

    assert state.search_turns == 1
    assert state.searched_degrees == pytest.approx(follow_logic.SEARCH_TURN_DEG)

    state.record_detection(make_detection(x=500))

    assert state.search_turns == 0
    assert state.searched_degrees == pytest.approx(0.0)
    assert state.preferred_search_direction() == "right"
