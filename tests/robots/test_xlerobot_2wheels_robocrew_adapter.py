#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import pytest

import lerobot.robots.xlerobot_2wheels.robocrew_adapter as robocrew_adapter_module


class FakeRobot:
    def __init__(self, *, connected: bool = True) -> None:
        self.is_connected = connected
        self.actions: list[dict[str, float]] = []
        self.stop_base_calls = 0
        self.disconnect_calls = 0

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.actions.append(dict(action))
        return action

    def stop_base(self) -> None:
        self.stop_base_calls += 1

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False


def make_adapter(monkeypatch: pytest.MonkeyPatch, *, connected: bool = True):
    sleep_calls: list[float] = []
    monkeypatch.setattr(robocrew_adapter_module.time, "sleep", lambda duration: sleep_calls.append(duration))
    robot = FakeRobot(connected=connected)
    adapter = robocrew_adapter_module.TwoWheelsServoAdapter(
        robot,
        right_arm_wheel_usb="COM4",
        linear_speed_mps=0.1,
        angular_speed_dps=30.0,
    )
    return adapter, robot, sleep_calls


def test_go_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, robot, sleep_calls = make_adapter(monkeypatch)

    adapter.go_forward(0.5)

    assert robot.actions == [{"x.vel": 0.1, "theta.vel": 0.0}]
    assert sleep_calls == [5.0]
    assert robot.stop_base_calls == 1


def test_go_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, robot, sleep_calls = make_adapter(monkeypatch)

    adapter.go_backward(0.5)

    assert robot.actions == [{"x.vel": -0.1, "theta.vel": 0.0}]
    assert sleep_calls == [5.0]
    assert robot.stop_base_calls == 1


def test_turn_left(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, robot, sleep_calls = make_adapter(monkeypatch)

    adapter.turn_left(90)

    assert robot.actions == [{"x.vel": 0.0, "theta.vel": 30.0}]
    assert sleep_calls == [3.0]
    assert robot.stop_base_calls == 1


def test_turn_right(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, robot, sleep_calls = make_adapter(monkeypatch)

    adapter.turn_right(90)

    assert robot.actions == [{"x.vel": 0.0, "theta.vel": -30.0}]
    assert sleep_calls == [3.0]
    assert robot.stop_base_calls == 1


def test_zero_inputs_do_not_send_motion(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, robot, sleep_calls = make_adapter(monkeypatch)

    adapter.go_forward(0)
    adapter.turn_left(0)

    assert robot.actions == []
    assert sleep_calls == []
    assert robot.stop_base_calls == 0


def test_negative_inputs_are_normalized_by_abs(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, robot, sleep_calls = make_adapter(monkeypatch)

    adapter.go_forward(-0.5)
    adapter.turn_right(-90)

    assert robot.actions == [
        {"x.vel": 0.1, "theta.vel": 0.0},
        {"x.vel": 0.0, "theta.vel": -30.0},
    ]
    assert sleep_calls == [5.0, 3.0]
    assert robot.stop_base_calls == 2


def test_disconnect_stops_and_disconnects_connected_robot(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, robot, sleep_calls = make_adapter(monkeypatch, connected=True)

    adapter.disconnect()

    assert sleep_calls == []
    assert robot.stop_base_calls == 1
    assert robot.disconnect_calls == 1
    assert robot.is_connected is False


def test_disconnect_is_safe_when_robot_is_already_disconnected(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, robot, sleep_calls = make_adapter(monkeypatch, connected=False)

    adapter.disconnect()

    assert sleep_calls == []
    assert robot.stop_base_calls == 1
    assert robot.disconnect_calls == 0
    assert robot.is_connected is False
