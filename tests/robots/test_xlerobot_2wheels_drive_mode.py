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

from types import SimpleNamespace

import pytest

from lerobot.motors import MotorCalibration
from lerobot.robots.xlerobot_2wheels.xlerobot_2wheels import XLerobot2Wheels


class FakeBus:
    def __init__(self) -> None:
        self.is_connected = True
        self.sync_write_calls: list[tuple[str, dict[str, float], dict[str, float]]] = []
        self.sync_read_values: dict[tuple[str, tuple[str, ...]], dict[str, float]] = {}

    def sync_write(self, data_name: str, values: dict[str, float], **kwargs) -> None:
        self.sync_write_calls.append((data_name, dict(values), dict(kwargs)))

    def sync_read(self, data_name: str, motors: list[str]) -> dict[str, float]:
        key = (data_name, tuple(motors))
        if key in self.sync_read_values:
            return dict(self.sync_read_values[key])
        return {motor: 0.0 for motor in motors}


def make_robot() -> XLerobot2Wheels:
    robot = XLerobot2Wheels.__new__(XLerobot2Wheels)
    robot.id = "test_xlerobot_2wheels"
    robot.config = SimpleNamespace(
        max_relative_target=None,
        wheel_radius=0.05,
        wheelbase=0.25,
        disable_torque_on_disconnect=True,
    )
    robot.calibration = {
        "base_left_wheel": MotorCalibration(id=9, drive_mode=1, homing_offset=0, range_min=0, range_max=4095),
        "base_right_wheel": MotorCalibration(
            id=10, drive_mode=0, homing_offset=0, range_min=0, range_max=4095
        ),
    }
    robot.bus1 = FakeBus()
    robot.bus2 = FakeBus()
    robot.left_arm_motors = []
    robot.right_arm_motors = []
    robot.head_motors = []
    robot.base_motors = ["base_left_wheel", "base_right_wheel"]
    robot.cameras = {}
    return robot


def test_apply_base_velocity_drive_mode_flips_only_inverted_wheel() -> None:
    robot = make_robot()

    corrected = robot._apply_base_velocity_drive_mode(
        {"base_left_wheel": 123, "base_right_wheel": -45}
    )

    assert corrected == {"base_left_wheel": -123, "base_right_wheel": -45}


def test_send_action_applies_drive_mode_to_goal_velocity() -> None:
    robot = make_robot()
    uncorrected = robot._body_to_wheel_raw(0.1, 0.0)

    sent_action = robot.send_action({"x.vel": 0.1, "theta.vel": 0.0})

    assert sent_action == {"x.vel": 0.1, "theta.vel": 0.0}
    assert robot.bus2.sync_write_calls == [
        (
            "Goal_Velocity",
            {
                "base_left_wheel": -uncorrected["base_left_wheel"],
                "base_right_wheel": uncorrected["base_right_wheel"],
            },
            {},
        )
    ]


def test_get_observation_restores_forward_velocity_semantics() -> None:
    robot = make_robot()
    corrected_feedback = robot._apply_base_velocity_drive_mode(robot._body_to_wheel_raw(0.1, 0.0))
    robot.bus2.sync_read_values[("Present_Velocity", tuple(robot.base_motors))] = corrected_feedback

    observation = robot.get_observation()

    assert observation["x.vel"] == pytest.approx(0.1, rel=1e-3, abs=1e-3)
    assert observation["theta.vel"] == pytest.approx(0.0, abs=1e-6)


def test_get_observation_restores_left_turn_semantics() -> None:
    robot = make_robot()
    corrected_feedback = robot._apply_base_velocity_drive_mode(robot._body_to_wheel_raw(0.0, 30.0))
    robot.bus2.sync_read_values[("Present_Velocity", tuple(robot.base_motors))] = corrected_feedback

    observation = robot.get_observation()

    assert observation["x.vel"] == pytest.approx(0.0, abs=1e-3)
    assert observation["theta.vel"] == pytest.approx(30.0, rel=1e-3, abs=1e-3)
