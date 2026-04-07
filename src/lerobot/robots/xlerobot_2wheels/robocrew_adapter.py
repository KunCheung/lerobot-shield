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

"""RoboCrew-compatible adapter for the XLerobot 2-wheel base."""

from __future__ import annotations

import time

from lerobot.utils.errors import DeviceNotConnectedError

from .xlerobot_2wheels import XLerobot2Wheels


class TwoWheelsServoAdapter:
    """Minimal RoboCrew-compatible wrapper around XLerobot2Wheels."""

    def __init__(
        self,
        robot: XLerobot2Wheels,
        *,
        right_arm_wheel_usb: str,
        linear_speed_mps: float,
        angular_speed_dps: float,
    ) -> None:
        self.robot = robot
        self.right_arm_wheel_usb = right_arm_wheel_usb
        # Keep this falsy so LLMAgent skips head and arm initialization.
        self.left_arm_head_usb = None
        self.linear_speed_mps = linear_speed_mps
        self.angular_speed_dps = angular_speed_dps

    def _run_for_duration(self, *, x_vel: float = 0.0, theta_vel: float = 0.0, duration_s: float) -> None:
        if duration_s <= 0:
            return

        try:
            self.robot.send_action({"x.vel": x_vel, "theta.vel": theta_vel})
            time.sleep(duration_s)
        finally:
            self.robot.stop_base()

    def go_forward(self, meters: float) -> None:
        distance = abs(float(meters))
        self._run_for_duration(x_vel=self.linear_speed_mps, duration_s=distance / self.linear_speed_mps)

    def go_backward(self, meters: float) -> None:
        distance = abs(float(meters))
        self._run_for_duration(x_vel=-self.linear_speed_mps, duration_s=distance / self.linear_speed_mps)

    def turn_left(self, degrees: float) -> None:
        angle = abs(float(degrees))
        self._run_for_duration(theta_vel=self.angular_speed_dps, duration_s=angle / self.angular_speed_dps)

    def turn_right(self, degrees: float) -> None:
        angle = abs(float(degrees))
        self._run_for_duration(theta_vel=-self.angular_speed_dps, duration_s=angle / self.angular_speed_dps)

    def disconnect(self) -> None:
        try:
            self.robot.stop_base()
        except Exception:
            pass

        try:
            if self.robot.is_connected:
                self.robot.disconnect()
        except DeviceNotConnectedError:
            pass
