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
        max_distance_per_step_m: float | None = None,
    ) -> None:
        self.robot = robot
        self.right_arm_wheel_usb = right_arm_wheel_usb
        # Keep this falsy so LLMAgent skips head and arm initialization.
        self.left_arm_head_usb = None
        self.linear_speed_mps = linear_speed_mps
        self.angular_speed_dps = angular_speed_dps
        self.max_distance_per_step_m = (
            abs(float(max_distance_per_step_m)) if max_distance_per_step_m is not None else None
        )

    def _run_for_duration(self, *, x_vel: float = 0.0, theta_vel: float = 0.0, duration_s: float) -> None:
        if duration_s <= 0:
            return

        print(f"[Tool Exec] command x.vel={x_vel:.2f} theta.vel={theta_vel:.2f} duration_s={duration_s:.2f}")
        try:
            self.robot.send_action({"x.vel": x_vel, "theta.vel": theta_vel})
            time.sleep(duration_s)
        finally:
            self.robot.stop_base()

    def _clip_step_distance(self, distance: float) -> float:
        if self.max_distance_per_step_m is None:
            return distance
        return min(distance, self.max_distance_per_step_m)

    def go_forward(self, meters: float) -> float:
        requested_meters = float(meters)
        requested_distance = abs(requested_meters)
        distance = self._clip_step_distance(requested_distance)
        duration_s = distance / self.linear_speed_mps
        print(
            "[Tool Exec] go_forward "
            f"requested_meters={requested_meters:.2f} meters={distance:.2f} "
            f"speed_mps={self.linear_speed_mps:.2f} duration_s={duration_s:.2f}"
        )
        if distance < requested_distance:
            print(
                "[Tool Exec] go_forward clipped "
                f"requested_meters={requested_distance:.2f} max_step_m={self.max_distance_per_step_m:.2f}"
            )
        self._run_for_duration(x_vel=self.linear_speed_mps, duration_s=duration_s)
        return distance

    def go_backward(self, meters: float) -> float:
        requested_meters = float(meters)
        requested_distance = abs(requested_meters)
        distance = self._clip_step_distance(requested_distance)
        duration_s = distance / self.linear_speed_mps
        print(
            "[Tool Exec] go_backward "
            f"requested_meters={requested_meters:.2f} meters={distance:.2f} "
            f"speed_mps={self.linear_speed_mps:.2f} duration_s={duration_s:.2f}"
        )
        if distance < requested_distance:
            print(
                "[Tool Exec] go_backward clipped "
                f"requested_meters={requested_distance:.2f} max_step_m={self.max_distance_per_step_m:.2f}"
            )
        self._run_for_duration(x_vel=-self.linear_speed_mps, duration_s=duration_s)
        return distance

    def turn_left(self, degrees: float) -> None:
        requested_degrees = float(degrees)
        angle = abs(requested_degrees)
        duration_s = angle / self.angular_speed_dps
        print(
            "[Tool Exec] turn_left "
            f"requested_degrees={requested_degrees:.2f} degrees={angle:.2f} "
            f"angular_speed_dps={self.angular_speed_dps:.2f} duration_s={duration_s:.2f}"
        )
        self._run_for_duration(theta_vel=self.angular_speed_dps, duration_s=duration_s)

    def turn_right(self, degrees: float) -> None:
        requested_degrees = float(degrees)
        angle = abs(requested_degrees)
        duration_s = angle / self.angular_speed_dps
        print(
            "[Tool Exec] turn_right "
            f"requested_degrees={requested_degrees:.2f} degrees={angle:.2f} "
            f"angular_speed_dps={self.angular_speed_dps:.2f} duration_s={duration_s:.2f}"
        )
        self._run_for_duration(theta_vel=-self.angular_speed_dps, duration_s=duration_s)

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
