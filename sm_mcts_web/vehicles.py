"""Continuous vehicle layer: Ackermann (bicycle) model + pure pursuit.

The planner never touches actuators — it outputs metric waypoint routes
(interface.Route). This module turns routes into continuous motion:

- ``AckermannCar`` integrates the kinematic bicycle model
  (x' = v cos θ, y' = v sin θ, θ' = v/L tan δ) with steering-rate,
  acceleration and speed limits.
- ``PurePursuit`` picks a speed-dependent lookahead point on the route
  polyline and converts it into a steering command.
- Human-driven cars are fed key states (throttle/brake/steer) instead.

Swapping the vehicle model (differential drive, drone, ...) only means
replacing this module — the planner interface stays untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class VehicleParams:
    wheelbase: float = 2.6        # m
    length: float = 4.4           # m, for rendering/collision only
    width: float = 1.9            # m
    max_steer: float = 0.55       # rad
    steer_rate: float = 2.2       # rad/s
    accel: float = 3.5            # m/s^2
    brake: float = 7.0            # m/s^2
    max_speed: float = 8.0        # m/s
    max_reverse: float = 2.5      # m/s


@dataclass
class AckermannCar:
    x: float
    y: float
    theta: float
    params: VehicleParams = field(default_factory=VehicleParams)
    speed: float = 0.0
    steer: float = 0.0

    def step(self, target_speed: float, target_steer: float, dt: float) -> None:
        p = self.params
        target_speed = max(-p.max_reverse, min(p.max_speed, target_speed))
        target_steer = max(-p.max_steer, min(p.max_steer, target_steer))

        # steering with rate limit
        steer_delta = max(-p.steer_rate * dt,
                          min(p.steer_rate * dt, target_steer - self.steer))
        self.steer += steer_delta

        # longitudinal with accel/brake limits
        rate = p.accel if abs(target_speed) > abs(self.speed) else p.brake
        speed_delta = max(-rate * dt, min(rate * dt, target_speed - self.speed))
        self.speed += speed_delta

        # kinematic bicycle
        self.x += self.speed * math.cos(self.theta) * dt
        self.y += self.speed * math.sin(self.theta) * dt
        self.theta += self.speed / p.wheelbase * math.tan(self.steer) * dt
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

    def stop(self) -> None:
        self.speed = 0.0


class PurePursuit:
    """Tracks a metric waypoint route with a speed-dependent lookahead."""

    def __init__(self, min_lookahead: float = 2.5, gain: float = 0.6):
        self.min_lookahead = min_lookahead
        self.gain = gain
        self.route: list = []
        self.cruise_speed: float = 0.0
        self._stall_time = 0.0     # seconds spent (almost) stationary
        self._reverse_time = 0.0   # remaining reverse-recovery seconds
        self._last_alpha = 0.0

    def set_route(self, waypoints, speed: float) -> None:
        self.route = list(waypoints)
        self.cruise_speed = speed

    def _prune_passed(self, car: AckermannCar) -> None:
        # drop waypoints we are already past (within capture radius)
        while self.route:
            wx, wy = self.route[0]
            if math.hypot(wx - car.x, wy - car.y) < 1.0:
                self.route.pop(0)
            else:
                break

    def command(self, car: AckermannCar, dt: float = 0.05):
        """-> (target_speed, target_steer). Empty route = brake to stop."""
        self._prune_passed(car)
        if not self.route:
            self._stall_time = 0.0
            return 0.0, 0.0

        # stall detection -> short reverse-recovery arc (a car wedged
        # against something cannot turn in place; backing up frees it)
        if self._reverse_time > 0.0:
            self._reverse_time -= dt
            steer = car.params.max_steer if self._last_alpha < 0 \
                else -car.params.max_steer
            return -1.8, steer
        if abs(car.speed) < 0.15:
            self._stall_time += dt
            if self._stall_time > 1.5:
                self._stall_time = 0.0
                self._reverse_time = 1.2
        else:
            self._stall_time = 0.0

        lookahead = max(self.min_lookahead, self.gain * abs(car.speed) + 1.5)
        target = self.route[-1]
        for wx, wy in self.route:
            if math.hypot(wx - car.x, wy - car.y) >= lookahead:
                target = (wx, wy)
                break

        dx, dy = target[0] - car.x, target[1] - car.y
        distance = math.hypot(dx, dy)
        alpha = math.atan2(dy, dx) - car.theta
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        self._last_alpha = alpha

        if abs(alpha) > 2.4 and distance < 4.0:
            # target essentially behind us: creep and turn hard
            steer = car.params.max_steer if alpha > 0 else -car.params.max_steer
            return 1.0, steer

        curvature = 2.0 * math.sin(alpha) / max(distance, 1e-3)
        steer = math.atan(car.params.wheelbase * curvature)

        # slow down for sharp turns and near the end of the route
        remaining = distance + 2.0 * max(0, len(self.route) - 1)
        speed = min(
            self.cruise_speed,
            math.sqrt(2.0 * car.params.brake * max(remaining - 1.0, 0.2)),
            self.cruise_speed * (1.0 - 0.65 * min(abs(alpha) / 1.6, 1.0)) + 0.8,
        )
        return speed, steer


def human_command(car: AckermannCar, keys: set):
    """Arrow/WASD key set -> (target_speed, target_steer)."""
    p = car.params
    if "up" in keys:
        target_speed = p.max_speed
    elif "down" in keys:
        target_speed = -p.max_reverse if car.speed < 0.5 else 0.0
    else:
        target_speed = 0.0
    steer = 0.0
    sign = -1.0 if car.speed < -0.1 else 1.0  # natural reverse steering
    if "left" in keys:
        steer = p.max_steer * sign
    if "right" in keys:
        steer = -p.max_steer * sign
    return target_speed, steer
