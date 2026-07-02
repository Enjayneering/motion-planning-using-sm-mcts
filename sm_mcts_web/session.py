"""Simulation session: continuous world loop + planner scheduling.

The session owns the vehicles and the clock. The planner is only ever
touched through the ``PlannerAdapter`` interface, runs in a worker thread
on its own cadence, and is allowed to be late or to fail — the vehicles
then simply keep tracking their last routes (graceful degradation).
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import time
from dataclasses import dataclass, field

from .interface import (
    AgentState,
    PlannerAdapter,
    Route,
    ScenarioSpec,
    WorldSnapshot,
)
from .vehicles import AckermannCar, PurePursuit, VehicleParams, human_command

logger = logging.getLogger("sm_mcts_web")


@dataclass
class SimStats:
    time_s: float = 0.0
    collisions_car_car: int = 0
    collisions_obstacle: int = 0
    last_plan_ms: float = 0.0
    plan_count: int = 0
    planner_errors: int = 0
    all_ai_reached: bool = False
    _colliding_pairs: set = field(default_factory=set)
    _colliding_obstacles: set = field(default_factory=set)


class SimSession:
    def __init__(self, scenario: ScenarioSpec, adapter: PlannerAdapter):
        self.scenario = scenario
        self.adapter = adapter
        self.stats = SimStats()
        self.human_keys: dict = {}   # agent_id -> set of pressed keys

        self.cars: dict = {}
        self.trackers: dict = {}
        self.routes: dict = {}
        for agent in scenario.agents:
            params = VehicleParams(max_speed=agent.max_speed)
            self.cars[agent.id] = AckermannCar(
                x=agent.start[0], y=agent.start[1], theta=agent.start[2],
                params=params,
            )
            if agent.kind == "ai":
                self.trackers[agent.id] = PurePursuit()
                self.routes[agent.id] = Route(waypoints=(), speed=0.0)
            else:
                self.human_keys[agent.id] = set()

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._plan_future = None
        self._last_plan_started = -1e9
        self._reset_done = False

    # ------------------------------------------------------------------
    def reset_planner(self) -> None:
        """Blocking planner reset (includes JIT warmup); call off-loop."""
        self.adapter.reset(self.scenario)
        self._reset_done = True

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------
    def _snapshot(self) -> WorldSnapshot:
        agents = []
        for spec in self.scenario.agents:
            car = self.cars[spec.id]
            reached = (
                spec.goal is not None
                and math.hypot(car.x - spec.goal[0], car.y - spec.goal[1]) < 2.0
            )
            agents.append(AgentState(
                id=spec.id, x=car.x, y=car.y, theta=car.theta,
                speed=car.speed, reached=reached,
            ))
        return WorldSnapshot(time_s=self.stats.time_s, agents=tuple(agents))

    def _maybe_replan(self) -> None:
        if not self._reset_done:
            return
        now = self.stats.time_s
        if self._plan_future is not None:
            if not self._plan_future.done():
                return
            started = self._last_plan_started
            try:
                routes = self._plan_future.result()
                for agent_id, route in routes.items():
                    if agent_id in self.trackers:
                        self.routes[agent_id] = route
                        self.trackers[agent_id].set_route(
                            route.waypoints, route.speed
                        )
                self.stats.plan_count += 1
                self.stats.last_plan_ms = 1e3 * (self._wall() - started)
            except Exception:
                logger.exception("planner failed; keeping previous routes")
                self.stats.planner_errors += 1
            self._plan_future = None
        if now - getattr(self, "_last_plan_sim_time", -1e9) \
                >= self.scenario.planner.replan_period_s:
            self._last_plan_sim_time = now
            self._last_plan_started = self._wall()
            snapshot = self._snapshot()
            self._plan_future = self._executor.submit(self.adapter.plan, snapshot)

    @staticmethod
    def _wall() -> float:
        return time.perf_counter()

    # ------------------------------------------------------------------
    def _check_collisions(self) -> None:
        specs = {a.id: a for a in self.scenario.agents}
        ids = list(self.cars)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = self.cars[ids[i]], self.cars[ids[j]]
                limit = specs[ids[i]].radius + specs[ids[j]].radius
                pair = (ids[i], ids[j])
                if math.hypot(a.x - b.x, a.y - b.y) < limit:
                    if pair not in self.stats._colliding_pairs:
                        self.stats.collisions_car_car += 1
                        self.stats._colliding_pairs.add(pair)
                    # simple physical response: both stop
                    a.stop()
                    b.stop()
                else:
                    self.stats._colliding_pairs.discard(pair)

        for agent_id, car in self.cars.items():
            radius = specs[agent_id].radius
            for k, obstacle in enumerate(self.scenario.obstacles):
                if not obstacle.blocking:
                    continue
                if self._hits_obstacle(car, radius, obstacle):
                    key = (agent_id, k)
                    if key not in self.stats._colliding_obstacles:
                        self.stats.collisions_obstacle += 1
                        self.stats._colliding_obstacles.add(key)
                    # push back along the last motion direction and stop
                    car.x -= math.cos(car.theta) * car.speed * 0.06
                    car.y -= math.sin(car.theta) * car.speed * 0.06
                    car.stop()
                else:
                    self.stats._colliding_obstacles.discard((agent_id, k))

    @staticmethod
    def _hits_obstacle(car, radius, obstacle) -> bool:
        px, py = car.x - obstacle.x, car.y - obstacle.y
        if obstacle.radius > 0:
            return math.hypot(px, py) < obstacle.radius + radius * 0.7
        cos_r, sin_r = math.cos(-obstacle.rotation), math.sin(-obstacle.rotation)
        lx = px * cos_r - py * sin_r
        ly = px * sin_r + py * cos_r
        dx = max(abs(lx) - obstacle.width / 2, 0.0)
        dy = max(abs(ly) - obstacle.height / 2, 0.0)
        return math.hypot(dx, dy) < radius * 0.7

    # ------------------------------------------------------------------
    def tick(self, dt: float) -> None:
        self._maybe_replan()
        for spec in self.scenario.agents:
            car = self.cars[spec.id]
            if spec.kind == "human":
                speed, steer = human_command(car, self.human_keys[spec.id])
            else:
                speed, steer = self.trackers[spec.id].command(car, dt)
            car.step(speed, steer, dt)
            # keep everyone inside the world
            car.x = min(max(car.x, 0.5), self.scenario.width_m - 0.5)
            car.y = min(max(car.y, 0.5), self.scenario.height_m - 0.5)
        self._check_collisions()
        self.stats.time_s += dt
        self.stats.all_ai_reached = all(
            a.reached for a in self._snapshot().agents
            if next(s for s in self.scenario.agents if s.id == a.id).kind == "ai"
        )

    # ------------------------------------------------------------------
    def state_json(self) -> dict:
        snapshot = self._snapshot()
        return {
            "type": "tick",
            "time_s": round(self.stats.time_s, 2),
            "agents": [
                {
                    "id": a.id, "x": round(a.x, 2), "y": round(a.y, 2),
                    "theta": round(a.theta, 3), "speed": round(a.speed, 2),
                    "reached": a.reached,
                }
                for a in snapshot.agents
            ],
            "routes": {
                agent_id: [[round(x, 1), round(y, 1)] for x, y in route.waypoints]
                for agent_id, route in self.routes.items()
            },
            "stats": {
                "plan_ms": round(self.stats.last_plan_ms, 0),
                "plans": self.stats.plan_count,
                "collisions_car_car": self.stats.collisions_car_car,
                "collisions_obstacle": self.stats.collisions_obstacle,
                "planner_errors": self.stats.planner_errors,
                "all_ai_reached": self.stats.all_ai_reached,
                "planner_ready": self._reset_done,
            },
            "debug": self._safe_debug_info(),
        }

    def _safe_debug_info(self) -> dict:
        try:
            return self.adapter.debug_info() or {}
        except Exception:
            logger.exception("debug_info failed")
            return {}
