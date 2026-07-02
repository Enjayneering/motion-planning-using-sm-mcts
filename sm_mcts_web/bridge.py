"""SM-MCTS adapter: the discrete real-time planner behind the continuous sim.

This is the reference implementation of ``PlannerAdapter`` — the research
planner (sm_mcts_jax) wrapped for the continuous world:

- The continuous map is rasterized once into an occupancy grid at
  ``PlannerConfig.resolution_m`` (blocking footprints only; roads etc. are
  decorative).
- On every ``plan()`` the live vehicle poses are snapped onto the grid, one
  SM-MCTS search runs (a single JIT-compiled XLA call), and the committed
  principal variation is converted back into metric waypoint routes that
  the simulator's pure-pursuit controller tracks.
- Human-driven vehicles are included in the game as agents with an
  **estimated goal** (their heading projected ahead). This is deliberately
  the most primitive possible intent model — the placeholder for research
  question RQ5 (docs/VISION.md); the search still treats the human as a
  responsive strategic agent, not a ballistic obstacle.
- Per-agent behavior ("cautious"/"normal"/"aggressive") maps to
  heterogeneous payoff weights.
"""

from __future__ import annotations

import math
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

from sm_mcts_jax import MCTSParams, RewardParams, build_world, search
from sm_mcts_jax.dynamics import unicycle_step
from sm_mcts_jax.environment import _time_expanded_field, goal_distances

from .interface import PlannerAdapter, Route, ScenarioSpec, WorldSnapshot

# payoff weights per behavior: (progress, proximity, time)
_BEHAVIORS = {
    "cautious":   (0.8, 1.00, 0.03),
    "normal":     (1.0, 0.50, 0.05),
    "aggressive": (1.2, 0.15, 0.08),
}
_HUMAN_LOOKAHEAD_CELLS = 6


def rasterize(scenario: ScenarioSpec, resolution: float,
              inflation: float = 0.0) -> np.ndarray:
    """Blocking footprints -> occupancy [H, W]; True = blocked. A cell is
    blocked when any of its 3x3 sample points lies inside a footprint that
    is grown by `inflation` meters (vehicle extent), so planned routes keep
    physical clearance from walls."""
    width = max(2, int(math.ceil(scenario.width_m / resolution)))
    height = max(2, int(math.ceil(scenario.height_m / resolution)))
    occ = np.zeros((height, width), dtype=bool)
    offsets = np.array([-0.3, 0.0, 0.3]) * resolution
    for obstacle in scenario.obstacles:
        if not obstacle.blocking:
            continue
        cos_r, sin_r = math.cos(-obstacle.rotation), math.sin(-obstacle.rotation)
        # bounding box in cells (generous)
        extent = (obstacle.radius
                  or math.hypot(obstacle.width, obstacle.height) / 2) + inflation
        c0 = max(0, int((obstacle.x - extent) / resolution) - 1)
        c1 = min(width - 1, int((obstacle.x + extent) / resolution) + 1)
        r0 = max(0, int((obstacle.y - extent) / resolution) - 1)
        r1 = min(height - 1, int((obstacle.y + extent) / resolution) + 1)
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                if occ[row, col]:
                    continue
                cx, cy = col * resolution, row * resolution
                for dx in offsets:
                    for dy in offsets:
                        px, py = cx + dx - obstacle.x, cy + dy - obstacle.y
                        if obstacle.radius > 0:
                            hit = (px * px + py * py
                                   <= (obstacle.radius + inflation) ** 2)
                        else:
                            lx = px * cos_r - py * sin_r
                            ly = px * sin_r + py * cos_r
                            hit = (abs(lx) <= obstacle.width / 2 + inflation
                                   and abs(ly) <= obstacle.height / 2 + inflation)
                        if hit:
                            occ[row, col] = True
                            break
                    if occ[row, col]:
                        break
    return occ


def nearest_free_cell(occ: np.ndarray, col: int, row: int) -> tuple:
    """BFS to the closest unblocked cell (col, row)."""
    height, width = occ.shape
    col = int(np.clip(col, 0, width - 1))
    row = int(np.clip(row, 0, height - 1))
    if not occ[row, col]:
        return col, row
    seen = {(col, row)}
    queue = deque([(col, row)])
    while queue:
        c, r = queue.popleft()
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nc, nr = c + dc, r + dr
            if 0 <= nc < width and 0 <= nr < height and (nc, nr) not in seen:
                if not occ[nr, nc]:
                    return nc, nr
                seen.add((nc, nr))
                queue.append((nc, nr))
    raise ValueError("map has no free cells")


class SMMCTSAdapter(PlannerAdapter):
    """The sm_mcts_jax planner behind the PlannerAdapter interface."""

    def __init__(self, seed: int = 0):
        self._seed = seed

    # ------------------------------------------------------------------
    def reset(self, scenario: ScenarioSpec) -> None:
        cfg = scenario.planner
        self._scenario = scenario
        self._res = cfg.resolution_m
        inflation = max((a.radius for a in scenario.agents), default=0.0)
        self._occ = rasterize(scenario, self._res, inflation=inflation)
        self._agents = list(scenario.agents)
        self._human_ix = [i for i, a in enumerate(self._agents)
                          if a.kind == "human"]
        self._ai_ix = [i for i, a in enumerate(self._agents) if a.kind == "ai"]
        if not self._agents:
            raise ValueError("scenario has no agents")

        starts, goals = [], []
        for agent in self._agents:
            col, row = nearest_free_cell(
                self._occ, round(agent.start[0] / self._res),
                round(agent.start[1] / self._res),
            )
            quarter = math.pi / 2
            theta = round(agent.start[2] / quarter) * quarter
            starts.append([float(col), float(row), theta])
            if agent.goal is not None:
                gc, gr = nearest_free_cell(
                    self._occ, round(agent.goal[0] / self._res),
                    round(agent.goal[1] / self._res),
                )
            else:  # human: provisional goal, re-estimated on every plan
                gc, gr = col, row
            goals.append([float(gc), float(gr)])

        max_radius = max(a.radius for a in self._agents)
        behaviors = [_BEHAVIORS.get(a.behavior, _BEHAVIORS["normal"])
                     for a in self._agents]
        self._reward_params = RewardParams(
            weight_progress=tuple(b[0] for b in behaviors),
            weight_proximity=tuple(b[1] for b in behaviors),
            weight_time=tuple(b[2] for b in behaviors),
        )
        self._mcts_params = MCTSParams(
            num_simulations=cfg.num_simulations,
            max_depth=12,
            rollout_depth=12,
            k_rollouts=2,
            c_uct=1.2,
            commit_depth=cfg.commit_depth,
            safety_filter=cfg.safety_filter,
        )
        self._env = build_world(
            self._occ, starts, goals,
            goal_radius=0.8,
            collision_radius=min(1.9, 2.0 * max_radius / self._res),
        )
        self._human_goal_cells = {i: tuple(goals[i]) for i in self._human_ix}
        self._rng = jax.random.PRNGKey(self._seed)
        self._steps = 0
        # trigger JIT compilation so the first live replan is fast
        self._search(self._env.starts, jnp.zeros(len(self._agents), bool))

    # ------------------------------------------------------------------
    def _search(self, states, reached):
        self._rng, key = jax.random.split(self._rng)
        result = search(
            self._env, self._mcts_params, self._reward_params,
            states, reached, jnp.int32(0), key,
        )
        jax.block_until_ready(result.action_plan)
        return result

    def _estimate_human_goal(self, index: int, state) -> tuple:
        """Project the human's heading ahead and snap to a free cell — the
        deliberately primitive intent model (RQ5 placeholder)."""
        col = state.x / self._res + math.cos(state.theta) * _HUMAN_LOOKAHEAD_CELLS
        row = state.y / self._res + math.sin(state.theta) * _HUMAN_LOOKAHEAD_CELLS
        return nearest_free_cell(self._occ, round(col), round(row))

    def _update_human_goals(self, snapshot: WorldSnapshot) -> None:
        """Swap in new assumed goals; only the changed agents' distance
        fields are recomputed (the rest of the env is reused)."""
        changed = False
        goals = np.array(self._env.goals)          # writable copies
        fields = np.array(self._env.dist_fields)
        for i in self._human_ix:
            goal_cell = self._estimate_human_goal(i, snapshot.agents[i])
            if goal_cell != self._human_goal_cells[i]:
                self._human_goal_cells[i] = goal_cell
                goals[i] = goal_cell
                fields[i] = _time_expanded_field(
                    np.asarray(self._env.occupancy), 1, True,
                    np.asarray(goal_cell, dtype=float),
                )
                changed = True
        if changed:
            self._env = self._env._replace(
                goals=jnp.asarray(goals, jnp.float32),
                dist_fields=jnp.asarray(fields, jnp.float32),
            )

    # ------------------------------------------------------------------
    def plan(self, snapshot: WorldSnapshot) -> dict:
        self._update_human_goals(snapshot)
        quarter = math.pi / 2
        rows = []
        for a in snapshot.agents:
            # snap live poses onto the nearest FREE cell — a car hugging a
            # wall may sit in an inflation-blocked cell, which would hand
            # the planner an illegal state
            col, row = nearest_free_cell(
                self._occ, round(a.x / self._res), round(a.y / self._res)
            )
            rows.append([float(col), float(row),
                         round(a.theta / quarter) * quarter])
        states = jnp.asarray(rows, dtype=jnp.float32)
        reached = goal_distances(self._env, states) <= self._env.goal_radius
        # humans never 'arrive' — they must stay live players in the game
        reached = reached.at[jnp.asarray(self._human_ix, int)].set(False) \
            if self._human_ix else reached

        result = self._search(states, reached)
        plan = np.asarray(result.action_plan)  # [D, n]

        routes = {}
        for i in self._ai_ix:
            agent = self._agents[i]
            if bool(reached[i]):
                routes[agent.id] = Route(waypoints=(), speed=0.0)
                continue
            pose = states[i]
            waypoints = []
            for k in range(plan.shape[0]):
                action = self._env.actions[i, plan[k, i]]
                pose = unicycle_step(pose, action, self._env.dt)
                # turn-in-place steps (v = 0) have no Ackermann equivalent:
                # skip the stationary waypoint — the heading change emerges
                # from the controller driving towards the next moving one
                if abs(float(action[0])) > 1e-3:
                    waypoints.append((float(pose[0]) * self._res,
                                      float(pose[1]) * self._res))
            if not waypoints:
                waypoints = self._potential_fallback(i, snapshot.agents[i])
            routes[agent.id] = Route(
                waypoints=tuple(waypoints), speed=agent.max_speed
            )
        self._steps += 1
        return routes

    def _potential_fallback(self, index: int, state) -> list:
        """If the committed plan never moves (pure reorientation), steer
        towards the steepest-descent neighbor of the steps-to-goal field so
        an Ackermann vehicle can realize the turn by driving."""
        fields = np.asarray(self._env.dist_fields)  # [n, P, H, W, 4]
        occ = np.asarray(self._env.occupancy[0])
        height, width = occ.shape
        col = int(np.clip(round(state.x / self._res), 0, width - 1))
        row = int(np.clip(round(state.y / self._res), 0, height - 1))
        best, best_phi = None, np.inf
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nc, nr = col + dc, row + dr
            if 0 <= nc < width and 0 <= nr < height and not occ[nr, nc]:
                phi = float(fields[index, 0, nr, nc].min())  # best heading
                if phi < best_phi:
                    best_phi, best = phi, (nc, nr)
        if best is None:
            return []
        return [(best[0] * self._res, best[1] * self._res)]
