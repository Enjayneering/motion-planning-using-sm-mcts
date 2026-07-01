"""Grid-world environments with vectorized free-space checks.

Worlds can be built programmatically or from ASCII art:

    #  wall / obstacle
    .  free space
    0-9        start cell of agent i (heading is initialized towards the goal)
    a-j        goal cell of agent i  (a -> agent 0, b -> agent 1, ...)

Example (two agents crossing an intersection)::

    env = ascii_world('''
        ##1##
        ##.##
        0...a
        ##.##
        ##b##
    ''', velocities=(0.0, 1.0), angular_velocities=(-1.57, 0.0, 1.57))
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .dynamics import make_action_set, unicycle_step


class GridWorld(NamedTuple):
    """Static environment + per-agent task definition (a pytree of arrays)."""

    occupancy: jnp.ndarray        # [H, W] bool, True = obstacle
    starts: jnp.ndarray           # [n_agents, 3] (x, y, theta)
    goals: jnp.ndarray            # [n_agents, 2] (x, y)
    actions: jnp.ndarray          # [n_agents, n_actions, 2] per-agent action set
    null_action: jnp.ndarray      # [n_agents] index of the "stand still" action
    dt: jnp.ndarray               # scalar
    goal_radius: jnp.ndarray      # scalar
    collision_radius: jnp.ndarray # scalar, minimum inter-agent distance

    @property
    def n_agents(self) -> int:
        return self.starts.shape[0]

    @property
    def n_actions(self) -> int:
        return self.actions.shape[1]


def build_world(
    occupancy,
    starts,
    goals,
    velocities=(0.0, 1.0),
    angular_velocities=(-1.5708, 0.0, 1.5708),
    actions=None,
    dt=1.0,
    goal_radius=0.8,
    collision_radius=0.8,
) -> GridWorld:
    """Assemble a GridWorld. `actions` overrides the (v, w) value grids."""
    occupancy = jnp.asarray(occupancy, dtype=bool)
    starts = jnp.asarray(starts, dtype=jnp.float32)
    goals = jnp.asarray(goals, dtype=jnp.float32)
    n_agents = starts.shape[0]
    if actions is None:
        actions = make_action_set(velocities, angular_velocities)
    actions = jnp.asarray(actions, dtype=jnp.float32)
    if actions.ndim == 2:
        actions = jnp.broadcast_to(actions, (n_agents,) + actions.shape)
    # the action closest to standing still; used to freeze agents at their goal
    null_action = jnp.argmin(
        jnp.abs(actions[..., 0]) + 1e-3 * jnp.abs(actions[..., 1]), axis=-1
    ).astype(jnp.int32)
    return GridWorld(
        occupancy=occupancy,
        starts=starts,
        goals=goals,
        actions=actions,
        null_action=null_action,
        dt=jnp.float32(dt),
        goal_radius=jnp.float32(goal_radius),
        collision_radius=jnp.float32(collision_radius),
    )


def ascii_world(art: str, **kwargs) -> GridWorld:
    """Parse an ASCII map (see module docstring for the legend)."""
    lines = [ln.strip() for ln in art.strip().splitlines() if ln.strip()]
    width = max(len(ln) for ln in lines)
    lines = [ln.ljust(width, "#") for ln in lines]
    height = len(lines)

    occupancy = np.zeros((height, width), dtype=bool)
    starts_xy: dict[int, tuple[float, float]] = {}
    goals_xy: dict[int, tuple[float, float]] = {}

    for row, ln in enumerate(lines):
        y = float(height - 1 - row)  # row 0 is the top of the drawing
        for col, ch in enumerate(ln):
            x = float(col)
            if ch == "#":
                occupancy[height - 1 - row, col] = True
            elif ch.isdigit():
                starts_xy[int(ch)] = (x, y)
            elif "a" <= ch <= "j":
                goals_xy[ord(ch) - ord("a")] = (x, y)

    if set(starts_xy) != set(goals_xy):
        raise ValueError(
            f"agents with starts {sorted(starts_xy)} do not match goals {sorted(goals_xy)}"
        )
    if not starts_xy:
        raise ValueError("no agents found in ASCII map")

    ids = sorted(starts_xy)
    starts = []
    goals = []
    for i in ids:
        sx, sy = starts_xy[i]
        gx, gy = goals_xy[i]
        theta = float(np.arctan2(gy - sy, gx - sx))
        starts.append([sx, sy, theta])
        goals.append([gx, gy])
    return build_world(occupancy, starts, goals, **kwargs)


# ---------------------------------------------------------------------------
# Vectorized geometry checks
# ---------------------------------------------------------------------------

_N_SEGMENT_SAMPLES = 5


def points_are_free(env: GridWorld, points: jnp.ndarray) -> jnp.ndarray:
    """points [..., 2] -> [...] bool. Free = inside the map and not on an obstacle."""
    height, width = env.occupancy.shape
    x, y = points[..., 0], points[..., 1]
    inside = (x >= -0.5) & (x <= width - 0.5) & (y >= -0.5) & (y <= height - 0.5)
    col = jnp.clip(jnp.round(x).astype(jnp.int32), 0, width - 1)
    row = jnp.clip(jnp.round(y).astype(jnp.int32), 0, height - 1)
    return inside & ~env.occupancy[row, col]


def segment_is_free(env: GridWorld, p0: jnp.ndarray, p1: jnp.ndarray) -> jnp.ndarray:
    """Line search along p0 -> p1 (both [..., 2]) against the occupancy grid."""
    ts = jnp.linspace(0.0, 1.0, _N_SEGMENT_SAMPLES)
    pts = p0[..., None, :] + ts[:, None] * (p1 - p0)[..., None, :]  # [..., S, 2]
    return jnp.all(points_are_free(env, pts), axis=-1)


def legal_action_mask(env: GridWorld, states: jnp.ndarray) -> jnp.ndarray:
    """Kinodynamically reachable, obstacle-free actions per agent.

    states [n_agents, 3] -> mask [n_agents, n_actions] bool
    """
    next_states = unicycle_step(states[:, None, :], env.actions, env.dt)
    p0 = jnp.broadcast_to(states[:, None, :2], next_states[..., :2].shape)
    return segment_is_free(env, p0, next_states[..., :2])


def goal_distances(env: GridWorld, states: jnp.ndarray) -> jnp.ndarray:
    """Euclidean distance of each agent to its own goal. [..., n_agents, 3] -> [..., n_agents]"""
    return jnp.linalg.norm(states[..., :2] - env.goals, axis=-1)


def step_world(env: GridWorld, states: jnp.ndarray, reached: jnp.ndarray,
               action_idx: jnp.ndarray):
    """Advance all agents one timestep; agents that reached their goal stay frozen.

    states [n_agents, 3], reached [n_agents] bool, action_idx [n_agents] int
    returns (next_states, next_reached)
    """
    acts = jnp.take_along_axis(
        env.actions, action_idx[:, None, None], axis=1
    )[:, 0, :]
    proposed = unicycle_step(states, acts, env.dt)
    next_states = jnp.where(reached[:, None], states, proposed)
    next_reached = reached | (goal_distances(env, next_states) <= env.goal_radius)
    return next_states, next_reached


def agents_collide(env: GridWorld, prev_states: jnp.ndarray,
                   next_states: jnp.ndarray) -> jnp.ndarray:
    """Per-agent flag: does the agent get closer than collision_radius to any
    other agent anywhere along the (linearly interpolated) transition?

    returns [n_agents] bool
    """
    ts = jnp.linspace(0.0, 1.0, _N_SEGMENT_SAMPLES)
    # [S, n_agents, 2]
    pts = prev_states[None, :, :2] + ts[:, None, None] * (
        next_states[:, :2] - prev_states[:, :2]
    )[None, :, :]
    # pairwise distances at each sample time: [S, n_agents, n_agents]
    diff = pts[:, :, None, :] - pts[:, None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)
    n = dist.shape[-1]
    dist = dist + jnp.eye(n) * 1e9  # ignore self-distance
    return jnp.any(dist < env.collision_radius, axis=(0, 2))
