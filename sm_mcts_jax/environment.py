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

Dynamic environments are a sequence of occupancy frames. The first frame
defines starts and goals; later frames only describe walls (any non-'#'
character is free space). Each frame is active for `frame_duration`
timesteps; with `cycle=True` the sequence repeats::

    env = ascii_world([frame_a, frame_b], frame_duration=4, cycle=True)

All free-space checks take the timestep, so the tree search plans with the
grid that will be active when a state is actually visited.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .dynamics import make_action_set, unicycle_step

_UNREACHABLE = 1e6


class GridWorld(NamedTuple):
    """Environment + per-agent task definition (a pytree of arrays)."""

    occupancy: jnp.ndarray        # [T, H, W] bool, True = obstacle (T = 1: static)
    starts: jnp.ndarray           # [n_agents, 3] (x, y, theta)
    goals: jnp.ndarray            # [n_agents, 2] (x, y)
    actions: jnp.ndarray          # [n_agents, n_actions, 2] per-agent action set
    null_action: jnp.ndarray      # [n_agents] index of the "stand still" action
    dt: jnp.ndarray               # scalar
    goal_radius: jnp.ndarray      # scalar
    collision_radius: jnp.ndarray # scalar, minimum inter-agent distance
    frame_duration: jnp.ndarray   # scalar int, timesteps each occupancy frame lasts
    cycle: jnp.ndarray            # scalar bool, repeat frames vs. hold the last one
    dist_fields: jnp.ndarray      # [n_agents, P, H, W, 4] steps-to-goal field

    @property
    def n_agents(self) -> int:
        return self.starts.shape[0]

    @property
    def n_actions(self) -> int:
        return self.actions.shape[1]

    @property
    def n_frames(self) -> int:
        return self.occupancy.shape[0]


def frame_index(env: GridWorld, t) -> jnp.ndarray:
    """Occupancy frame active at (integer) timestep t."""
    idx = jnp.asarray(t, jnp.int32) // env.frame_duration
    n = env.n_frames
    return jnp.where(env.cycle, idx % n, jnp.clip(idx, 0, n - 1))


# heading index h -> (d_row, d_col); h * 90deg is the heading angle
_HEADINGS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # E, N, W, S (row = y)


def _time_expanded_field(frames: np.ndarray, frame_duration: int, cycle: bool,
                         goal_xy) -> np.ndarray:
    """Exact minimum steps-to-goal over the time-expanded unicycle graph.

    frames [T, H, W] bool -> field [P, H, W, 4] float32 with P the schedule
    period (P = 1 for static worlds) and 4 discrete headings (E, N, W, S).
    One step applies one action of the discrete unicycle model:
    v in {0, 1} x turn in {-90deg, 0, +90deg} — i.e. wait, turn in place, or
    move one cell along the current heading (optionally while turning).

    Because time, cell AND heading are part of the state, the field values
    both "wait two steps until the gate opens, then pass" and the turn steps
    needed to line up with the gate correctly. A heading-blind per-frame
    distance field would flip whenever the environment changes and reward
    aimless spinning on waiting plateaus.
    """
    n_frames, height, width = frames.shape
    period = 1 if n_frames == 1 else n_frames * frame_duration
    frame_of = lambda t: frames[(t // frame_duration) % n_frames]

    dist = np.full((period, height, width, 4), _UNREACHABLE, dtype=np.float32)
    goal_col = int(round(float(goal_xy[0])))
    goal_row = int(round(float(goal_xy[1])))

    from collections import deque
    queue: deque = deque()
    for t in range(period):
        if not frame_of(t)[goal_row, goal_col]:
            dist[t, goal_row, goal_col, :] = 0.0
            for h in range(4):
                queue.append((t, goal_row, goal_col, h))

    def predecessor_times(t2: int):
        if period == 1:
            return (0,)
        if cycle:
            return ((t2 - 1) % period,)
        preds = [t2 - 1] if t2 > 0 else []
        if t2 == period - 1:
            preds.append(period - 1)  # absorbing last frame
        return tuple(preds)

    # reverse BFS, unit cost: predecessor (t1, r1, c1, h1) applies action
    # (v, turn) and lands in (t2, r2, c2, h2) with h2 = (h1 + turn) % 4 and
    # (r2, c2) = (r1, c1) + v * direction(h1)
    while queue:
        t2, r2, c2, h2 = queue.popleft()
        d = dist[t2, r2, c2, h2]
        for t1 in predecessor_times(t2):
            occ_departure = frame_of(t1)
            for turn in (-1, 0, 1):
                h1 = (h2 - turn) % 4
                d_row, d_col = _HEADINGS[h1]
                for v in (0, 1):
                    r1, c1 = r2 - v * d_row, c2 - v * d_col
                    if not (0 <= r1 < height and 0 <= c1 < width):
                        continue
                    if occ_departure[r1, c1]:
                        continue
                    if d + 1.0 < dist[t1, r1, c1, h1]:
                        dist[t1, r1, c1, h1] = d + 1.0
                        queue.append((t1, r1, c1, h1))
    return dist


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
    frame_duration=1,
    cycle=True,
) -> GridWorld:
    """Assemble a GridWorld. `actions` overrides the (v, w) value grids.

    `occupancy` may be [H, W] (static) or [T, H, W] (dynamic frames).
    """
    occupancy = jnp.asarray(occupancy, dtype=bool)
    if occupancy.ndim == 2:
        occupancy = occupancy[None]
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
    # time-expanded steps-to-goal per agent (for payoffs and rollout
    # guidance; avoids the local minima of Euclidean distance and values
    # waiting for scheduled openings correctly)
    occupancy_np = np.asarray(occupancy)
    goals_np = np.asarray(goals)
    dist_fields = np.stack([
        _time_expanded_field(
            occupancy_np, int(frame_duration), bool(cycle), goals_np[i]
        )
        for i in range(n_agents)
    ])
    return GridWorld(
        occupancy=occupancy,
        starts=starts,
        goals=goals,
        actions=actions,
        null_action=null_action,
        dt=jnp.float32(dt),
        goal_radius=jnp.float32(goal_radius),
        collision_radius=jnp.float32(collision_radius),
        frame_duration=jnp.int32(frame_duration),
        cycle=jnp.bool_(cycle),
        dist_fields=jnp.asarray(dist_fields, dtype=jnp.float32),
    )


def _parse_frame(art: str):
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
    return occupancy, starts_xy, goals_xy


def ascii_world(art, snap_heading: bool = True, **kwargs) -> GridWorld:
    """Parse ASCII map(s) — a string, or a list of frames for dynamic worlds.

    Starts/goals are read from the first frame only; all frames must share
    the same size. See the module docstring for the legend.

    With `snap_heading` (default) the initial heading towards the goal is
    rounded to the nearest multiple of 90°. Combined with 90°-increment
    angular actions this keeps every reachable pose exactly on the grid —
    important in maps with 1-cell-wide corridors.
    """
    frames = [art] if isinstance(art, str) else list(art)
    parsed = [_parse_frame(f) for f in frames]
    if len({occ.shape for occ, _, _ in parsed}) > 1:
        raise ValueError("all frames must have the same dimensions")
    occupancy = np.stack([occ for occ, _, _ in parsed])
    _, starts_xy, goals_xy = parsed[0]

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
        if snap_heading:
            quarter = np.pi / 2.0
            theta = float(np.round(theta / quarter) * quarter)
        starts.append([sx, sy, theta])
        goals.append([gx, gy])
    return build_world(occupancy, starts, goals, **kwargs)


# ---------------------------------------------------------------------------
# Vectorized geometry checks
# ---------------------------------------------------------------------------

_N_SEGMENT_SAMPLES = 5


def points_are_free(env: GridWorld, points: jnp.ndarray, t=0) -> jnp.ndarray:
    """points [..., 2] -> [...] bool. Free = inside the map and not on an
    obstacle of the frame active at timestep `t` (broadcast against points)."""
    height, width = env.occupancy.shape[1:]
    x, y = points[..., 0], points[..., 1]
    inside = (x >= -0.5) & (x <= width - 0.5) & (y >= -0.5) & (y <= height - 0.5)
    col = jnp.clip(jnp.round(x).astype(jnp.int32), 0, width - 1)
    row = jnp.clip(jnp.round(y).astype(jnp.int32), 0, height - 1)
    tidx = jnp.broadcast_to(frame_index(env, t), col.shape)
    return inside & ~env.occupancy[tidx, row, col]


def segment_is_free(env: GridWorld, p0: jnp.ndarray, p1: jnp.ndarray, t=0) -> jnp.ndarray:
    """Line search along the transition p0 -> p1 (both [..., 2]) that starts
    at timestep `t` and arrives at `t + 1`. Sample times are floored, i.e.
    the world changes exactly at integer timesteps (as in the original)."""
    ts = jnp.linspace(0.0, 1.0, _N_SEGMENT_SAMPLES)
    pts = p0[..., None, :] + ts[:, None] * (p1 - p0)[..., None, :]  # [..., S, 2]
    times = jnp.floor(jnp.asarray(t, jnp.float32) + ts).astype(jnp.int32)  # [S]
    return jnp.all(points_are_free(env, pts, times), axis=-1)


def legal_action_mask(env: GridWorld, states: jnp.ndarray, t=0) -> jnp.ndarray:
    """Kinodynamically reachable, obstacle-free actions per agent at time t.

    states [n_agents, 3] -> mask [n_agents, n_actions] bool
    """
    next_states = unicycle_step(states[:, None, :], env.actions, env.dt)
    p0 = jnp.broadcast_to(states[:, None, :2], next_states[..., :2].shape)
    return segment_is_free(env, p0, next_states[..., :2], t)


def goal_distances(env: GridWorld, states: jnp.ndarray) -> jnp.ndarray:
    """Euclidean distance of each agent to its own goal. [..., n_agents, 3] -> [..., n_agents]"""
    return jnp.linalg.norm(states[..., :2] - env.goals, axis=-1)


def goal_potential(env: GridWorld, states: jnp.ndarray, t=0) -> jnp.ndarray:
    """Time-expanded steps-to-goal potential (obstacle-, schedule- and
    heading-aware).

    states [..., n_agents, 3] -> [..., n_agents]. A small Euclidean term is
    added for a sub-cell gradient. Used for payoffs and rollout guidance —
    unlike Euclidean distance it cannot trap agents behind walls, it values
    waiting for a scheduled opening as progress, and it accounts for the
    turn steps of the unicycle model.
    """
    height, width = env.occupancy.shape[1:]
    period = env.dist_fields.shape[1]
    col = jnp.clip(jnp.round(states[..., 0]).astype(jnp.int32), 0, width - 1)
    row = jnp.clip(jnp.round(states[..., 1]).astype(jnp.int32), 0, height - 1)
    heading = jnp.mod(
        jnp.round(states[..., 2] / (jnp.pi / 2.0)).astype(jnp.int32), 4
    )
    t = jnp.asarray(t, jnp.int32)
    tidx = jnp.where(env.cycle, t % period, jnp.clip(t, 0, period - 1))
    tidx = jnp.broadcast_to(tidx, row.shape)
    agent_ids = jnp.arange(env.n_agents)
    phi = env.dist_fields[agent_ids, tidx, row, col, heading]
    return phi + 0.1 * goal_distances(env, states)


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
