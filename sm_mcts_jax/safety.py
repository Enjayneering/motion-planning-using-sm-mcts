"""Provably collision-free action masking (one-step maximin safety filter).

Enable with ``MCTSParams(safety_filter=True)``. The filter replaces the
plain obstacle mask everywhere the search consults legal actions, turning
the soft collision *penalty* into a hard *constraint*. See docs/SAFETY.md
for the assumptions and the (short) proof; the informal statement:

    If all agents start mutually separated and each agent — independently —
    only ever picks actions from its filtered set, then no two agents ever
    come closer than the collision radius, at any point of any transition.
    Moreover the filtered set is never empty: standing still is always in
    it, so the filter cannot paint an agent into a corner.

The rule has two parts, both computable by every agent alone from the
publicly observable state (no communication):

1.  **Stay-disk avoidance** — an action is only safe if the swept path
    keeps more than the collision radius away from every other agent's
    *current* position.
2.  **Priority pruning** — agent i must additionally be safe against every
    action that survived filtering for the higher-priority agents j < i
    (priorities are the public agent indices).

Part 1 guarantees that "stand still" always survives (everyone else's safe
actions stay away from my position). Part 2 breaks the circularity of
"safe against whom?" without communication and gives pairwise safety for
any combination of independently chosen filtered actions.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .dynamics import unicycle_step
from .environment import GridWorld, legal_action_mask

_N_SWEEP_SAMPLES = 5


def _swept_paths(env: GridWorld, states: jnp.ndarray) -> jnp.ndarray:
    """Sample points of every agent's swept path under every action.

    states [n, 3] -> [n, A, S, 2], linearly interpolated over the step.
    """
    next_all = unicycle_step(states[:, None, :], env.actions, env.dt)  # [n,A,3]
    p0 = states[:, None, :2]                                            # [n,1,2]
    delta = next_all[..., :2] - p0                                      # [n,A,2]
    ts = jnp.linspace(0.0, 1.0, _N_SWEEP_SAMPLES)                       # [S]
    return p0[:, :, None, :] + ts[None, None, :, None] * delta[:, :, None, :]


def safe_action_mask(env: GridWorld, states: jnp.ndarray,
                     reached: jnp.ndarray, t=0) -> jnp.ndarray:
    """Per-agent action mask that is obstacle-legal AND collision-safe
    against every combination of the other agents' masked actions.

    states [n, 3], reached [n] -> mask [n, A] bool
    """
    n_agents, n_actions = env.actions.shape[0], env.actions.shape[1]
    radius = env.collision_radius

    legal = legal_action_mask(env, states, t)
    # agents that reached their goal are frozen: their only action is stay
    null_onehot = jnp.arange(n_actions)[None, :] == env.null_action[:, None]
    legal = jnp.where(reached[:, None], null_onehot, legal)

    seg = _swept_paths(env, states)  # [n, A, S, 2]

    # --- Condition 1: keep clear of every other agent's current position
    pos = states[:, :2]                                            # [n, 2]
    d_stay = jnp.linalg.norm(
        seg[:, :, :, None, :] - pos[None, None, None, :, :], axis=-1
    )                                                              # [n, A, S, n]
    self_mask = jnp.eye(n_agents, dtype=bool)                      # ignore self
    d_stay = jnp.where(self_mask[:, None, None, :], jnp.inf, d_stay)
    clear_of_stays = jnp.min(d_stay, axis=(2, 3)) > radius         # [n, A]

    # --- pairwise swept-vs-swept distances (same interpolation times)
    diff = seg[:, :, None, None, :, :] - seg[None, None, :, :, :, :]
    d_pair = jnp.linalg.norm(diff, axis=-1)                        # [n,A,n,B,S]
    conflicts = jnp.min(d_pair, axis=-1) <= radius                 # [n,A,n,B]

    # --- Condition 2: sequential priority pruning (agent 0 first)
    base = legal & clear_of_stays
    safe_rows = []
    for i in range(n_agents):
        row = base[i]
        for j in range(i):
            # unsafe if some already-safe action of a higher-priority agent
            # could sweep within the collision radius
            row = row & ~jnp.any(conflicts[i, :, j, :] & safe_rows[j], axis=-1)
        safe_rows.append(row)
    return jnp.stack(safe_rows)


def check_initial_separation(env: GridWorld) -> None:
    """Assumption A1 of the safety proof: agents start mutually separated."""
    pos = np.asarray(env.starts)[:, :2]
    dist = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    dist = dist + np.eye(env.n_agents) * 1e9
    min_dist = float(dist.min())
    if min_dist <= float(env.collision_radius):
        raise ValueError(
            f"safety filter requires pairwise start separation > "
            f"collision_radius ({float(env.collision_radius)}), got {min_dist}"
        )
