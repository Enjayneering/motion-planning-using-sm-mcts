"""Asynchronous decentralized planning: staggered replanning clocks.

The classical game models are synchronous — either agents move strictly in
turns (sequential) or they all decide at the same instant (simultaneous).
Real robots do neither: each runs its own control loop, and the loops are
not phase-locked. This module models that middle ground in discrete time:

- Agent i replans at world step t = 0 and then on its own schedule, given
  by a period, a phase offset, and optionally a random jitter per replan
  (a discrete stand-in for independent Poisson clocks).
- Between replans the agent executes its **committed plan** — the robust
  action sequence along the principal variation of its last search
  (``SearchResult.action_plan``). It does not react in between.
- When it replans, it observes only the *current world state* (positions,
  headings, goal flags) — not the others' committed plans.

Why this matters (docs/ASYNC.md): simultaneous replanning is what makes
the both-dodge-the-same-way conflict possible — both agents revise their
strategy at the same instant, based on the same stale picture, and can
keep mirroring each other. With staggered clocks, whoever replans next
reacts to what the other has already *committed*, which is exactly the
setting in which asynchronous best-response dynamics are known to settle
into pure equilibria instead of cycling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .environment import GridWorld, agents_collide, goal_distances, step_world
from .mcts import MCTSParams, search
from .planner import Trajectory
from .rewards import RewardParams


@dataclass
class AsyncTrajectory(Trajectory):
    """Episode record; ``replanned[t, i]`` marks agent i replanning at t."""

    replanned: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), bool)
    )

    @property
    def replans_per_agent(self) -> np.ndarray:
        return self.replanned.sum(axis=0)


class AsyncDecentralizedPlanner:
    """One independent search per agent, on individually staggered clocks.

    periods[i]: steps between agent i's replans (1 = every step).
    phases[i]:  offset of agent i's schedule after the mandatory initial
                plan at t = 0 (must be < periods[i]).
    jitter:     if > 0, each interval is lengthened by a uniform random
                integer in [0, jitter] — stochastic, non-phase-locked
                clocks (the discrete analogue of Poisson revision clocks).
    """

    def __init__(
        self,
        env: GridWorld,
        mcts_params: MCTSParams | None = None,
        reward_params: RewardParams | None = None,
        periods=None,
        phases=None,
        jitter: int = 0,
        seed: int = 0,
    ):
        self.env = env
        self.mcts_params = mcts_params or MCTSParams()
        self.reward_params = reward_params or RewardParams()
        n = env.n_agents
        self.periods = list(periods) if periods is not None else [1] * n
        self.phases = list(phases) if phases is not None else [0] * n
        self.jitter = int(jitter)
        if len(self.periods) != n or len(self.phases) != n:
            raise ValueError("periods/phases must have one entry per agent")
        if any(p < 1 for p in self.periods):
            raise ValueError("periods must be >= 1")
        if any(not 0 <= ph < p for ph, p in zip(self.phases, self.periods)):
            raise ValueError("phases must satisfy 0 <= phase < period")
        max_interval = max(self.periods) + self.jitter
        if self.mcts_params.commit_depth < max_interval:
            raise ValueError(
                f"commit_depth ({self.mcts_params.commit_depth}) must cover "
                f"the longest replan interval ({max_interval})"
            )
        if self.mcts_params.safety_filter:
            from .safety import check_initial_separation
            check_initial_separation(env)
        self._rng = jax.random.PRNGKey(seed)
        self._np_rng = np.random.default_rng(seed)

    def _next_key(self) -> jax.Array:
        self._rng, key = jax.random.split(self._rng)
        return key

    def warmup(self) -> float:
        t0 = time.perf_counter()
        result = search(
            self.env, self.mcts_params, self.reward_params, self.env.starts,
            jnp.zeros((self.env.n_agents,), bool), jnp.int32(0),
            self._next_key(),
        )
        jax.block_until_ready(result.action_plan)
        return time.perf_counter() - t0

    def _next_interval(self, agent: int) -> int:
        extra = int(self._np_rng.integers(0, self.jitter + 1)) if self.jitter else 0
        return self.periods[agent] + extra

    def run_episode(self, max_steps: int = 60,
                    verbose: bool = False) -> AsyncTrajectory:
        env = self.env
        n = env.n_agents
        depth = self.mcts_params.commit_depth
        states = env.starts
        reached = goal_distances(env, states) <= env.goal_radius

        plans = np.zeros((n, depth), np.int32)
        cursor = np.zeros(n, np.int32)
        next_replan = np.zeros(n, np.int64)  # everyone plans at t = 0

        states_log = [np.asarray(states)]
        reached_log = [np.asarray(reached)]
        actions_log, collision_log, time_log, replan_log = [], [], [], []

        for t in range(max_steps):
            if bool(reached.all()):
                break

            replanning = [i for i in range(n) if t >= next_replan[i]]
            t0 = time.perf_counter()
            for i in replanning:
                result = search(
                    env, self.mcts_params, self.reward_params, states,
                    reached, jnp.int32(t), self._next_key(),
                )
                plans[i] = np.asarray(result.action_plan)[:, i]
                cursor[i] = 0
                if t == 0 and self.phases[i] > 0:
                    next_replan[i] = self.phases[i]
                else:
                    next_replan[i] = t + self._next_interval(i)
            plan_time = time.perf_counter() - t0

            action_idx = jnp.asarray(
                [plans[i][min(int(cursor[i]), depth - 1)] for i in range(n)],
                dtype=jnp.int32,
            )
            cursor += 1

            next_states, next_reached = step_world(env, states, reached, action_idx)
            collided = bool(jnp.any(agents_collide(env, states, next_states)))

            states_log.append(np.asarray(next_states))
            reached_log.append(np.asarray(next_reached))
            actions_log.append(np.asarray(action_idx))
            collision_log.append(collided)
            time_log.append(plan_time)
            replan_mask = np.zeros(n, bool)
            replan_mask[replanning] = True
            replan_log.append(replan_mask)

            if verbose:
                print(
                    f"t={t:3d} replan={replan_mask.astype(int)} "
                    f"plan={plan_time * 1e3:7.1f}ms "
                    f"reached={np.asarray(next_reached).astype(int)} "
                    f"collision={collided}"
                )
            states, reached = next_states, next_reached

        return AsyncTrajectory(
            states=np.stack(states_log),
            reached=np.stack(reached_log),
            actions=(
                np.stack(actions_log) if actions_log
                else np.zeros((0, n), np.int32)
            ),
            collisions=np.asarray(collision_log, dtype=bool),
            plan_times=np.asarray(time_log, dtype=np.float64),
            replanned=(
                np.stack(replan_log) if replan_log
                else np.zeros((0, n), bool)
            ),
        )
