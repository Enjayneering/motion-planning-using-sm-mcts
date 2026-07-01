"""Receding-horizon (MPC-style) planning loop on top of the SM-MCTS search.

Every world timestep, a full search is run from the current joint state and
the decoupled robust joint action is executed — exactly the closed-loop
scheme of the original implementation, but with a search that runs as one
JIT-compiled XLA call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .environment import GridWorld, agents_collide, goal_distances, step_world
from .mcts import MCTSParams, search
from .rewards import RewardParams


@dataclass
class Trajectory:
    """Closed-loop episode record (numpy, ready for plotting/analysis)."""

    states: np.ndarray        # [T+1, n_agents, 3]
    reached: np.ndarray       # [T+1, n_agents] bool
    actions: np.ndarray       # [T, n_agents] chosen action indices
    collisions: np.ndarray    # [T] bool, any collision during the transition
    plan_times: np.ndarray    # [T] wall-clock seconds per planning step
    root_visits: list = field(default_factory=list)  # per step [n, A]

    @property
    def n_steps(self) -> int:
        return self.actions.shape[0]

    @property
    def all_reached(self) -> bool:
        return bool(self.reached[-1].all())

    @property
    def any_collision(self) -> bool:
        return bool(self.collisions.any())

    def summary(self) -> str:
        avg_ms = 1e3 * float(self.plan_times.mean()) if self.n_steps else 0.0
        return (
            f"steps={self.n_steps} all_reached={self.all_reached} "
            f"collisions={int(self.collisions.sum())} "
            f"avg_plan_time={avg_ms:.1f}ms "
            f"({1.0 / max(avg_ms / 1e3, 1e-9):.1f} plans/s)"
        )


class Planner:
    """Convenience wrapper: holds the environment + search configuration."""

    def __init__(
        self,
        env: GridWorld,
        mcts_params: MCTSParams | None = None,
        reward_params: RewardParams | None = None,
        seed: int = 0,
    ):
        self.env = env
        self.mcts_params = mcts_params or MCTSParams()
        self.reward_params = reward_params or RewardParams()
        self._rng = jax.random.PRNGKey(seed)
        if self.mcts_params.safety_filter:
            from .safety import check_initial_separation
            check_initial_separation(env)

    def _next_key(self) -> jax.Array:
        self._rng, key = jax.random.split(self._rng)
        return key

    def warmup(self) -> float:
        """Trigger JIT compilation; returns compile wall-clock seconds."""
        t0 = time.perf_counter()
        result = search(
            self.env,
            self.mcts_params,
            self.reward_params,
            self.env.starts,
            jnp.zeros((self.env.n_agents,), bool),
            jnp.int32(0),
            self._next_key(),
        )
        jax.block_until_ready(result.action_idx)
        return time.perf_counter() - t0

    def plan(self, states: jnp.ndarray, reached: jnp.ndarray, t: int = 0):
        """One SM-MCTS search from the given joint state at world time t."""
        result = search(
            self.env, self.mcts_params, self.reward_params, states, reached,
            jnp.int32(t), self._next_key(),
        )
        jax.block_until_ready(result.action_idx)
        return result

    def run_episode(self, max_steps: int = 60, verbose: bool = False) -> Trajectory:
        """Closed-loop receding-horizon episode until all agents reach their
        goals (or `max_steps` is exhausted)."""
        env = self.env
        states = env.starts
        reached = goal_distances(env, states) <= env.goal_radius

        states_log = [np.asarray(states)]
        reached_log = [np.asarray(reached)]
        actions_log, collision_log, time_log, visits_log = [], [], [], []

        for step in range(max_steps):
            if bool(reached.all()):
                break

            t0 = time.perf_counter()
            result = self.plan(states, reached, t=step)
            plan_time = time.perf_counter() - t0

            next_states, next_reached = step_world(
                env, states, reached, result.action_idx
            )
            collided = bool(jnp.any(agents_collide(env, states, next_states)))

            states_log.append(np.asarray(next_states))
            reached_log.append(np.asarray(next_reached))
            actions_log.append(np.asarray(result.action_idx))
            collision_log.append(collided)
            time_log.append(plan_time)
            visits_log.append(np.asarray(result.root_visits))

            if verbose:
                print(
                    f"t={step:3d} plan={plan_time * 1e3:7.1f}ms "
                    f"nodes={int(result.num_nodes)} "
                    f"reached={np.asarray(next_reached).astype(int)} "
                    f"collision={collided}"
                )
            states, reached = next_states, next_reached

        return Trajectory(
            states=np.stack(states_log),
            reached=np.stack(reached_log),
            actions=(
                np.stack(actions_log)
                if actions_log
                else np.zeros((0, env.n_agents), np.int32)
            ),
            collisions=np.asarray(collision_log, dtype=bool),
            plan_times=np.asarray(time_log, dtype=np.float64),
            root_visits=visits_log,
        )
