"""Decentralized planning: one independent SM-MCTS search per agent.

In the centralized `Planner`, a single search recommends the full joint
action. Here every agent runs its *own* search from the same observed world
state (with its own RNG stream), simulating the other agents inside its tree
to anticipate their reactions — but only executes its **own** action
component. The world then advances with the actions the agents chose
independently. Coordination is not imposed; it has to *emerge* from the
agents solving the same game.

This mirrors the mcts-vs-mcts experiments of the original repository and is
the scientifically interesting mode: in symmetric situations several
equilibria exist and independent searches may momentarily pick inconsistent
ones (both agents dodge to the same side). The per-step `predictions` record
lets you quantify this: agent i's search also outputs the action it expects
from agent j, which can be compared with what j actually did.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .environment import GridWorld, agents_collide, goal_distances, step_world
from .mcts import MCTSParams, search
from .planner import Trajectory
from .rewards import RewardParams


@dataclass
class DecentralizedTrajectory(Trajectory):
    """Episode record with per-agent predictions.

    predictions[t, i, j] = action index that agent i's search expected
    agent j to take at step t (the diagonal is what was executed).
    """

    predictions: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0, 0), np.int32)
    )

    @property
    def prediction_consistency(self) -> float:
        """Fraction of cross-predictions (i != j) that matched agent j's
        executed action, over all steps where j was still moving."""
        if self.predictions.size == 0:
            return float("nan")
        n_steps, n_agents, _ = self.predictions.shape
        executed = self.actions  # [T, n]
        active = ~self.reached[:-1]  # [T, n], reached at the start of the step
        hits, total = 0, 0
        for t in range(n_steps):
            for i in range(n_agents):
                for j in range(n_agents):
                    if i == j or not active[t, j]:
                        continue
                    hits += int(self.predictions[t, i, j] == executed[t, j])
                    total += 1
        return hits / total if total else float("nan")

    def summary(self) -> str:
        base = super().summary()
        return f"{base} consistency={self.prediction_consistency:.2f}"


class DecentralizedPlanner:
    """N independent searches per world timestep, one per agent.

    All agents share the same (correct) model of the world and of each
    other's payoffs — common knowledge, as in the original mcts-vs-mcts
    setup — but their searches are stochastically independent. The N
    searches are batched with `vmap` into a single XLA call.
    """

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
        self._search_all = jax.jit(
            jax.vmap(
                partial(search, env, self.mcts_params, self.reward_params),
                in_axes=(None, None, None, 0),
            )
        )

    def _next_keys(self) -> jax.Array:
        self._rng, sub = jax.random.split(self._rng)
        return jax.random.split(sub, self.env.n_agents)

    def warmup(self) -> float:
        t0 = time.perf_counter()
        result = self._search_all(
            self.env.starts,
            jnp.zeros((self.env.n_agents,), bool),
            jnp.int32(0),
            self._next_keys(),
        )
        jax.block_until_ready(result.action_idx)
        return time.perf_counter() - t0

    def plan(self, states: jnp.ndarray, reached: jnp.ndarray, t: int = 0):
        """Run all N searches; returns (own_actions [n], predictions [n, n])."""
        result = self._search_all(states, reached, jnp.int32(t), self._next_keys())
        predictions = result.action_idx          # [n_searches, n_agents]
        own_actions = jnp.diagonal(predictions)  # agent i executes row i, col i
        jax.block_until_ready(own_actions)
        return own_actions, predictions

    def run_episode(self, max_steps: int = 60,
                    verbose: bool = False) -> DecentralizedTrajectory:
        env = self.env
        states = env.starts
        reached = goal_distances(env, states) <= env.goal_radius

        states_log = [np.asarray(states)]
        reached_log = [np.asarray(reached)]
        actions_log, collision_log, time_log, prediction_log = [], [], [], []

        for step in range(max_steps):
            if bool(reached.all()):
                break

            t0 = time.perf_counter()
            own_actions, predictions = self.plan(states, reached, t=step)
            plan_time = time.perf_counter() - t0

            next_states, next_reached = step_world(env, states, reached, own_actions)
            collided = bool(jnp.any(agents_collide(env, states, next_states)))

            states_log.append(np.asarray(next_states))
            reached_log.append(np.asarray(next_reached))
            actions_log.append(np.asarray(own_actions))
            prediction_log.append(np.asarray(predictions))
            collision_log.append(collided)
            time_log.append(plan_time)

            if verbose:
                print(
                    f"t={step:3d} plan={plan_time * 1e3:7.1f}ms "
                    f"reached={np.asarray(next_reached).astype(int)} "
                    f"collision={collided}"
                )
            states, reached = next_states, next_reached

        n = env.n_agents
        return DecentralizedTrajectory(
            states=np.stack(states_log),
            reached=np.stack(reached_log),
            actions=(
                np.stack(actions_log) if actions_log
                else np.zeros((0, n), np.int32)
            ),
            collisions=np.asarray(collision_log, dtype=bool),
            plan_times=np.asarray(time_log, dtype=np.float64),
            predictions=(
                np.stack(prediction_log) if prediction_log
                else np.zeros((0, n, n), np.int32)
            ),
        )
