"""Per-agent transition payoffs (general-sum, decoupled).

Mirrors the payoff structure of the original SM-MCTS implementation
(progress / collision / goal components with configurable weights), but
computed as one vectorized function over all agents.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .environment import GridWorld, agents_collide, goal_distances


@dataclass(frozen=True)
class RewardParams:
    weight_progress: float = 1.0    # reward for moving towards the own goal
    weight_collision: float = 4.0   # penalty for violating the collision radius
    weight_proximity: float = 0.5   # soft penalty for getting close to others
    weight_goal: float = 3.0        # one-time bonus for reaching the goal
    weight_time: float = 0.05       # per-step cost while not at the goal


def transition_rewards(
    env: GridWorld,
    params: RewardParams,
    prev_states: jnp.ndarray,   # [n_agents, 3]
    next_states: jnp.ndarray,   # [n_agents, 3]
    prev_reached: jnp.ndarray,  # [n_agents] bool
    next_reached: jnp.ndarray,  # [n_agents] bool
) -> jnp.ndarray:
    """Reward vector [n_agents] for one joint transition."""
    v_max = jnp.max(jnp.abs(env.actions[..., 0]), axis=-1)  # [n_agents]
    max_step = jnp.maximum(v_max * env.dt, 1e-6)

    # progress towards the own goal, normalized to [-1, 1] per step
    progress = (
        goal_distances(env, prev_states) - goal_distances(env, next_states)
    ) / max_step

    # hard collision along the swept transition
    collided = agents_collide(env, prev_states, next_states)

    # soft proximity shaping (Gaussian bump around other agents)
    diff = next_states[:, None, :2] - next_states[None, :, :2]
    dist = jnp.linalg.norm(diff, axis=-1)
    dist = dist + jnp.eye(dist.shape[0]) * 1e9
    nearest = jnp.min(dist, axis=-1)
    proximity = jnp.exp(-0.5 * (nearest / jnp.maximum(env.collision_radius, 1e-6)) ** 2)

    arrived_now = next_reached & ~prev_reached
    active = ~prev_reached  # frozen agents collect no further payoff

    reward = (
        params.weight_progress * progress
        - params.weight_collision * collided.astype(jnp.float32)
        - params.weight_proximity * proximity
        + params.weight_goal * arrived_now.astype(jnp.float32)
        - params.weight_time
    )
    return jnp.where(active, reward, 0.0)
