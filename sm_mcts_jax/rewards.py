"""Per-agent transition payoffs (general-sum, decoupled).

Mirrors the payoff structure of the original SM-MCTS implementation
(progress / collision / goal components with configurable weights), but
computed as one vectorized function over all agents.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .environment import GridWorld, agents_collide, goal_potential


@dataclass(frozen=True)
class RewardParams:
    """Payoff weights. Each weight is either a scalar (shared by all agents)
    or a tuple with one entry per agent — heterogeneous "personalities"
    (cautious/aggressive) while staying hashable/static for the jit."""

    weight_progress: float | tuple = 1.0   # moving towards the own goal
    weight_collision: float | tuple = 4.0  # violating the collision radius
    weight_proximity: float | tuple = 0.5  # soft penalty for closeness
    weight_goal: float | tuple = 3.0       # one-time bonus at the goal
    weight_time: float | tuple = 0.05      # per-step cost until arrival


def transition_rewards(
    env: GridWorld,
    params: RewardParams,
    prev_states: jnp.ndarray,   # [n_agents, 3]
    next_states: jnp.ndarray,   # [n_agents, 3]
    prev_reached: jnp.ndarray,  # [n_agents] bool
    next_reached: jnp.ndarray,  # [n_agents] bool
    t_next=0,                   # world timestep at which next_states holds
) -> jnp.ndarray:
    """Reward vector [n_agents] for one joint transition."""
    v_max = jnp.max(jnp.abs(env.actions[..., 0]), axis=-1)  # [n_agents]
    max_step = jnp.maximum(v_max * env.dt, 1e-6)

    # progress in time-expanded steps-to-goal, roughly [-1, 1] per step;
    # departure and arrival potential are taken at their own timesteps, so
    # waiting for a scheduled opening counts as progress
    progress = (
        goal_potential(env, prev_states, t_next - 1)
        - goal_potential(env, next_states, t_next)
    ) / max_step
    progress = jnp.clip(progress, -2.0, 2.0)  # robust to field jumps

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

    w = lambda value: jnp.asarray(value, dtype=jnp.float32)  # scalar or [n]
    reward = (
        w(params.weight_progress) * progress
        - w(params.weight_collision) * collided.astype(jnp.float32)
        - w(params.weight_proximity) * proximity
        + w(params.weight_goal) * arrived_now.astype(jnp.float32)
        - w(params.weight_time)
    )
    return jnp.where(active, reward, 0.0)
