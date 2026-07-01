"""Kinodynamic model: discrete-action unicycle, fully vectorized.

State layout per agent: [x, y, theta]
Action layout:          [v, omega]
"""

from __future__ import annotations

import jax.numpy as jnp


def make_action_set(velocities, angular_velocities) -> jnp.ndarray:
    """Cartesian product of velocity and angular-velocity values.

    Returns an array of shape [n_actions, 2] with columns (v, omega).
    """
    v, w = jnp.meshgrid(
        jnp.asarray(velocities, dtype=jnp.float32),
        jnp.asarray(angular_velocities, dtype=jnp.float32),
        indexing="ij",
    )
    return jnp.stack([v.ravel(), w.ravel()], axis=-1)


def unicycle_step(state: jnp.ndarray, action: jnp.ndarray, dt) -> jnp.ndarray:
    """First-order unicycle integration. Broadcasts over leading dimensions.

    state:  [..., 3] (x, y, theta)
    action: [..., 2] (v, omega)
    """
    x, y, th = state[..., 0], state[..., 1], state[..., 2]
    v, w = action[..., 0], action[..., 1]
    x_new = x + v * jnp.cos(th) * dt
    y_new = y + v * jnp.sin(th) * dt
    # keep theta in (-pi, pi]
    th_new = jnp.mod(th + w * dt + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    return jnp.stack([x_new, y_new, th_new], axis=-1)
