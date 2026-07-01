import jax.numpy as jnp
import numpy as np

from sm_mcts_jax.dynamics import make_action_set, unicycle_step


def test_action_set_shape():
    actions = make_action_set([0.0, 1.0], [-1.0, 0.0, 1.0])
    assert actions.shape == (6, 2)
    assert np.isclose(np.asarray(actions)[:, 0].max(), 1.0)


def test_straight_motion():
    state = jnp.array([0.0, 0.0, 0.0])
    action = jnp.array([1.0, 0.0])
    nxt = np.asarray(unicycle_step(state, action, dt=1.0))
    assert np.allclose(nxt, [1.0, 0.0, 0.0], atol=1e-6)


def test_turn_in_place_wraps_angle():
    state = jnp.array([2.0, 3.0, jnp.pi * 0.75])
    action = jnp.array([0.0, jnp.pi * 0.75])
    nxt = np.asarray(unicycle_step(state, action, dt=1.0))
    assert np.allclose(nxt[:2], [2.0, 3.0], atol=1e-6)
    assert -np.pi < nxt[2] <= np.pi  # wrapped into (-pi, pi]
    assert np.isclose(nxt[2], -np.pi / 2, atol=1e-5)


def test_batched_step():
    states = jnp.zeros((4, 3))
    actions = jnp.tile(jnp.array([1.0, 0.0]), (4, 1))
    nxt = unicycle_step(states, actions, dt=0.5)
    assert nxt.shape == (4, 3)
    assert np.allclose(np.asarray(nxt)[:, 0], 0.5)
