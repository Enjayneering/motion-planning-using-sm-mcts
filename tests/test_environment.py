import jax.numpy as jnp
import numpy as np
import pytest

from sm_mcts_jax.environment import (
    ascii_world,
    legal_action_mask,
    points_are_free,
    step_world,
)

INTERSECTION = """
##1##
##.##
0...a
##.##
##b##
"""


def test_ascii_parsing():
    env = ascii_world(INTERSECTION)
    assert env.n_agents == 2
    assert env.occupancy.shape == (5, 5)
    # agent 0: left -> right along the middle row (y = 2)
    assert np.allclose(np.asarray(env.starts)[0, :2], [0.0, 2.0])
    assert np.allclose(np.asarray(env.goals)[0], [4.0, 2.0])
    # agent 1: top -> bottom along the middle column (x = 2)
    assert np.allclose(np.asarray(env.starts)[1, :2], [2.0, 4.0])
    assert np.allclose(np.asarray(env.goals)[1], [2.0, 0.0])


def test_mismatched_agents_raise():
    with pytest.raises(ValueError):
        ascii_world("""
        0.a
        .b.
        """)


def test_points_are_free():
    env = ascii_world(INTERSECTION)
    free = points_are_free(env, jnp.array([[0.0, 2.0], [0.0, 0.0], [-3.0, 2.0]]))
    assert np.asarray(free).tolist() == [True, False, False]


def test_legal_mask_blocks_walls():
    env = ascii_world(INTERSECTION)
    # agent 0 turned to face the wall above it, agent 1 at its start
    states = jnp.array(env.starts).at[0, 2].set(jnp.pi / 2)
    mask = np.asarray(legal_action_mask(env, states))
    assert mask.shape == (2, env.n_actions)
    # standing still is always legal from a free cell
    assert mask[0, int(env.null_action[0])]
    assert mask.any(axis=1).all()
    # driving forward into the wall is illegal (actions with v=1)
    forward = np.asarray(env.actions[0][:, 0]) > 0
    assert not mask[0, forward].any()


def test_reached_agents_freeze():
    env = ascii_world(INTERSECTION)
    states = env.starts
    reached = jnp.array([True, False])
    action_idx = jnp.array([env.n_actions - 1, int(env.null_action[1])])
    nxt, _ = step_world(env, states, reached, action_idx)
    assert np.allclose(np.asarray(nxt)[0], np.asarray(states)[0])
