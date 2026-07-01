"""Tests for the maximin safety filter, including an exhaustive check of the
theorem: every combination of independently chosen safe actions is
collision-free, and the safe set is never empty."""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from sm_mcts_jax import DecentralizedPlanner, MCTSParams, Planner, ascii_world
from sm_mcts_jax.environment import agents_collide, step_world
from sm_mcts_jax.safety import check_initial_separation, safe_action_mask

HEAD_ON = """
###########
#.........#
#b0.....1a#
#.........#
###########
"""

SAFE_PARAMS = MCTSParams(num_simulations=128, max_depth=8, rollout_depth=8,
                         k_rollouts=2, safety_filter=True)


def _open_room_env():
    return ascii_world("""
        .....
        .0.1.
        .....
        .b.a.
        .....
    """)


def test_stay_action_always_safe():
    """Lemma 1: standing still survives the filter in every configuration."""
    env = _open_room_env()
    rng = np.random.default_rng(0)
    free = np.argwhere(~np.asarray(env.occupancy[0]))
    for _ in range(50):
        cells = free[rng.choice(len(free), size=env.n_agents, replace=False)]
        headings = rng.choice([0.0, np.pi / 2, np.pi, -np.pi / 2],
                              size=env.n_agents)
        states = jnp.asarray(
            np.column_stack([cells[:, 1], cells[:, 0], headings]),
            dtype=jnp.float32,
        )
        mask = np.asarray(
            safe_action_mask(env, states, jnp.zeros(env.n_agents, bool))
        )
        for i in range(env.n_agents):
            assert mask[i, int(env.null_action[i])], (
                f"stay filtered out for agent {i} at {np.asarray(states)}"
            )
            assert mask[i].any()  # theorem's non-emptiness


def test_every_safe_combination_is_collision_free():
    """Theorem: exhaustive product over the masked sets never collides."""
    env = _open_room_env()
    # adversarial configuration: two agents face to face, one cell apart
    states = jnp.asarray(
        [[1.0, 2.0, 0.0], [2.0, 2.0, np.pi]], dtype=jnp.float32
    )
    env2 = ascii_world("""
        .....
        .0.1.
        .....
        .b.a.
        .....
    """)
    reached = jnp.zeros(2, bool)
    mask = np.asarray(safe_action_mask(env2, states, reached))
    assert mask.any(axis=1).all()
    choices = [np.flatnonzero(mask[i]) for i in range(2)]
    for combo in itertools.product(*choices):
        action_idx = jnp.asarray(combo, dtype=jnp.int32)
        next_states, _ = step_world(env2, states, reached, action_idx)
        assert not bool(jnp.any(agents_collide(env2, states, next_states))), (
            f"collision for safe combination {combo}"
        )


def test_moving_into_others_cell_is_filtered():
    env = _open_room_env()
    states = jnp.asarray(
        [[1.0, 2.0, 0.0], [2.0, 2.0, np.pi]], dtype=jnp.float32
    )
    mask = np.asarray(safe_action_mask(env, states, jnp.zeros(2, bool)))
    forward = np.asarray(env.actions[0][:, 0]) > 0
    # agent 0 faces agent 1 head on: driving forward would enter its stay disk
    assert not mask[0, forward].any()


def test_filtered_episode_has_zero_collisions():
    env = ascii_world(HEAD_ON)
    # seed 17 produced a transient collision without the filter
    planner = DecentralizedPlanner(env, SAFE_PARAMS, seed=17)
    traj = planner.run_episode(max_steps=40)
    assert not traj.any_collision, traj.summary()
    assert traj.all_reached, traj.summary()


def test_centralized_filtered_episode():
    env = ascii_world(HEAD_ON)
    planner = Planner(env, SAFE_PARAMS, seed=0)
    traj = planner.run_episode(max_steps=40)
    assert not traj.any_collision, traj.summary()
    assert traj.all_reached, traj.summary()


def test_initial_separation_check():
    env = _open_room_env()
    check_initial_separation(env)  # fine
    bad = env._replace(collision_radius=jnp.float32(5.0))
    with pytest.raises(ValueError):
        check_initial_separation(bad)
