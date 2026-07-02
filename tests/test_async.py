import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sm_mcts_jax import AsyncDecentralizedPlanner, MCTSParams, ascii_world
from sm_mcts_jax.mcts import search
from sm_mcts_jax.rewards import RewardParams

INTERSECTION = """
##1##
##.##
0...a
##.##
##b##
"""

SMALL_PARAMS = MCTSParams(num_simulations=128, max_depth=8, rollout_depth=8,
                          k_rollouts=2)


def test_action_plan_shape_and_head():
    env = ascii_world(INTERSECTION)
    result = search(
        env, SMALL_PARAMS, RewardParams(), env.starts,
        jnp.zeros((env.n_agents,), bool), jnp.int32(0), jax.random.PRNGKey(0),
    )
    assert result.action_plan.shape == (SMALL_PARAMS.commit_depth, env.n_agents)
    assert np.array_equal(np.asarray(result.action_plan[0]),
                          np.asarray(result.action_idx))


def test_replan_schedule_matches_phases():
    env = ascii_world(INTERSECTION)
    planner = AsyncDecentralizedPlanner(
        env, SMALL_PARAMS, periods=[2, 2], phases=[0, 1], seed=0
    )
    traj = planner.run_episode(max_steps=12)
    replanned = traj.replanned
    # everyone plans at t = 0
    assert replanned[0].all()
    for t in range(1, replanned.shape[0]):
        assert replanned[t, 0] == (t % 2 == 0)   # phase 0: t = 0, 2, 4, ...
        assert replanned[t, 1] == (t % 2 == 1)   # phase 1: t = 0, 1, 3, ...
    # interleaved clocks never revise together after t = 0
    assert not (replanned[1:].sum(axis=1) == 2).any()


def test_async_episode_reaches_goals():
    env = ascii_world(INTERSECTION)
    planner = AsyncDecentralizedPlanner(
        env, SMALL_PARAMS, periods=[2, 2], phases=[0, 1], seed=1
    )
    traj = planner.run_episode(max_steps=30)
    assert traj.all_reached, traj.summary()


def test_commit_depth_must_cover_interval():
    env = ascii_world(INTERSECTION)
    with pytest.raises(ValueError):
        AsyncDecentralizedPlanner(env, SMALL_PARAMS, periods=[8, 8])
    with pytest.raises(ValueError):
        AsyncDecentralizedPlanner(env, SMALL_PARAMS, periods=[2, 2],
                                  phases=[0, 2])
