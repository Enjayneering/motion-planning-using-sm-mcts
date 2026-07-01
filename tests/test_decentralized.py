import numpy as np

from sm_mcts_jax import DecentralizedPlanner, MCTSParams, ascii_world

INTERSECTION = """
##1##
##.##
0...a
##.##
##b##
"""

SMALL_PARAMS = MCTSParams(num_simulations=128, max_depth=8, rollout_depth=8,
                          k_rollouts=2)


def test_decentralized_episode_reaches_goals():
    env = ascii_world(INTERSECTION)
    planner = DecentralizedPlanner(env, SMALL_PARAMS, seed=1)
    traj = planner.run_episode(max_steps=30)
    assert traj.all_reached, traj.summary()
    assert not traj.any_collision, traj.summary()


def test_predictions_shape_and_consistency_range():
    env = ascii_world(INTERSECTION)
    planner = DecentralizedPlanner(env, SMALL_PARAMS, seed=3)
    traj = planner.run_episode(max_steps=30)
    n_steps, n_agents = traj.actions.shape
    assert traj.predictions.shape == (n_steps, n_agents, n_agents)
    # executed actions are the diagonal of each prediction matrix
    for t in range(n_steps):
        assert np.array_equal(np.diagonal(traj.predictions[t]), traj.actions[t])
    assert 0.0 <= traj.prediction_consistency <= 1.0
