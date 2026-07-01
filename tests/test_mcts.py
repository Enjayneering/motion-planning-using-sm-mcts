import jax
import jax.numpy as jnp
import numpy as np

from sm_mcts_jax import MCTSParams, Planner, ascii_world, search
from sm_mcts_jax.mcts import decode_joint, encode_joint
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


def test_joint_action_roundtrip():
    n_agents, n_actions = 3, 6
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = jnp.asarray(rng.integers(0, n_actions, size=n_agents), dtype=jnp.int32)
        joint = encode_joint(a, n_actions)
        back = decode_joint(joint, n_agents, n_actions)
        assert np.array_equal(np.asarray(back), np.asarray(a))


def test_search_returns_legal_actions():
    env = ascii_world(INTERSECTION)
    result = search(
        env, SMALL_PARAMS, RewardParams(), env.starts,
        jnp.zeros((env.n_agents,), bool), jnp.int32(0), jax.random.PRNGKey(0),
    )
    assert result.action_idx.shape == (env.n_agents,)
    legal = np.asarray(
        __import__("sm_mcts_jax.environment", fromlist=["legal_action_mask"])
        .legal_action_mask(env, env.starts)
    )
    for agent, a in enumerate(np.asarray(result.action_idx)):
        assert legal[agent, a], "chosen root action must be obstacle-free"
    assert int(result.num_nodes) > 1


def test_episode_reaches_goals_without_collision():
    env = ascii_world(INTERSECTION)
    planner = Planner(env, mcts_params=SMALL_PARAMS, seed=1)
    traj = planner.run_episode(max_steps=30)
    assert traj.all_reached, traj.summary()
    assert not traj.any_collision, traj.summary()


def test_episode_three_agents():
    env = ascii_world("""
        .......
        .0...a.
        .......
        .b...1.
        .......
        .2.....
        .....c.
        """)
    planner = Planner(env, mcts_params=SMALL_PARAMS, seed=2)
    traj = planner.run_episode(max_steps=40)
    assert traj.all_reached, traj.summary()
