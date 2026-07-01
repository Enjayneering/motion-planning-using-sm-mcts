"""Simultaneous-move MCTS with decoupled UCT (DUCT), fully in JAX.

The search follows Lanctot, Lisy & Winands (2013): every agent keeps its own
action statistics per node, actions are selected independently per agent, and
the joint action indexes the child node. In contrast to the original Python
object tree, the whole tree lives in preallocated arrays and one complete
search (selection / expansion / rollout / backpropagation for all
simulations) compiles to a single XLA program — this is what makes the
planner real-time capable.

Tree layout (N = max_nodes, n = n_agents, A = n_actions, J = A**n):

    states            [N, n, 3]   agent poses at the node
    reached           [N, n]      goal-reached flags
    depth             [N]         distance from the root in timesteps
    parent            [N]         parent node index (-1 for the root)
    action_from_parent[N]         joint action index that led here
    children          [N, J]      child node per joint action (-1 = unvisited)
    node_visits       [N]
    action_visits     [N, n, A]   decoupled per-agent visit counts
    action_qsum       [N, n, A]   decoupled per-agent payoff sums
    legal             [N, n, A]   obstacle-free actions per agent
    reward_to_node    [N, n]      per-agent reward of the edge parent -> node
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .dynamics import unicycle_step
from .environment import (
    GridWorld,
    goal_distances,
    legal_action_mask,
    segment_is_free,
    step_world,
)
from .rewards import RewardParams, transition_rewards

UNVISITED = jnp.int32(-1)


@dataclass(frozen=True)
class MCTSParams:
    """Static search configuration (hashable, triggers recompile on change)."""

    num_simulations: int = 512   # tree-search iterations per planning step
    max_depth: int = 12          # planning horizon inside the tree (timesteps)
    rollout_depth: int = 12      # heuristic rollout horizon beyond the leaf
    k_rollouts: int = 4          # rollouts averaged per leaf evaluation
    c_uct: float = 1.4           # UCT exploration constant
    discount: float = 0.95       # payoff discount per timestep
    rollout_temperature: float = 0.5  # Gumbel noise scale of the rollout policy

    @property
    def max_nodes(self) -> int:
        # each simulation expands at most one node
        return self.num_simulations + 2


class Tree(NamedTuple):
    states: jnp.ndarray
    reached: jnp.ndarray
    depth: jnp.ndarray
    parent: jnp.ndarray
    action_from_parent: jnp.ndarray
    children: jnp.ndarray
    node_visits: jnp.ndarray
    action_visits: jnp.ndarray
    action_qsum: jnp.ndarray
    legal: jnp.ndarray
    reward_to_node: jnp.ndarray
    num_nodes: jnp.ndarray


class SearchResult(NamedTuple):
    action_idx: jnp.ndarray    # [n_agents] chosen action per agent (robust)
    root_visits: jnp.ndarray   # [n_agents, n_actions] decoupled visit counts
    root_q: jnp.ndarray        # [n_agents, n_actions] decoupled mean payoffs
    num_nodes: jnp.ndarray     # scalar, expanded tree size


# ---------------------------------------------------------------------------
# Joint action index <-> per-agent action indices
# ---------------------------------------------------------------------------

def _powers(n_agents: int, n_actions: int) -> jnp.ndarray:
    return n_actions ** jnp.arange(n_agents, dtype=jnp.int32)


def encode_joint(action_idx: jnp.ndarray, n_actions: int) -> jnp.ndarray:
    return jnp.sum(action_idx * _powers(action_idx.shape[-1], n_actions), axis=-1)


def decode_joint(joint_idx: jnp.ndarray, n_agents: int, n_actions: int) -> jnp.ndarray:
    return (joint_idx // _powers(n_agents, n_actions)) % n_actions


# ---------------------------------------------------------------------------
# Search building blocks
# ---------------------------------------------------------------------------

def _init_tree(env: GridWorld, params: MCTSParams, root_state, root_reached) -> Tree:
    n_agents, n_actions = env.actions.shape[0], env.actions.shape[1]
    n_joint = n_actions ** n_agents
    N = params.max_nodes
    tree = Tree(
        states=jnp.zeros((N, n_agents, 3), jnp.float32),
        reached=jnp.zeros((N, n_agents), bool),
        depth=jnp.zeros((N,), jnp.int32),
        parent=jnp.full((N,), UNVISITED),
        action_from_parent=jnp.zeros((N,), jnp.int32),
        children=jnp.full((N, n_joint), UNVISITED),
        node_visits=jnp.zeros((N,), jnp.int32),
        action_visits=jnp.zeros((N, n_agents, n_actions), jnp.int32),
        action_qsum=jnp.zeros((N, n_agents, n_actions), jnp.float32),
        legal=jnp.zeros((N, n_agents, n_actions), bool),
        reward_to_node=jnp.zeros((N, n_agents), jnp.float32),
        num_nodes=jnp.int32(1),
    )
    return tree._replace(
        states=tree.states.at[0].set(root_state),
        reached=tree.reached.at[0].set(root_reached),
        legal=tree.legal.at[0].set(legal_action_mask(env, root_state)),
    )


def _node_is_terminal(tree: Tree, params: MCTSParams, node) -> jnp.ndarray:
    return (tree.depth[node] >= params.max_depth) | jnp.all(tree.reached[node])


def _choose_joint_action(env: GridWorld, params: MCTSParams, tree: Tree,
                         node, rng) -> jnp.ndarray:
    """Decoupled UCT: independent argmax per agent, then joint encoding."""
    visits = tree.action_visits[node].astype(jnp.float32)      # [n, A]
    qsum = tree.action_qsum[node]
    legal = tree.legal[node]
    node_visits = tree.node_visits[node].astype(jnp.float32)

    q = qsum / jnp.maximum(visits, 1.0)
    explore = params.c_uct * jnp.sqrt(
        jnp.log(node_visits + 1.0) / jnp.maximum(visits, 1.0)
    )
    noise = jax.random.uniform(rng, visits.shape)
    score = q + explore + 1e-6 * noise            # tiny noise breaks ties
    score = jnp.where(visits == 0, 1e6 + noise, score)  # try unvisited first
    score = jnp.where(legal, score, -jnp.inf)

    action_idx = jnp.argmax(score, axis=-1).astype(jnp.int32)  # [n_agents]
    # agents that already reached their goal are pinned to the null action
    action_idx = jnp.where(tree.reached[node], env.null_action, action_idx)
    return encode_joint(action_idx, env.n_actions)


def _select(env, params, tree: Tree, rng):
    """Descend from the root until an unexpanded joint action or a terminal
    node is hit. Returns (node, joint_action, child)."""

    def cond_fun(carry):
        _, _, _, child = carry
        return (child != UNVISITED) & ~_node_is_terminal(tree, params, child)

    def body_fun(carry):
        rng, _, _, child = carry
        rng, sub = jax.random.split(rng)
        node = child
        ja = _choose_joint_action(env, params, tree, node, sub)
        return rng, node, ja, tree.children[node, ja]

    rng, sub = jax.random.split(rng)
    ja0 = _choose_joint_action(env, params, tree, 0, sub)
    init = (rng, jnp.int32(0), ja0, tree.children[0, ja0])
    _, node, ja, child = jax.lax.while_loop(cond_fun, body_fun, init)
    return node, ja, child


def _expand(env, params, reward_params, tree: Tree, parent, joint_action):
    """Materialize the child of (parent, joint_action) as a new node."""
    idx = tree.num_nodes
    action_idx = decode_joint(joint_action, env.n_agents, env.n_actions)
    prev_states = tree.states[parent]
    prev_reached = tree.reached[parent]
    next_states, next_reached = step_world(env, prev_states, prev_reached, action_idx)
    reward = transition_rewards(
        env, reward_params, prev_states, next_states, prev_reached, next_reached
    )
    tree = tree._replace(
        states=tree.states.at[idx].set(next_states),
        reached=tree.reached.at[idx].set(next_reached),
        depth=tree.depth.at[idx].set(tree.depth[parent] + 1),
        parent=tree.parent.at[idx].set(parent),
        action_from_parent=tree.action_from_parent.at[idx].set(joint_action),
        children=tree.children.at[parent, joint_action].set(idx),
        legal=tree.legal.at[idx].set(legal_action_mask(env, next_states)),
        reward_to_node=tree.reward_to_node.at[idx].set(reward),
        num_nodes=tree.num_nodes + 1,
    )
    return tree, idx


def _rollout_policy_step(env: GridWorld, params: MCTSParams, states, reached, rng):
    """Goal-directed stochastic policy: Gumbel-perturbed greedy progress."""
    next_all = unicycle_step(states[:, None, :], env.actions, env.dt)  # [n, A, 3]
    p0 = jnp.broadcast_to(states[:, None, :2], next_all[..., :2].shape)
    legal = segment_is_free(env, p0, next_all[..., :2])                # [n, A]

    dist_now = goal_distances(env, states)[:, None]                    # [n, 1]
    dist_next = jnp.linalg.norm(
        next_all[..., :2] - env.goals[:, None, :], axis=-1
    )                                                                  # [n, A]
    v_max = jnp.max(jnp.abs(env.actions[..., 0]), axis=-1, keepdims=True)
    progress = (dist_now - dist_next) / jnp.maximum(v_max * env.dt, 1e-6)

    gumbel = jax.random.gumbel(rng, progress.shape)
    score = progress + params.rollout_temperature * gumbel
    score = jnp.where(legal, score, -jnp.inf)
    action_idx = jnp.argmax(score, axis=-1).astype(jnp.int32)
    return jnp.where(reached, env.null_action, action_idx)


def _rollout(env, params, reward_params, states, reached, rng) -> jnp.ndarray:
    """Simulate `rollout_depth` steps; returns discounted per-agent payoff."""

    def step(carry, rng):
        states, reached, disc = carry
        action_idx = _rollout_policy_step(env, params, states, reached, rng)
        next_states, next_reached = step_world(env, states, reached, action_idx)
        reward = transition_rewards(
            env, reward_params, states, next_states, reached, next_reached
        )
        carry = (next_states, next_reached, disc * params.discount)
        return carry, disc * reward

    keys = jax.random.split(rng, params.rollout_depth)
    _, rewards = jax.lax.scan(step, (states, reached, jnp.float32(1.0)), keys)
    return jnp.sum(rewards, axis=0)  # [n_agents]


def _backup(env, params, tree: Tree, leaf, value):
    """Propagate the per-agent value from the leaf back to the root."""
    agent_ids = jnp.arange(env.n_agents)
    tree = tree._replace(node_visits=tree.node_visits.at[leaf].add(1))

    def cond_fun(carry):
        _, node, _ = carry
        return node != 0

    def body_fun(carry):
        tree, node, value = carry
        parent = tree.parent[node]
        action_idx = decode_joint(
            tree.action_from_parent[node], env.n_agents, env.n_actions
        )
        value = tree.reward_to_node[node] + params.discount * value
        tree = tree._replace(
            action_visits=tree.action_visits.at[parent, agent_ids, action_idx].add(1),
            action_qsum=tree.action_qsum.at[parent, agent_ids, action_idx].add(value),
            node_visits=tree.node_visits.at[parent].add(1),
        )
        return tree, parent, value

    tree, _, _ = jax.lax.while_loop(cond_fun, body_fun, (tree, leaf, value))
    return tree


# ---------------------------------------------------------------------------
# Full search
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("params", "reward_params"))
def search(
    env: GridWorld,
    params: MCTSParams,
    reward_params: RewardParams,
    root_state: jnp.ndarray,    # [n_agents, 3]
    root_reached: jnp.ndarray,  # [n_agents] bool
    rng: jax.Array,
) -> SearchResult:
    """Run one full SM-MCTS search and pick per-agent robust actions."""
    tree = _init_tree(env, params, root_state, root_reached)

    def one_simulation(_, carry):
        tree, rng = carry
        rng, k_select, k_rollout = jax.random.split(rng, 3)

        node, joint_action, child = _select(env, params, tree, rng=k_select)
        expand_needed = (child == UNVISITED) & ~_node_is_terminal(tree, params, node)

        tree, leaf = jax.lax.cond(
            expand_needed,
            lambda t: _expand(env, params, reward_params, t, node, joint_action),
            lambda t: (t, jnp.where(child == UNVISITED, node, child)),
            tree,
        )

        rollout_keys = jax.random.split(k_rollout, params.k_rollouts)
        returns = jax.vmap(
            lambda k: _rollout(
                env, params, reward_params, tree.states[leaf], tree.reached[leaf], k
            )
        )(rollout_keys)
        value = jnp.mean(returns, axis=0)

        tree = _backup(env, params, tree, leaf, value)
        return tree, rng

    tree, _ = jax.lax.fori_loop(
        0, params.num_simulations, one_simulation, (tree, rng)
    )

    # final move: robust-separate (most visited action per agent), as in the
    # original implementation's 'robust-separate' feature flag
    root_visits = tree.action_visits[0]
    root_q = tree.action_qsum[0] / jnp.maximum(root_visits, 1)
    masked_visits = jnp.where(tree.legal[0], root_visits, -1)
    action_idx = jnp.argmax(masked_visits, axis=-1).astype(jnp.int32)
    action_idx = jnp.where(root_reached, env.null_action, action_idx)

    return SearchResult(
        action_idx=action_idx,
        root_visits=root_visits,
        root_q=root_q,
        num_nodes=tree.num_nodes,
    )
