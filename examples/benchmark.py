"""Measure planning latency for different simulation budgets and agent counts.

Run with `python examples/benchmark.py`. On a GPU (pip install "jax[cuda12]")
the same code runs without modification and substantially faster.
"""

import jax

from sm_mcts_jax import MCTSParams, Planner, ascii_world

TWO_AGENTS = """
    ##1##
    ##.##
    0...a
    ##.##
    ##b##
"""

THREE_AGENTS = """
    .......
    .0...a.
    .......
    .b...1.
    .......
    .2.....
    .....c.
"""

FOUR_AGENTS = """
    0......c
    ........
    ...##...
    2..##..a
    ........
    d......1
    ........
    b......3
"""


def bench(name: str, art: str, sims: int):
    env = ascii_world(art)
    planner = Planner(env, MCTSParams(num_simulations=sims), seed=0)
    compile_s = planner.warmup()
    traj = planner.run_episode(max_steps=60)
    print(
        f"{name:12s} agents={env.n_agents} sims={sims:5d} "
        f"compile={compile_s:5.1f}s  {traj.summary()}"
    )


def main():
    print(f"jax {jax.__version__} on {jax.devices()}")
    for sims in (256, 512, 1024):
        bench("intersection", TWO_AGENTS, sims)
    for sims in (256, 512):
        bench("three-agents", THREE_AGENTS, sims)
        bench("four-agents", FOUR_AGENTS, sims)


if __name__ == "__main__":
    main()
