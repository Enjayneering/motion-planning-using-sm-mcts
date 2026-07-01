"""Centralized vs. decentralized SM-MCTS: statistical comparison.

For every scenario and mode, N independent episodes (different seeds) are
run. Reported per condition:

- success rate (all agents reach their goals within the step budget),
  with a 95% Wilson confidence interval
- number of episodes containing at least one collision
- steps to completion (mean +- std over successful episodes)
- planning latency per world step
- prediction consistency (decentralized only): how often agent i's search
  correctly anticipated agent j's executed action

Success and collision counts of the two modes are compared with Fisher's
exact test (two-sided). Results are cached as JSON so scenarios can be run
one at a time:

    python examples/experiment_decentralized.py head_on     [n_seeds]
    python examples/experiment_decentralized.py bottleneck  [n_seeds]
    python examples/experiment_decentralized.py gates       [n_seeds]
    python examples/experiment_decentralized.py report
"""

from __future__ import annotations

import json
import sys
from math import comb, sqrt
from pathlib import Path

from sm_mcts_jax import DecentralizedPlanner, MCTSParams, Planner, ascii_world

RESULTS = Path(__file__).with_name("experiment_results.json")

HEAD_ON = """
    ###########
    #.........#
    #b0.....1a#
    #.........#
    ###########
"""

BOTTLENECK = """
    #########
    #0..#..1#
    #...#...#
    #.......#
    #...#...#
    #b..#..a#
    #########
"""

GATE_LEFT = """
    0.........1
    d.........c
    ...........
    ...........
    ##.########
    ...........
    ...........
    b.........a
    2.........3
"""

GATE_RIGHT = """
    0.........1
    d.........c
    ...........
    ...........
    ########.##
    ...........
    ...........
    b.........a
    2.........3
"""

SCENARIOS = {
    "head_on": dict(
        env=lambda: ascii_world(HEAD_ON),
        params=MCTSParams(num_simulations=512),
        max_steps=40,
    ),
    "bottleneck": dict(
        env=lambda: ascii_world(BOTTLENECK),
        params=MCTSParams(num_simulations=512),
        max_steps=40,
    ),
    "gates": dict(
        env=lambda: ascii_world([GATE_LEFT, GATE_RIGHT], frame_duration=4,
                                cycle=True),
        params=MCTSParams(num_simulations=512, max_depth=12, rollout_depth=16,
                          k_rollouts=2, c_uct=1.2),
        max_steps=80,
    ),
}


# ---------------------------------------------------------------------------
# Statistics (dependency-free)
# ---------------------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact test for the 2x2 table [[a, b], [c, d]]."""
    n = a + b + c + d
    row1, col1 = a + b, a + c

    def hypergeom(k: int) -> float:
        return comb(row1, k) * comb(n - row1, col1 - k) / comb(n, col1)

    p_obs = hypergeom(a)
    k_min = max(0, col1 - (n - row1))
    k_max = min(row1, col1)
    return min(1.0, sum(
        hypergeom(k) for k in range(k_min, k_max + 1)
        if hypergeom(k) <= p_obs * (1 + 1e-9)
    ))


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def run_scenario(name: str, n_seeds: int) -> dict:
    spec = SCENARIOS[name]
    results: dict = {"n_seeds": n_seeds, "modes": {}}
    for mode in ("centralized", "decentralized"):
        episodes = []
        env = spec["env"]()
        for seed in range(n_seeds):
            cls = Planner if mode == "centralized" else DecentralizedPlanner
            planner = cls(env, spec["params"], seed=seed)
            traj = planner.run_episode(max_steps=spec["max_steps"])
            episode = {
                "seed": seed,
                "success": traj.all_reached,
                "collision_episodes": traj.any_collision,
                "collision_steps": int(traj.collisions.sum()),
                "steps": traj.n_steps,
                "plan_ms": 1e3 * float(traj.plan_times.mean()),
            }
            if mode == "decentralized":
                episode["consistency"] = traj.prediction_consistency
            episodes.append(episode)
            print(f"{name} {mode} seed={seed}: {traj.summary()}", flush=True)
        results["modes"][mode] = episodes
    return results


def _aggregate(episodes: list[dict]) -> dict:
    n = len(episodes)
    successes = sum(e["success"] for e in episodes)
    collisions = sum(e["collision_episodes"] for e in episodes)
    steps_ok = [e["steps"] for e in episodes if e["success"]]
    agg = {
        "n": n,
        "successes": successes,
        "success_ci": wilson_interval(successes, n),
        "collision_episodes": collisions,
        "steps_mean": (sum(steps_ok) / len(steps_ok)) if steps_ok else None,
        "steps_std": (
            sqrt(sum((s - sum(steps_ok) / len(steps_ok)) ** 2 for s in steps_ok)
                 / len(steps_ok)) if steps_ok else None
        ),
        "plan_ms": sum(e["plan_ms"] for e in episodes) / n,
    }
    cons = [e["consistency"] for e in episodes if "consistency" in e]
    if cons:
        agg["consistency"] = sum(cons) / len(cons)
    return agg


def report() -> None:
    data = json.loads(RESULTS.read_text())
    for name, res in data.items():
        cen = _aggregate(res["modes"]["centralized"])
        dec = _aggregate(res["modes"]["decentralized"])
        n = cen["n"]
        p_success = fisher_exact(
            cen["successes"], n - cen["successes"],
            dec["successes"], n - dec["successes"],
        )
        p_collision = fisher_exact(
            cen["collision_episodes"], n - cen["collision_episodes"],
            dec["collision_episodes"], n - dec["collision_episodes"],
        )
        print(f"\n=== {name} (n = {n} seeds per mode) ===")
        header = (f"{'':16s}{'success':>12s}{'95% CI':>18s}"
                  f"{'coll.epis':>10s}{'steps':>14s}{'plan/step':>11s}")
        print(header)
        for label, agg in (("centralized", cen), ("decentralized", dec)):
            lo, hi = agg["success_ci"]
            steps = (f"{agg['steps_mean']:.1f} ± {agg['steps_std']:.1f}"
                     if agg["steps_mean"] is not None else "—")
            print(f"{label:16s}{agg['successes']:>7d}/{agg['n']:<4d}"
                  f"{f'[{lo:.2f}, {hi:.2f}]':>18s}"
                  f"{agg['collision_episodes']:>10d}"
                  f"{steps:>14s}"
                  f"{agg['plan_ms']:>9.0f}ms")
        if "consistency" in dec:
            print(f"{'':16s}decentralized prediction consistency: "
                  f"{dec['consistency']:.2f}")
        print(f"{'':16s}Fisher exact p (success): {p_success:.3f}   "
              f"p (collision episodes): {p_collision:.3f}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "report":
        report()
        return
    name = sys.argv[1]
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[name] = run_scenario(name, n_seeds)
    RESULTS.write_text(json.dumps(data, indent=2))
    print(f"saved -> {RESULTS}")


if __name__ == "__main__":
    main()
