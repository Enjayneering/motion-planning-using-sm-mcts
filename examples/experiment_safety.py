"""Safety filter on vs. off: decentralized planning, N seeds per condition.

The theorem (docs/SAFETY.md) predicts *exactly zero* collision steps with
the filter enabled — not approximately zero. The experiment verifies this
end-to-end through the search + world loop and measures the price of the
guarantee: extra steps per episode (conservatism) and planning latency.

    python examples/experiment_safety.py head_on     [n_seeds]
    python examples/experiment_safety.py bottleneck  [n_seeds]
    python examples/experiment_safety.py gates       [n_seeds]
    python examples/experiment_safety.py report
"""

from __future__ import annotations

import dataclasses
import json
import sys
from math import sqrt
from pathlib import Path

from sm_mcts_jax import DecentralizedPlanner
from experiment_decentralized import SCENARIOS, wilson_interval

RESULTS = Path(__file__).with_name("experiment_safety_results.json")


def run_scenario(name: str, n_seeds: int) -> dict:
    spec = SCENARIOS[name]
    results: dict = {"n_seeds": n_seeds, "conditions": {}}
    for filtered in (False, True):
        label = "filter_on" if filtered else "filter_off"
        params = dataclasses.replace(spec["params"], safety_filter=filtered)
        episodes = []
        env = spec["env"]()
        for seed in range(n_seeds):
            planner = DecentralizedPlanner(env, params, seed=seed)
            traj = planner.run_episode(max_steps=spec["max_steps"])
            episodes.append({
                "seed": seed,
                "success": traj.all_reached,
                "collision_steps": int(traj.collisions.sum()),
                "steps": traj.n_steps,
                "plan_ms": 1e3 * float(traj.plan_times.mean()),
                "consistency": traj.prediction_consistency,
            })
            print(f"{name} {label} seed={seed}: {traj.summary()}", flush=True)
        results["conditions"][label] = episodes
    return results


def report() -> None:
    data = json.loads(RESULTS.read_text())
    for name, res in data.items():
        print(f"\n=== {name} (n = {res['n_seeds']} seeds per condition, "
              f"decentralized) ===")
        print(f"{'':12s}{'success':>10s}{'95% CI':>16s}{'coll.steps':>12s}"
              f"{'steps':>14s}{'plan/step':>11s}{'consist.':>10s}")
        for label, eps in res["conditions"].items():
            n = len(eps)
            succ = sum(e["success"] for e in eps)
            lo, hi = wilson_interval(succ, n)
            coll = sum(e["collision_steps"] for e in eps)
            steps_ok = [e["steps"] for e in eps if e["success"]]
            mean = sum(steps_ok) / len(steps_ok) if steps_ok else float("nan")
            std = (sqrt(sum((s - mean) ** 2 for s in steps_ok) / len(steps_ok))
                   if steps_ok else float("nan"))
            ms = sum(e["plan_ms"] for e in eps) / n
            cons = sum(e["consistency"] for e in eps) / n
            print(f"{label:12s}{succ:>6d}/{n:<3d}{f'[{lo:.2f}, {hi:.2f}]':>16s}"
                  f"{coll:>12d}{f'{mean:.1f} ± {std:.1f}':>14s}"
                  f"{ms:>9.0f}ms{cons:>10.2f}")


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
