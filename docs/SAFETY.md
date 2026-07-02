# The safety filter: assumptions, rule, and proof

`MCTSParams(safety_filter=True)` replaces the plain obstacle mask with a
filter that makes agent–agent collisions **impossible by construction** —
independent of how good the search is, how many simulations it runs, or
whether the agents plan centrally or independently. This note states the
assumptions precisely and gives the full proof. It is short on purpose:
the point of a safety argument is that it fits on one page.

## Intuition first

Two ideas, both computable by every agent alone from what it can see:

1. **Treat everyone's current spot as reserved.** My action is only
   allowed if my path this step stays clear of every other agent's
   *current* position. Consequence: whatever the others do, they can rely
   on their own spot being safe — so *standing still is always allowed*.
   The filter can never leave an agent without options.

2. **Break the tie by public priority.** "Safe against the others" is
   circular (safe against *which* of their actions?). We break the circle
   with the agent index: agent 0 filters only by rule 1; agent 1 must
   additionally be safe against everything that survived agent 0's filter;
   agent 2 against agents 0 and 1; and so on. Everyone can compute
   everyone's filter — the ordering is public, no communication needed.

Rule 1 guarantees you always have a safe action. Rule 2 guarantees that
any way the agents independently pick from their filtered sets, no pair
can collide.

## Assumptions

- **A1 (initial separation).** At the start, every two agents are more
  than the collision radius R apart. (Checked at planner construction.)
- **A2 (stop action).** Every agent's action set contains an action that
  keeps it at its current position (v = 0). *(True for the default action
  sets; `build_world` requires it implicitly via `null_action`.)*
- **A3 (common knowledge).** All agents observe the same joint state
  (positions, headings, goal-reached flags, time) and know each other's
  action sets and the priority order (= agent index). The filter is a
  deterministic function of these, so all agents compute identical filters.
- **A4 (motion model).** Within one timestep agents move along straight
  interpolated segments, and two agents collide iff their interpolated
  positions come within R of each other at the same interpolation instant
  — the same convention used by the simulator's collision check.
- **A5 (environment).** The (possibly time-varying) map never turns the
  cell under a stationary agent into an obstacle. (Otherwise "stand
  still" may be obstacle-illegal; the filter concerns agent–agent
  collisions and inherits obstacle handling from the legality mask.)

## The rule

Let `swept(i, a)` be agent i's interpolated path when it plays action a
this timestep. Compute, in increasing order of agent index i:

```
SAFE_i = { a legal for i :
             (1) swept(i, a) keeps distance > R from the current
                 position of every agent j != i, and
             (2) for every j < i and every b in SAFE_j:
                 swept(i, a) and swept(j, b) keep distance > R
                 at every common interpolation instant }
```

Agents that already reached their goal are frozen: their set is `{stay}`.
Each agent then lets its MCTS search only over `SAFE_i` (in the tree, in
rollouts, and at the root).

## Lemma 1 (no agent is ever stuck): `stay ∈ SAFE_i`, always.

*Proof.* Condition (1) for `stay`: the swept path of `stay` is the single
point `pos_i`; by the induction invariant (pairwise distances > R, base
case A1) it is more than R from every `pos_j`. Condition (2): any
`b ∈ SAFE_j` with j < i itself satisfies condition (1), so `swept(j, b)`
stays more than R away from `pos_i` — which is exactly `swept(i, stay)`.
Hence `stay` meets both conditions. ∎

## Theorem (no collision, ever)

If A1–A5 hold and every agent always plays some action from its `SAFE_i`
(chosen arbitrarily and independently — different searches, different
random seeds, no communication), then at no time do two agents come within
distance R of each other, at any instant of any transition.

*Proof by induction over timesteps.* **Invariant:** at the start of the
step, all pairwise distances exceed R (A1 gives the base case).

Induction step: take any two agents i > j with chosen actions
`a_i ∈ SAFE_i`, `a_j ∈ SAFE_j`. Since j < i, condition (2) of `SAFE_i`
was evaluated against *every* element of `SAFE_j` — in particular against
`a_j`. Therefore `swept(i, a_i)` and `swept(j, a_j)` stay more than R
apart at every instant of the transition (A4). This covers the whole
continuous transition *and* its endpoint, so the invariant holds again at
the next step. Lemma 1 guarantees the choice was possible at all. By A3
every agent computed the same `SAFE` sets, so "i filtered against j's set"
refers to the set j actually used. ∎

Two things the theorem does **not** say — stated here so the thesis can be
precise about them:

- **No liveness guarantee.** The filter guarantees you never crash, not
  that you arrive. Deadlocks/livelocks are not excluded by this argument;
  they are prevented in practice by the search optimizing progress within
  the safe sets and by stochastic symmetry breaking (see docs/THEORY.md,
  §3b, for the geometric-resolution argument — with the filter enabled,
  that argument only has to carry liveness, no longer safety).
- **Conservatism.** Condition (1) forbids moving into a cell that another
  agent is *currently* occupying, even if that agent is about to leave it.
  Convoys therefore keep a one-cell headway, and tight swaps can cost an
  extra step. Empirically this cost did not materialize in the three
  benchmark scenarios — filtered episodes were slightly *shorter* (head-on
  9.4 vs. 10.1 steps, bottleneck 14.3 vs. 15.8, gates 23.0 vs. 24.5) with
  higher prediction consistency, because removing the conflicting branches
  also removes the coordination ambiguity the searches otherwise have to
  resolve. Expect the headway cost to appear in narrower maps (single-lane
  convoys). Planning time roughly doubles.

## Where this sits in the literature

The construction is a discrete, simultaneous-move cousin of reciprocal
velocity obstacles / ORCA (van den Berg et al., 2011) and of control
barrier function safety filters: a myopic, worst-case-feasible action
constraint wrapped around an arbitrary planner. The priority trick for
breaking the mutual-dependence circle is standard in prioritized
multi-robot planning; the observation that condition (1) makes `stay`
universally safe (Lemma 1) is what keeps the filtered game well-defined
without any communication.

## Verifying the proof mechanically

`tests/test_safety.py::test_every_safe_combination_is_collision_free`
enumerates the *entire product* of filtered action sets in an adversarial
face-to-face configuration and asserts the simulator's collision check
never fires — the theorem, checked exhaustively for that instance.
`test_stay_action_always_safe` randomizes configurations and checks
Lemma 1. `examples/experiment_safety.py` measures the conservatism cost
end-to-end.
