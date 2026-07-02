# Asynchronous replanning: model, hypothesis, and what the data actually said

This branch tests the hypothesis that the *timing* of decisions — neither
strictly sequential nor strictly simultaneous, but staggered like real
robot control loops — breaks the coordination symmetry that causes
both-dodge-the-same-way conflicts in decentralized planning.

**Spoiler, because honesty beats a good story:** the hypothesis survived
only in refined form. Asynchrony alone did not eliminate conflicts; the
dominant factors turned out to be *commitment duration* and — the key
insight — *commitment observability*. This is a finding, not a failure:
it tells you exactly what the next theorem needs.

## The model

`AsyncDecentralizedPlanner` gives every agent its own replanning clock:

- Agent i replans at t = 0 and then every `periods[i]` steps, shifted by
  `phases[i]`, optionally with a uniform random `jitter` added per
  interval (a discrete stand-in for independent Poisson clocks).
- Between replans it executes its **committed plan**: the robust action
  sequence along the principal variation of its last search
  (`SearchResult.action_plan`, length `MCTSParams.commit_depth`).
- At a replan it observes the current world state (positions, headings,
  goal flags) — **not** the others' committed plans.

## The experiment

Symmetric head-on corridor, decentralized, 256 simulations per search
(deliberately noisy equilibrium selection), 100 seeds per condition
(`examples/experiment_async.py`):

| condition | clocks | coll. episodes | coll. steps | simultaneous revisions/episode |
|---|---|---|---|---|
| sync_fast | both every step | **0/100** | 0 | 8.9 |
| sync_slow | both every 2 steps, in phase | 6/100 | 9 | 4.2 |
| async | every 2 steps, phases 0/1 | 4/100 | 6 | **0.0** |
| jitter | every 2 steps + U{0,1} | 12/100 | 22 | 1.6 |

Fisher exact (collision episodes): sync_fast vs sync_slow p = 0.029,
sync_fast vs jitter p = 0.0003, sync_slow vs async p = 0.75 (n.s.).
All 400 episodes reached their goals.

## Interpretation — three findings

**1. Commitment duration dominates.** Replanning every step (sync_fast)
had zero conflicts; committing for two steps raised the rate
significantly, in *every* timing variant. Reacting late is worse than
revising simultaneously. For the receding-horizon planner this is direct
evidence that high replanning frequency is itself a safety mechanism.

**2. Asynchrony helps less than the theory of asynchronous best-response
dynamics predicts — and the reason is identifiable.** Deterministic
interleaving achieved exactly what it promises mechanically (zero
simultaneous revisions) and trended better than in-phase commitment
(4 vs. 6 episodes, not significant at these base rates). But it could not
reach zero, because the theory's key premise is not satisfied: in
asynchronous best-response dynamics the reviser observes the opponent's
*current strategy* and responds to it. Our reviser observes only
*positions* — it re-solves the simultaneous-move game from scratch and
can still mispredict what the other agent has already committed to do.
Asynchrony without commitment observability removes the revision
coincidence but not the prediction error.

**3. Random clocks are the worst of both worlds (in discrete time).**
The continuous-time intuition "independent random clocks never tick
simultaneously" does not transfer: with discrete steps, jittered clocks
coincided 1.6 times per episode, *and* the variable commitment horizon
made partners harder to predict. Jitter more than doubled the conflict
rate of the in-phase baseline. If you want asynchrony in a discrete
implementation, engineer deterministic interleaving; don't rely on
randomness.

## The refined thesis (and the next theorem)

> Asynchrony breaks coordination symmetry **if and only if** the
> revising agent can condition on the others' current commitments.

That is the missing ingredient, and it is exactly what asynchronous
best-response convergence (weakly acyclic / potential games) assumes.
Two concrete follow-ups, in increasing order of realism:

1. **Commitment broadcast:** let the reviser see the others' committed
   action sequences (robots announcing intent — standard in cooperative
   driving). Inside the search, committed agents are simulated as playing
   their announced plan instead of being co-searched. Prediction: the
   conflict rate of `async` drops to (near) zero, and the potential-game
   convergence argument applies cleanly.
2. **Commitment inference:** estimate the others' committed motion from
   observed heading/velocity (they are part of the state). Cheaper
   assumption, noisier signal — quantifies how much observability the
   guarantee actually needs.

Safety note: committed execution is currently incompatible with the
safety filter's per-step guarantee (a committed action is not re-checked
against the world at execution time). Combining the two needs
reservation-style filtering over the whole commitment window — the
continuous-time generalization sketched in docs/THEORY.md §3a.
