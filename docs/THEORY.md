# What can be proven, and what has to be shown empirically

This note collects the theoretical status of decentralized SM-MCTS motion
planning as implemented in this repository — what is known, what is
plausible but unproven, and what a rigorous evaluation should therefore
claim. It is deliberately honest about the gaps; they are the interesting
research questions.

## Setting

N agents move simultaneously on a shared map (general-sum Markov game with
full state observability and common knowledge of payoffs). Each agent runs
an independent SM-MCTS search with decoupled UCT (DUCT) from the observed
joint state, executes only its own action component, and replans every step
(receding horizon). Collisions are discouraged by payoff ("punishing", as
in the original implementation), not forbidden by constraint.

## 1. Convergence of the search itself

- For *sequential* perfect-information MDPs, UCT converges to the optimal
  action as the simulation budget grows (Kocsis & Szepesvári, 2006).
- For *simultaneous-move* games the picture is subtler. Decoupled UCT —
  what this repo (and the original) uses by default — does **not** converge
  to a Nash equilibrium in general; there are small matrix-game
  counterexamples (Shafiei, Sturtevant & Schaeffer, 2009). DUCT can settle
  on pure joint actions that are not equilibria.
- SM-MCTS with *Hannan-consistent* selection policies (Exp3, regret
  matching) provably converges to an (approximate) Nash equilibrium in
  **two-player zero-sum** simultaneous-move games (Lisý, Kovařík, Lanctot &
  Bošanský, NeurIPS 2013 — the theory companion to Lanctot et al., 2013).
  This is the strongest available guarantee, and it is the reason the
  original codebase carries Exp3 and regret-matching variants.
- Our game is **general-sum** (agents are not strictly adversarial). There,
  independent Hannan-consistent learners converge to the set of *coarse
  correlated equilibria* — a weaker solution concept that does not pin down
  a unique prediction of play.

**Consequence:** even with infinite compute, "the search converges" does
not by itself imply "no collisions". Asymptotic search optimality is a
statement about each agent's best response, not about which of several
equilibria the *population* of agents coordinates on.

## 2. Why a global no-collision theorem cannot come from asymptotics alone

The head-on corridor scenario has (at least) two strict, symmetric
equilibria: (up, down) and (down, up). Both are individually optimal;
selecting between them is a pure *coordination problem*. Two independent,
symmetric planners can each play their part of a **different** equilibrium
— both dodge to the same side — and this is consistent with every
convergence guarantee above. Our experiments reproduce exactly this event
(`docs/media/head_on_conflict.gif`): with 512 simulations per step it
occurred in 1 of 60 decentralized episodes, and was resolved within three
steps by replanning.

There is also a fundamental obstruction in the deterministic limit: in a
perfectly symmetric configuration, *deterministic* symmetric strategies
preserve the symmetry forever (both agents keep mirroring each other) — a
livelock. Any deadlock-freedom result must therefore rely on
**randomization** as the symmetry-breaking mechanism. Stochastic rollouts
and stochastic tie-breaking give SM-MCTS this property for free.

## 3. What *can* plausibly be proven (open, and a nice thesis theorem)

A realistic target is a two-part statement:

**(a) Safety by construction, not by asymptotics.** Replace the collision
*penalty* with a robust action *mask*: agent i may only choose actions that
are collision-free against **every** legal action of the others in the next
step (a maximin/one-step control-barrier filter, in the spirit of velocity
obstacles / ORCA, van den Berg et al. 2011). Then "no collision ever" holds
deterministically, by induction over timesteps — independent of search
quality. The cost is conservatism, and the burden shifts entirely to part
(b). This filter is a straightforward extension of `legal_action_mask` and
is left as future work.

**(b) Liveness via geometric conflict resolution.** Model a conflict step
as the event that the agents' independently sampled action components are
incompatible. If, at every replanning step, the joint (product) strategy
puts probability at least δ > 0 on some compatible, progress-making joint
action — which nondegenerate mixing over finitely many actions provides —
then the probability that a conflict persists for k consecutive steps is at
most (1 − δ)^k. Conflicts then end almost surely, with expected resolution
time ≤ 1/δ, and the receding-horizon loop makes progress between conflicts.
Turning this sketch into a theorem requires making δ explicit for DUCT (or
switching the root to Exp3/regret matching, whose mixed strategies make the
lower bound natural) and handling the coupling between conflicts and the
goal potential. To our knowledge no such result exists for SM-MCTS motion
planning — it is a genuinely publishable contribution, not textbook
material.

## 4. What the experiments can honestly claim

Because the global guarantee is unavailable for the penalty-based planner,
the empirical claim must carry the weight, and it should be phrased as an
*estimate with confidence bounds*, not as a proof:

- With n = 20 seeds per condition and 20/20 successes, the 95% Wilson lower
  confidence bound on the per-episode success probability is 0.84. All-
  success at n = 100 would raise it to ≈ 0.96, n = 300 to ≈ 0.99: the
  claim strengthens only like O(1/n), so budget seeds accordingly.
- Comparing centralized vs. decentralized planning is an **equivalence**
  question: use two-sided Fisher exact tests (done in
  `examples/experiment_decentralized.py`) plus the CI overlap, and report
  the decentralized coordination cost (extra steps, prediction
  consistency) as effect sizes.
- The per-step **prediction consistency** (how often agent i's search
  anticipated agent j's executed action; 0.67–0.73 in our runs against a
  1/6 uniform baseline) is the most direct measurable evidence that the
  agents are solving the *same* game rather than merely avoiding each other
  reactively.

## References

- L. Kocsis, C. Szepesvári. *Bandit Based Monte-Carlo Planning.* ECML 2006.
- M. Shafiei, N. Sturtevant, J. Schaeffer. *Comparing UCT versus CFR in
  Simultaneous Games.* IJCAI GIGA Workshop 2009.
- M. Lanctot, V. Lisý, M. H. M. Winands. *Monte Carlo Tree Search in
  Simultaneous Move Games with Applications to Goofspiel.* IJCAI CGW 2013.
- V. Lisý, V. Kovařík, M. Lanctot, B. Bošanský. *Convergence of Monte Carlo
  Tree Search in Simultaneous Move Games.* NeurIPS 2013.
- J. van den Berg, S. J. Guy, M. Lin, D. Manocha. *Reciprocal n-Body
  Collision Avoidance* (ORCA). ISRR 2011.
