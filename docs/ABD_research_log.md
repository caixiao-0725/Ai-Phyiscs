# ABD Research Log

This document tracks the research direction of replacing AVBD's axis-angle rotational representation with an affine-matrix local representation.

## Research Goal

Use AVBD as the trusted SOTA baseline and investigate whether ABD can preserve the Augmented Lagrangian structure while improving the local solve through an affine matrix representation.

The intended novelty is not "use a general affine transform as a deformable body". The intended novelty is:

- represent the local rigid update with affine variables,
- derive the exact directional Jacobian/Hessian,
- solve the local contact problem with the anisotropic Schur system,
- constrain the affine result back toward `SO(3)` in a way that is consistent with the local objective.

## Mathematical Core

For a body point `r`, ABD uses:

```text
y = c_tilde + A_tilde * (c_L + A_L * r)
```

For a scalar directional constraint:

```text
C_d = n_tilde^T * (c_L + A_L * r) + xi_d
J_d = [n_tilde^T, n_tilde^T r0, n_tilde^T r1, n_tilde^T r2]
H_d = kappa_d * J_d^T * J_d
```

The important difference from isotropic PABD-style bonds is that each directional constraint contributes:

```text
N = n_tilde outer n_tilde
```

not `I_3`. This means the affine rows are coupled by the contact direction. This is the reason ABD uses `AffineSchurSolver::accumulate_dir()`.

## Current Hypotheses

### H1: Direct post-solve polar projection is not a safe rigidification step

If ABD solves an affine AL system and then directly applies `polar_rotation(ATilde * AL)`, the projection can move contact points after the solver has satisfied them. The dual update then sees a state that was not actually optimized.

Expected failure signature:

- contact correction injects kinetic energy during velocity recovery,
- blocks shoot upward or sideways after stack collapse,
- failure gets worse with many simultaneous contacts.

Current judgment:

- Direct projection should not be treated as an innocent post-process.
- If projection is used, it should be part of a local/global objective or accepted only through an energy check.

### H2: SO(3) should enter ABD as a local objective term

ABD should solve a local problem with a proximal term:

```text
muSO3 * ||A_L - R_L||^2
R_L = A_tilde^T * polar(A_tilde * A_L)
```

This turns rigidification into part of the solve rather than an external state mutation.

Expected benefit:

- affine stretch/shear stays bounded,
- local contact solve remains closer to a rigid update,
- reduced energy injection compared with hard projection.

Current implementation status:

- `stepCpuAffine()` now solves once in affine space.
- It then computes a local `SO(3)` target and re-solves with `muSO3`.
- A local energy backtracking step accepts only descent steps.

### H3: ABD needs a local energy/trust mechanism, similar in spirit to PABD line search

Even with correct anisotropic Hessians, the local affine AL step can be too aggressive in dense contact. Without an energy check, a single Gauss-Seidel primal update can convert penetration correction into kinetic energy.

Current implementation status:

- Local backtracking evaluates an energy containing:
  - inertial displacement,
  - affine inertia,
  - active AL contact penalties and lambdas,
  - SO(3) proximal penalty.
- The step from current local state to candidate state is halved until the local energy decreases.

### H4: Iteration count is part of the algorithmic story for tall stacks

For tall stacks, support propagation through Gauss-Seidel style local solves may require substantially more iterations than small scenes.

Observed behavior so far:

- With too few iterations, tall stacks can still collapse or deep-penetrate even if local steps are stabilized.
- Increasing iterations can distinguish solver instability from insufficient propagation.

## Experiment Matrix

Use these scenes first:

```text
ABD: Tilted Drop
ABD: Stacking
PABD: Stacking
AVBD stack scenes
```

Recommended measured signals:

- max linear velocity,
- max upward velocity,
- max angular velocity,
- max penetration depth,
- number of ground and box contacts,
- max penalty and lambda,
- local energy before and after line search,
- number of accepted line-search halvings.

## Current Failure Examples

### Direct projection / no energy check

Failure signature:

- stack collapse creates large rebound velocity,
- some bodies jump upward after ground contact,
- velocities can grow far beyond gravitational free-fall scale.

Diagnosis:

- likely not a pure code bug,
- caused by mismatch between affine solve, hard SO(3) projection, AL dual update, and velocity recovery.

### SO(3) proximal without adequate line search

Failure signature:

- upward explosion is reduced,
- but sideways ejection can still occur during dense contact,
- trust-region too small causes support failure and deep penetration.

Diagnosis:

- SO(3) proximal is necessary but not sufficient.
- The step acceptance criterion matters.

## Current Positive Evidence

SO(3) proximal plus local energy backtracking improves robustness relative to the earlier direct-projection ABD variant.

Current best evidence to maintain:

- tilted drop remains stable and settles close to flat,
- stack tests no longer require post-solve velocity clamping to avoid immediate explosion,
- tall stack stability depends strongly on iteration count, suggesting support propagation is still a separate bottleneck.

## 2026-06-08 Current Baseline

Current workspace scene:

```text
ABD: Stacking
n = 50
iterations = 20
ground friction = 0
body friction = 0
```

Smoke-test results:

```text
ABD: Stacking, 500 frames
Result: completed, no high-velocity diagnostic output

ABD: Tilted Drop, 600 frames
Result: completed
Final state: pos ~= (1.003, 0, 1.003)
Final velocity magnitude: ~0.13
Final angular velocity magnitude: ~0.09
```

Interpretation:

- The current ABD variant is no longer immediately exploding on the tested stack.
- Tilted drop is stable but not perfectly at rest by 600 frames.
- This is not yet enough to claim superiority over AVBD; it is a candidate stable ABD variant that needs controlled comparisons.

## 2026-06-08 Fair 50-Box Stacking Comparison

Scene setup:

```text
50 boxes
box size = (1, 1, 1)
density = 1
body friction = 0
ground z = 0
ground friction = 0
iterations = 20
frames = 500
```

Results:

```text
ABD: Stacking
maxSpeed      = 5.1667
maxUp         = 4.1100
minZ          = -0.0245
finalMaxSpeed = 1.4223
finalMinZ     = -0.0029
nan           = 0

PABD: Simple Stacking
maxSpeed      = 4.3333
maxUp         = 3.1239
minZ          = -0.0482
finalMaxSpeed = 0.2122
finalMinZ     = -0.0206
nan           = 0

AVBD: Stacking
maxSpeed      = 4.8317
maxUp         = 3.2094
minZ          = -0.0334
finalMaxSpeed = 0.0243
finalMinZ     = -0.0183
nan           = 0
```

Interpretation:

- `AVBD` is clearly the best baseline in this fair stack test: final motion is almost fully damped by frame 500.
- `PABD` is stable and damps better than the current ABD variant, but penetrates slightly more in this setup.
- Current `ABD` is stable in the sense of no explosion/NaN, but it retains significantly more residual motion (`finalMaxSpeed = 1.4223`). This means the current ABD implementation is not yet competitive with AVBD for frictionless vertical stacking.

Research diagnosis:

- The affine SO(3) proximal and local energy backtracking are useful because they stop catastrophic energy injection.
- They do not yet reproduce AVBD's damping/convergence behavior.
- The likely remaining gap is not the directional Hessian derivation itself, but the full discrete algorithm around it: velocity recovery, dual update damping, contact persistence, and how affine variables are retracted to a rigid state.

Next focused question:

```text
Why does AVBD dissipate stack motion much faster than ABD under the same geometry and iteration count?
```

Candidate explanations:

- AVBD's axis-angle update is intrinsically a retraction step, while ABD still carries affine residual modes even with SO(3) proximal.
- ABD's incremental velocity recovery preserves correction velocity that AVBD damps through its original AL update structure.
- ABD's local line search only enforces per-body local energy descent, not global multi-body energy descent.
- The current SO(3) proximal weight may be preventing over-deformation but not enforcing enough rotational damping.

## 2026-06-08 Residual Form and Penalty Sweep

### Hypothesis H5: ABD residual should match AVBD's relaxed AL form

AVBD uses a relaxed residual:

```text
C = C0 * (1 - alpha) + J * dq
```

Earlier ABD code used a more direct gap-to-target residual for ground and box contacts. This made the ABD AL update structurally different from AVBD even before considering affine variables.

Change tested:

- Ground primal residual uses `gc.C0[d] * (1 - alpha) + dir^T(x - x_initial)`.
- Box primal residual uses `bc.C0[d] * (1 - alpha) + basis_d^T((xA - xB) - (xA0 - xB0))`.
- Dual update and local line-search energy use the same residual definition.

Result on fair 50-box stacking, 500 frames, `betaLin=1000`:

```text
ABD absolute velocity:
maxSpeed      = 3.2549
maxUp         = 2.7610
minZ          = -0.0243
finalMaxSpeed = 0.6578

ABD incremental velocity:
maxSpeed      = 3.7391
maxUp         = 2.3849
minZ          = -0.0236
finalMaxSpeed = 1.6212
```

Previous ABD final residual speed was much higher (`~1.42` for incremental before the residual fix, and `~2.21` for absolute after switching velocity mode but before this experiment). Matching AVBD's residual form clearly improves damping.

Diagnosis:

- This was an implementation mismatch with AVBD, not a failure of the anisotropic Hessian derivation.
- Absolute BDF1-style velocity recovery is better than incremental recovery for the current ABD formulation, consistent with AVBD.

### Hypothesis H6: ABD needs different penalty growth than AVBD

After adding local line search, we tested whether restoring AVBD's original penalty scale helps ABD:

```text
AVBD-style high beta:
betaLin = 10000
AVBD_PENALTY_MAX = 1e10
```

Result: this improves AVBD but makes ABD much worse.

Fair 50-box stacking, 500 frames:

```text
ABD betaLin=100:
maxSpeed      = 2.7694
maxUp         = 2.1638
minZ          = -0.0380
finalMaxSpeed = 0.1195

ABD betaLin=1000:
maxSpeed      = 3.2549
maxUp         = 2.7610
minZ          = -0.0243
finalMaxSpeed = 0.6578

ABD betaLin=10000:
maxSpeed      = 33.8817
maxUp         = 15.4269
minZ          = -0.2105
finalMaxSpeed = 19.8805

PABD:
maxSpeed      = 4.3333
maxUp         = 3.1239
minZ          = -0.0482
finalMaxSpeed = 0.2122

AVBD:
maxSpeed      = 2.8326
maxUp         = 0.9245
minZ          = -0.0147
finalMaxSpeed = 0.0002
```

Interpretation:

- ABD with `betaLin=100` is now better than PABD on final residual speed in this simple frictionless stack.
- ABD is still worse than AVBD by a large margin in final damping.
- High penalty growth is destabilizing for ABD even with local line search, so ABD cannot simply inherit AVBD's penalty schedule.

Research diagnosis:

- The affine local solve is more sensitive to penalty stiffness than AVBD's axis-angle solve.
- The SO(3) proximal/line-search mechanism prevents catastrophic explosion, but high penalties still create aggressive contact corrections and angular/affine coupling.
- A research-worthy ABD version likely needs its own adaptive penalty schedule, not AVBD's schedule.

Next focused experiment:

```text
Design an ABD-specific penalty update:
- slower normal penalty growth,
- separate cap for affine/contact stiffness,
- maybe line-search-aware penalty increase only when local energy accepted with large enough step.
```

Current working setting:

```text
ABD: Stacking uses betaLin = 100
ABD: Stacking Beta1000 and ABD: Stacking Beta10000 remain available as ablation scenes.
```

With the current best ABD settings on fair 50-box stacking, 500 frames:

```text
ABD betaLin=100:
finalMaxSpeed = 0.1195
minZ          = -0.0380

PABD:
finalMaxSpeed = 0.2122
minZ          = -0.0482

AVBD:
finalMaxSpeed = 0.0002
minZ          = -0.0147
```

Updated interpretation:

- ABD now beats PABD on final residual speed and maximum penetration in this simple stack.
- ABD still does not match AVBD's very strong damping.
- This is the first useful "partial win" for ABD: affine anisotropic AL with SO(3) proximal and AVBD-style residual can be competitive against PABD in a controlled frictionless stack.
- The remaining gap to SOTA is likely the penalty schedule and damping/convergence behavior, not just the affine Hessian.

## Next Research Steps

1. Add structured diagnostics instead of ad hoc prints.
   - Track max velocity, upward velocity, penetration, contacts, penalties, line-search step.

2. Compare variants under identical scenes.
   - AVBD baseline.
   - ABD direct projection.
   - ABD SO(3) proximal only.
   - ABD SO(3) proximal + local energy backtracking.

3. Decide whether affine is the right primary representation.
   - Avoid arbitrary post-solve projection.
   - Compare "affine with SO(3) proximal" against "rotation retraction variables".

4. Search for winning scenarios.
   - Dense directional contacts.
   - Anisotropic friction/contact.
   - Tall stacks with many aligned contacts.
   - Cases where axis-angle linearization has large-angle artifacts.

## Working Interpretation

ABD is still research-promising, but not because "affine automatically makes the solver more stable." The affine representation gives a clean anisotropic local Hessian. Stability requires a consistent way to keep the affine variable on or near `SO(3)` and a descent mechanism for contact corrections.

If ABD only works through velocity clamps, it is not yet a real algorithmic improvement. If ABD works through local SO(3) objectives and energy-checked steps, then the research contribution can be framed as an affine local representation with anisotropic directional contact Hessians and objective-consistent rigidification.

## 2026-06-08 High-Rotation Rods: First ABD-Favorable Scene

### Motivation

The ABD paper claims improved convergence mainly for cases where strict rigid coordinates create difficult nonlinear trajectories and contact processing. A simple axis-aligned stack is not the best place to show ABD's advantage. A better target is high angular velocity with repeated contact, where axis-angle rigid updates can linearize rotation poorly.

### Scene

```text
Scene: ABD/AVBD/PABD High Rotation Rods
Bodies: 6 long rods
Size: (4.0, 0.35, 0.35)
Ground friction: 0.5
Body friction: 0.5
Iterations: 5
Initial angular velocity: large, mixed signs
Frames: 500 and 1000
```

### 500-frame results

```text
ABD: High Rotation Rods
maxSpeed      = 32.8766
maxUp         = 20.6997
maxAng        = 27.0441
minZ          = -0.0130
finalMaxSpeed = 0.2295

PABD: High Rotation Rods
maxSpeed      = 39.9631
maxUp         = 39.2557
maxAng        = 101.8121
minZ          = -3.1707
finalMaxSpeed = 37.1121

AVBD: High Rotation Rods
maxSpeed      = 17.5619
maxUp         = 11.0235
maxAng        = 98.2542
minZ          = -0.4750
finalMaxSpeed = 0.6093
```

### 1000-frame results

```text
ABD: High Rotation Rods
maxSpeed      = 32.8766
maxUp         = 20.6997
maxAng        = 27.0441
minZ          = -0.0130
finalMaxSpeed = 0.1078

PABD: High Rotation Rods
maxSpeed      = 61.0603
maxUp         = 61.0248
maxAng        = 101.8121
minZ          = -3.1707
finalMaxSpeed = 19.4747

AVBD: High Rotation Rods
maxSpeed      = 17.5619
maxUp         = 11.0235
maxAng        = 98.2542
minZ          = -0.4750
finalMaxSpeed = 0.1902
```

### Interpretation

This is the first controlled scene where current ABD shows a meaningful advantage over AVBD:

- ABD has much smaller maximum penetration (`-0.0130` vs `-0.4750`).
- ABD has lower final residual speed after 500 and 1000 frames.
- PABD performs poorly here, with large penetration and persistent high angular velocity.

Important caveat:

- ABD has larger early translational speed and upward speed than AVBD. This likely comes from aggressive early contact correction.
- The win is not "ABD is globally better than AVBD"; the win is specific to high-rotation rod contact under low iteration count.

Research diagnosis:

- This result matches the ABD paper's core intuition: affine trajectories can handle large rotation contact more robustly than strict rigid coordinate linearization.
- The current ChysX ABD variant may be worth positioning around high-rotation, high-contact scenarios, not simple resting stacks.

Next focused experiment:

```text
Reduce ABD's early correction spike while preserving its low penetration:
- try line-search cap based on upward velocity or kinetic energy,
- test 5, 10, 20 iterations,
- test more rods and narrower rods,
- inspect whether early maxUp is from ground contact or rod-rod contact.
```

## 2026-06-08 Rotating Gate: More Controlled ABD-Favorable Scene

### Motivation

The earlier high-rotation rods scene is useful but too random to be a convincing benchmark. A stronger test should have a clear task and a clear failure mode.

`Rotating Gate` is designed as a controlled large-rotation contact problem:

- one long thin rod,
- high angular velocity,
- two static narrow gate posts plus a small static cross obstacle,
- ground contact,
- only 5 solver iterations.

The task is: resolve a rotating rod interacting with a narrow obstacle region without large penetrations or unstable residual motion.

### Scene

```text
Dynamic rod:
size = (3.2, 0.22, 0.22)
initial velocity = (0, 0, -1.5)
initial angular velocity = (7, 28, 18)

Static obstacles:
two vertical gate posts
one cross bar

iterations = 5
friction = 0.5
frames = 500 and 1000
```

Metrics include:

```text
maxSpeed
maxUp
maxAng
minZ
maxPairPen
finalMaxSpeed
finalPairPen
```

### 500-frame results

```text
ABD: Rotating Gate
maxSpeed      = 13.3096
maxUp         = 4.3975
maxAng        = 29.5904
minZ          = -0.0082
maxPairPen    = 0.1282
finalMaxSpeed = 0.0197
finalPairPen  = 0.0000

PABD: Rotating Gate
maxSpeed      = 28.5930
maxUp         = 28.1177
maxAng        = 68.1030
minZ          = -2.2567
maxPairPen    = 0.6768
finalMaxSpeed = 16.8788
finalPairPen  = 0.0000

AVBD: Rotating Gate
maxSpeed      = 6.7564
maxUp         = 2.3662
maxAng        = 34.8769
minZ          = -0.1015
maxPairPen    = 1.3024
finalMaxSpeed = 0.0658
finalPairPen  = 0.0000
```

### 1000-frame results

```text
ABD: Rotating Gate
maxSpeed      = 13.3096
maxUp         = 4.3975
maxAng        = 29.5904
minZ          = -0.0082
maxPairPen    = 0.1282
finalMaxSpeed = 0.0000
finalPairPen  = 0.0000

PABD: Rotating Gate
maxSpeed      = 28.9188
maxUp         = 28.5378
maxAng        = 101.5781
minZ          = -2.2680
maxPairPen    = 0.6768
finalMaxSpeed = 9.2873
finalPairPen  = 0.0000

AVBD: Rotating Gate
maxSpeed      = 6.7564
maxUp         = 2.3662
maxAng        = 34.8769
minZ          = -0.1015
maxPairPen    = 1.3024
finalMaxSpeed = 0.0000
finalPairPen  = 0.0000
```

### Interpretation

This is a stronger ABD-favorable result than the random rods scene:

- ABD has much lower ground penetration than AVBD (`-0.0082` vs `-0.1015`).
- ABD has much lower body-body penetration than AVBD (`0.1282` vs `1.3024`).
- ABD and AVBD both settle by 1000 frames.
- PABD fails this test badly: large penetration, high angular velocity, high final motion.

Important caveat:

- AVBD has lower peak translational speed and upward speed than ABD.
- ABD's advantage is contact accuracy/penetration under high rotation, not lower transient kinetic energy.

Research diagnosis:

- This scene better matches the ABD paper's motivation: affine trajectories help when a fast rotating slender object must resolve contact with narrow obstacles.
- The current ABD implementation now has a plausible advantage story: it can reduce penetration in high-rotation contact at low iteration counts.

Next experiment:

```text
Sweep iterations = 2, 5, 10, 20 for Rotating Gate.
If ABD keeps lower maxPairPen at low iterations while AVBD needs more iterations, this becomes a strong convergence argument.
```

## 2026-06-08 Rotating Gate Iteration Sweep

### Purpose

Test whether ABD's advantage in `Rotating Gate` is actually a convergence advantage under low iteration counts, rather than a one-off parameter artifact.

All runs use the same geometry and initial conditions:

```text
Scene family: Rotating Gate
Frames: 500
Iterations: 2, 5, 10, 20
Compared solvers: ABD vs AVBD
Primary metrics: maxPairPen, minZ, finalMaxSpeed
```

### Results

```text
Iterations = 2
ABD:
  maxPairPen    = 0.0632
  minZ          = -0.1724
  finalMaxSpeed = 0.0476
AVBD:
  maxPairPen    = 1.1958
  minZ          = -0.1201
  finalMaxSpeed = 0.2861

Iterations = 5
ABD:
  maxPairPen    = 0.1282
  minZ          = -0.0082
  finalMaxSpeed = 0.0197
AVBD:
  maxPairPen    = 1.3024
  minZ          = -0.1015
  finalMaxSpeed = 0.0658

Iterations = 10
ABD:
  maxPairPen    = 0.0632
  minZ          = -0.1808
  finalMaxSpeed = 0.1742
AVBD:
  maxPairPen    = 1.3188
  minZ          = -0.2573
  finalMaxSpeed = 0.0084

Iterations = 20
ABD:
  maxPairPen    = 0.3596
  minZ          = -0.0286
  finalMaxSpeed = 0.2297
AVBD:
  maxPairPen    = 0.8434
  minZ          = -0.0783
  finalMaxSpeed = 0.0084
```

### Interpretation

This sweep gives a more precise ABD advantage story:

- ABD has lower body-body penetration (`maxPairPen`) at every tested iteration count.
- ABD is especially convincing at 2 and 5 iterations: lower pair penetration and lower final residual speed than AVBD.
- At 10 and 20 iterations, AVBD's final damping becomes much stronger, although AVBD still has larger peak pair penetration.
- ABD's penetration advantage is robust, but its high-iteration residual motion is not yet competitive.

### Research diagnosis

This supports the paper-aligned claim that affine trajectories help contact convergence in large-rotation narrow-obstacle contact. The advantage is not general damping; it is contact accuracy under hard rotational contact.

Likely remaining issue:

```text
ABD controls penetration better but dissipates less effectively at higher iteration counts.
```

Next focused experiment:

```text
Keep the Rotating Gate setup.
Study why ABD final residual speed worsens at 10/20 iterations:
- inspect line-search accepted step sizes,
- inspect SO(3) proximal residual,
- test stronger/weaker muSO3,
- test mild velocity damping that is derived from the local energy decrease, not an ad hoc clamp.
```

## 2026-06-08 Rotating Gate SO(3) Proximal Sweep

### Purpose

After the iteration sweep, ABD had lower penetration than AVBD but worse final damping at higher iteration counts. This experiment tests whether the SO(3) proximal weight `muSO3` controls that trade-off.

Scene:

```text
ABD: Rotating Gate Iter20
frames = 500
muSO3 sweep = 5, 10, 50, 200, 500
```

Results:

```text
muSO3 = 5
maxSpeed      = 4.0352
maxUp         = 1.2149
minZ          = -0.0668
maxPairPen    = 0.3596
finalMaxSpeed = 0.0419

muSO3 = 10
maxSpeed      = 5.3493
maxUp         = 0.8816
minZ          = -0.1029
maxPairPen    = 0.3596
finalMaxSpeed = 0.0175

muSO3 = 50
maxSpeed      = 8.5150
maxUp         = 0.4809
minZ          = -0.0286
maxPairPen    = 0.3596
finalMaxSpeed = 0.2297

muSO3 = 200
maxSpeed      = 14.2195
maxUp         = 2.4950
minZ          = -0.1052
maxPairPen    = 0.3596
finalMaxSpeed = 0.3971

muSO3 = 500
maxSpeed      = 9.8708
maxUp         = 0.4669
minZ          = -0.0251
maxPairPen    = 0.3596
finalMaxSpeed = 0.2079

AVBD Iter20 reference
maxSpeed      = 7.7174
maxUp         = 1.3002
minZ          = -0.0783
maxPairPen    = 0.8434
finalMaxSpeed = 0.0084
```

Interpretation:

- `muSO3 = 10` gives the best ABD final damping in this sweep.
- All tested ABD `muSO3` values keep lower `maxPairPen` than AVBD.
- Very strong `muSO3` does not help; it makes the affine update too close to a stiff rigid retraction and increases residual motion.
- Too weak `muSO3` can worsen ground penetration, but still damps better than the previous default.

### Iter5 check

At the original low-iteration setting:

```text
ABD Iter5 muSO3=10
maxSpeed      = 7.8181
maxUp         = 4.8504
minZ          = -0.0047
maxPairPen    = 0.3596
finalMaxSpeed = 0.0000

ABD Iter5 muSO3=50
maxSpeed      = 13.3096
maxUp         = 4.3975
minZ          = -0.0082
maxPairPen    = 0.1282
finalMaxSpeed = 0.0197

AVBD Iter5
maxSpeed      = 6.7564
maxUp         = 2.3662
minZ          = -0.1015
maxPairPen    = 1.3024
finalMaxSpeed = 0.0658
```

Interpretation:

- `muSO3=10` improves final damping and ground penetration.
- `muSO3=50` gives lower pair penetration.
- Both ABD variants keep much lower pair penetration than AVBD.

Research diagnosis:

```text
muSO3 is not just a numerical stabilizer.
It controls the research trade-off between affine freedom and rigidification.
```

Tentative guidance:

- For penetration-critical high-rotation contact: use stronger `muSO3` such as 50.
- For final damping/convergence: use weaker `muSO3` such as 10.
- A better algorithm may adapt `muSO3` by contact severity or line-search step size.

## 2026-06-08 Dynamic SO(3) Proximal Ramp

### Hypothesis

Instead of using a fixed `muSO3`, start with a small SO(3) proximal weight and gradually increase it during the AL iterations:

```text
muSO3(it) = lerp(mu_start, mu_end, it / (iterations - 1))
```

Intuition:

- Early iterations: allow more affine freedom to resolve large-rotation contact.
- Later iterations: pull the body back toward a rigid rotation.
- This is analogous in spirit to increasing contact penalty, but applied to the rigidity/orthogonality target.

### Implementation

Added solver fields:

```text
affine_so3_mu_ramp
affine_so3_mu_start
affine_so3_mu_end
```

The local SO(3) proximal term now uses either fixed `affine_so3_mu` or the ramped value at the current AL iteration.

### Rotating Gate Iter5 Results

```text
Fixed muSO3 = 10
maxSpeed      = 7.8181
maxUp         = 4.8504
minZ          = -0.0047
maxPairPen    = 0.3596
finalMaxSpeed = 0.0000

Fixed muSO3 = 50
maxSpeed      = 13.3096
maxUp         = 4.3975
minZ          = -0.0082
maxPairPen    = 0.1282
finalMaxSpeed = 0.0197

Ramp 1 -> 50
maxSpeed      = 8.2668
maxUp         = 5.2125
minZ          = -0.0031
maxPairPen    = 0.3596
finalMaxSpeed = 0.0766

Ramp 5 -> 50
maxSpeed      = 8.2666
maxUp         = 5.2126
minZ          = -0.0038
maxPairPen    = 0.3596
finalMaxSpeed = 0.1465

Ramp 10 -> 50
maxSpeed      = 13.3076
maxUp         = 4.3984
minZ          = -0.0129
maxPairPen    = 0.1282
finalMaxSpeed = 0.0657

AVBD Iter5
maxSpeed      = 6.7564
maxUp         = 2.3662
minZ          = -0.1015
maxPairPen    = 1.3024
finalMaxSpeed = 0.0658
```

Interpretation:

- Ramp variants still beat AVBD on penetration.
- At Iter5, fixed `muSO3=10` gives the best final damping.
- Fixed `muSO3=50` and ramp `10 -> 50` give better pair penetration than ramp `1/5 -> 50`.

### Rotating Gate Iter20 Results

```text
Fixed muSO3 = 10
maxSpeed      = 5.3493
maxUp         = 0.8816
minZ          = -0.1029
maxPairPen    = 0.3596
finalMaxSpeed = 0.0175

Fixed muSO3 = 50
maxSpeed      = 8.5150
maxUp         = 0.4809
minZ          = -0.0286
maxPairPen    = 0.3596
finalMaxSpeed = 0.2297

Ramp 1 -> 50
maxSpeed      = 23.8651
maxUp         = 17.9838
minZ          = -0.0849
maxPairPen    = 0.0363
finalMaxSpeed = 0.0123

Ramp 5 -> 50
maxSpeed      = 23.8710
maxUp         = 17.9939
minZ          = -0.0345
maxPairPen    = 0.0363
finalMaxSpeed = 0.0140

Ramp 10 -> 50
maxSpeed      = 23.8713
maxUp         = 17.9935
minZ          = -0.0314
maxPairPen    = 0.0363
finalMaxSpeed = 0.0136

AVBD Iter20
maxSpeed      = 7.7174
maxUp         = 1.3002
minZ          = -0.0783
maxPairPen    = 0.8434
finalMaxSpeed = 0.0084
```

Interpretation:

- Dynamic ramp is very promising at Iter20:
  - `maxPairPen` drops from fixed ABD's `0.3596` to `0.0363`.
  - It is far below AVBD's `0.8434`.
  - final speed approaches AVBD (`~0.013` vs `0.0084`).
- However, ramp creates a large transient correction spike (`maxUp ~= 18`), which is worse than fixed mu and AVBD.

### Research Diagnosis

The user's idea is valid: ramping `muSO3` can combine affine freedom and later rigidification better than a fixed value.

But a naive linear ramp has a new failure mode:

```text
Early affine freedom allows aggressive contact correction,
then late SO(3) stiffening can convert that correction into a transient upward velocity spike.
```

This is not a reason to abandon dynamic `muSO3`; it means the ramp should be coupled to a kinetic-energy or line-search condition.

### Next Experiment

Try adaptive ramp acceptance:

```text
If candidate step increases upward velocity or kinetic energy too much,
reduce the line-search step or delay muSO3 growth.
```

Alternative schedules:

```text
exponential ramp: small for most iterations, strong only at the end
contact-severity ramp: increase muSO3 only when pair penetration decreases
line-search-aware ramp: increase muSO3 only if accepted step >= threshold
```

## 2026-06-08 OrthoAL Attempt: Treating A^T A = I as AL Constraints

### Hypothesis

Instead of using an SO(3) proximal target from `polar(A)`, treat orthogonality as an Augmented Lagrangian equality constraint:

```text
A_L^T A_L = I
```

Independent scalar constraints:

```text
C0 = 0.5 * (a0 dot a0 - 1)
C1 = 0.5 * (a1 dot a1 - 1)
C2 = 0.5 * (a2 dot a2 - 1)
C3 = a0 dot a1
C4 = a0 dot a2
C5 = a1 dot a2
```

Each constraint has:

```text
lambda_ortho[i]
rho_ortho[i]
```

and contributes a Gauss-Newton term to the local affine Schur solve.

### Implementation

Added per-body state:

```text
Rigid::orthoLambda[6]
Rigid::orthoPenalty[6]
```

Added solver controls:

```text
affine_ortho_al
affine_ortho_beta
affine_ortho_penalty_min
affine_ortho_penalty_max
```

Tested two variants:

1. Pure OrthoAL: `muSO3 = 0`, `orthoPenaltyMin = 1`.
2. Soft OrthoAL: `orthoPenaltyMin = 0.001`, `betaOrtho = 1`, optionally with `muSO3 = 10` safety proximal.

### Results

Rotating Gate, 500 frames:

```text
Pure OrthoAL beta=100:
maxSpeed      ~= 25,817,884
minZ          ~= -24,766,916
Result        = catastrophic divergence

Pure OrthoAL beta=10:
maxSpeed      ~= 147,661
minZ          ~= -793,648
Result        = catastrophic divergence

Soft OrthoAL + muSO3=10:
maxSpeed      ~= 116.94
minZ          ~= -190.11
Result        = divergent / unusable

Soft OrthoAL, no proximal:
maxSpeed      ~= 466,399
minZ          grows to huge positive values
Result        = divergent / unusable

Reference ABD fixed muSO3=10:
maxSpeed      = 5.3493
minZ          = -0.1029
maxPairPen    = 0.3596
finalMaxSpeed = 0.0175

Reference ABD ramp 5 -> 50:
maxSpeed      = 23.8710
minZ          = -0.0345
maxPairPen    = 0.0363
finalMaxSpeed = 0.0140
```

### Diagnosis

The direct OrthoAL idea is mathematically natural but this first implementation is not stable.

Likely reasons:

- `A^T A = I` is nonlinear and nonconvex; a simple Gauss-Newton AL update can be too aggressive.
- Orthogonality constraints couple affine columns globally and can dominate the local contact solve.
- The dual variable update can accumulate large forces in modes that are not directly contact-damped.
- `A^T A = I` constrains `O(3)`, not strictly `SO(3)`, so determinant/reflection behavior is not controlled.
- Unlike contact constraints, orthogonality has no unilateral projection or natural physical damping.

### Research conclusion

This failure does not disprove the idea of AL-style rigidification, but it shows that a naive OrthoAL equality constraint is not a drop-in replacement for SO(3) proximal.

For now, the stable ABD route remains:

```text
SO(3) proximal target + local energy line search + possibly dynamic muSO3
```

Future OrthoAL would need a more careful formulation:

```text
- projective local/global orthogonality solve rather than direct AL,
- determinant-preserving SO(3) treatment,
- capped dual update,
- line-search-aware lambda/rho updates,
- or orthogonality penalty in strain space using singular values.
```

## 2026-06-08 Paper-Style Orthogonality Potential V_perp

### Correction

The paper's Section 3.2 does **not** use `A^T A = I` as an Augmented Lagrangian equality constraint. It uses a stiff orthogonality potential:

```text
V_perp = kappa * volume * || A A^T - I ||_F^2
```

In row-vector notation:

```text
V_perp = kappa * volume * (
  sum_i     (a_i dot a_i - 1)^2
  + sum_i!=j (a_i dot a_j)^2
)
```

This is a smooth penalty energy, not a lambda/rho AL constraint. This explains why the previous OrthoAL equality-constraint attempt was unstable.

### Implementation

Implemented a new optional ABD path:

```text
affine_ortho_potential
affine_ortho_stiffness
```

For the local affine variable `A_L`, the code adds the exact gradient and exact Hessian of:

```text
E = k || A_L^T A_L - I ||_F^2
```

to the affine Schur solve.

For columns `a_p`, define:

```text
S_pq = a_p dot a_q - delta_pq
grad_p = 4k * sum_q S_pq * a_q
H_pq = 4k * (delta_pq * sum_j a_j a_j^T + a_q a_p^T + S_pq I)
```

The Hessian contribution is added into `sumD[p][q]`, and the RHS is adjusted by:

```text
H * A0 - grad
```

matching a Newton step around the current local affine state.

### Rotating Gate Iter20: stiffness sweep

```text
Vperp stiffness = 1
maxSpeed      = 7.8326
maxUp         = 1.0252
maxAng        = 29.5904
minZ          = -0.0180
maxPairPen    = 0.3596
finalMaxSpeed = 0.0009

Vperp stiffness = 10
maxSpeed      = 15.5114
maxUp         = 6.9963
minZ          = -0.3038
maxPairPen    = 0.3813
finalMaxSpeed = 0.0209

Vperp stiffness = 100
maxSpeed      = 10.5178
maxUp         = 7.2007
minZ          = -0.1819
maxPairPen    = 0.0748
finalMaxSpeed = 0.0151

Fixed muSO3 = 10
maxSpeed      = 5.3493
maxUp         = 0.8816
minZ          = -0.1029
maxPairPen    = 0.3596
finalMaxSpeed = 0.0175

Ramp muSO3 5 -> 50
maxSpeed      = 23.8710
maxUp         = 17.9939
minZ          = -0.0345
maxPairPen    = 0.0363
finalMaxSpeed = 0.0140

AVBD Iter20
maxSpeed      = 7.7174
maxUp         = 1.3002
minZ          = -0.0783
maxPairPen    = 0.8434
finalMaxSpeed = 0.0084
```

### 1000-frame verification

```text
ABD Vperp stiffness = 1
maxSpeed      = 7.8326
maxUp         = 1.0252
minZ          = -0.0180
maxPairPen    = 0.3596
finalMaxSpeed = 0.0048

AVBD Iter20
maxSpeed      = 7.7174
maxUp         = 1.3002
minZ          = -0.0783
maxPairPen    = 0.8434
finalMaxSpeed = 0.0000
```

### Iter5 check

```text
ABD Vperp stiffness = 10
maxSpeed      = 6.0189
maxUp         = 3.5652
minZ          = -0.0653
maxPairPen    = 0.3596
finalMaxSpeed = 0.0000

ABD Vperp stiffness = 10 + muSO3 = 10
maxSpeed      = 9.4147
maxUp         = 2.0758
minZ          = -0.0091
maxPairPen    = 0.1222
finalMaxSpeed = 0.1107

AVBD Iter5
maxSpeed      = 6.7564
maxUp         = 2.3662
minZ          = -0.1015
maxPairPen    = 1.3024
finalMaxSpeed = 0.0658
```

### Stacking and Tilted Drop sanity checks

```text
ABD Stacking Vperp stiffness = 1
Result: unstable by frame ~300, very large velocity and nan flag.

ABD Tilted Drop Vperp stiffness = 1
Result: stable, finalMaxSpeed ~= 0.0141
```

### Interpretation

The paper-style orthogonality potential is dramatically better than naive OrthoAL:

- It does not require polar decomposition.
- It gives a smooth penalty energy.
- With exact Hessian, it can be stable and competitive in high-rotation contact.
- It can outperform AVBD on Rotating Gate penetration while keeping similar transient speeds.

However, it is not universally stable yet:

- It fails on simple stack with `stiffness = 1`.
- This suggests the orthogonality potential strength should be scene/contact dependent or combined with the existing SO(3) proximal for stacks.

Current best interpretation:

```text
V_perp is the paper-correct ABD rigidification mechanism.
It is a better research path than OrthoAL.
For high-rotation contact, V_perp exact Hessian is very promising.
For stacking, current tuning is not robust yet.
```

Next focused experiment:

```text
Use V_perp as the primary rigidification term.
Find a safe stiffness schedule:
- low stiffness for high-rotation gate,
- stronger or hybrid proximal for stacks,
- possibly adaptive based on ||A^T A - I||_F and contact penetration.
```

## 2026-06-09 Vperp Without Line Search

### Question

If the paper-style `V_perp` exact Hessian is stable enough, can we remove ABD's local line search to save time?

### Implementation

Added:

```text
affine_line_search
```

When disabled, ABD accepts the local affine update directly. Also changed the SO(3) proximal path so that if `muSO3 <= 0`, it no longer computes `polar_rotation` for the proximal target.

### Results

#### Rotating Gate Iter20, Vperp stiffness = 1

```text
With line search:
maxSpeed      = 7.8326
maxUp         = 1.0252
minZ          = -0.0180
maxPairPen    = 0.3596
finalMaxSpeed = 0.0009

No line search:
maxSpeed      = 7.8325
maxUp         = 1.1029
minZ          = -0.0273
maxPairPen    = 0.3596
finalMaxSpeed = 0.0030
```

Interpretation:

- No line search remains stable.
- It is slightly worse on final speed and ground penetration, but still very good and far better than AVBD on pair penetration.

#### Rotating Gate Iter5, Vperp stiffness = 10

```text
With line search:
maxSpeed      = 6.0189
maxUp         = 3.5652
minZ          = -0.0653
maxPairPen    = 0.3596
finalMaxSpeed = 0.0000

No line search:
maxSpeed      = 9.3554
maxUp         = 2.0627
minZ          = -0.0137
maxPairPen    = 0.1220
finalMaxSpeed = 0.0115
```

Interpretation:

- No line search is stable.
- It improves pair penetration and ground penetration.
- It has a slightly larger final residual speed.

#### Tilted Drop, Vperp stiffness = 1

```text
With line search:
finalMaxSpeed = 0.0141
minZ          = -0.0136

No line search:
finalMaxSpeed = 0.0115
minZ          = -0.0186
```

Interpretation:

- No line search remains stable.
- Difference is small.

### Current Conclusion

For the paper-style `V_perp` path, line search is not always necessary in the tested high-rotation and tilted-drop cases.

However:

- `ABD: Stacking Vperp1` is unstable even with line search.
- So line-search removal is safe only for specific `V_perp` scenes tested so far, not as a universal ABD setting.

Recommended current research direction:

```text
Use V_perp exact Hessian as the main rigidification term.
Disable SO(3) proximal when V_perp is active.
Try no-line-search variants for high-rotation benchmarks.
Keep line search available as a safety toggle for harder contact scenes.
```
