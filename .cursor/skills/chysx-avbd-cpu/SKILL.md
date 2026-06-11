---
name: chysx-avbd-cpu
description: Provides ChysX avbd_cpu solver context for AVBD, PABD, and ABD. Use when analyzing src/rigid/avbd_cpu, debugging rigid-body stacking stability, comparing AVBD/PABD/ABD, or discussing innovation points around affine-matrix ABD.
disable-model-invocation: true
---

# ChysX avbd_cpu Solver Context

## Scope

Use this skill when working in `src/rigid/avbd_cpu` or scenes that exercise these solvers.

There are three related rigid-body algorithms in this folder:

- `AVBD`: the reproduced paper algorithm and current SOTA baseline. Treat it as the most trusted implementation.
- `PABD`: a new solver from the user's advisor. Its stability is still being validated.
- `ABD`: the user's derivation based on AVBD, replacing axis-angle rotation with an affine matrix representation. This is implemented on CPU and is a potential innovation direction.

## ABD Research Intent

`ABD` is intended as a research extension of the SOTA `AVBD` paper algorithm: keep the Augmented Lagrangian solver structure, but replace the axis-angle rotational representation with an affine matrix representation. The goal is not to simply copy `PABD`'s barrier model, but to understand whether an affine local representation can improve robustness, convergence, or extensibility while staying close to AVBD.

The derivation in `docs/isotropic_vs_anisotropic.md` is central context. In the exact directional formulation, a material point is written as:

```text
y = c_tilde + A_tilde * (c_L + A_L * r)
```

For a world-space directional constraint `n_w^T y - t`, define `n_tilde = A_tilde^T n_w`, so:

```text
C_d = n_tilde^T * (c_L + A_L * r) + xi_d
J_d = [n_tilde^T, n_tilde^T r0, n_tilde^T r1, n_tilde^T r2]
H_d = kappa_d * J_d^T * J_d
```

The key structural point is that `H_d` is anisotropic: its 3x3 subblocks use `N = n_tilde outer n_tilde`, not the isotropic `I_3` used by a full 3D PABD bond. This is why `AffineSchurSolver::accumulate_dir()` is the relevant implementation path for exact directional contacts.

For ABD's paper-style rigidification potential, see `docs/ABD_Orthogonality_Potential.md`. It derives `V_perp = k ||AA^T-I||_F^2`, its gradient, exact Hessian, Gauss-Newton alternative, and positive-definiteness properties. Key point: this is a smooth potential, not an Augmented Lagrangian equality constraint.

## User Goals

The user wants to:

- Ensure CPU `PABD` and CPU `ABD` are stable enough for comparison and development.
- Understand why `ABD` stacking is unstable despite the expectation that `ABD` should be at least as stable as `PABD`.
- Preserve Augmented Lagrangian reasoning for `ABD`; do not switch ABD to barrier energy unless explicitly requested.
- Use diagnostic output and batch scene runs freely to understand solver behavior.

## Debugging Priorities

When debugging `ABD`, focus on:

- Energy injection during collision response, especially after position correction is converted back to velocity.
- Interaction between affine updates, `polar_rotation`, and constraint satisfaction.
- Penalty and lambda growth across Augmented Lagrangian iterations.
- Contact multiplicity: ground and box-box contacts can apply many simultaneous penalty forces.
- Velocity recovery choice: absolute recovery from corrected positions can turn over-correction into large rebound velocity.
- Differences from `PABD`: PABD uses IPC-style barrier/contact energy and local solves, while ABD uses AL constraints and Schur-complement affine updates.
- Whether SO(3) handling is part of the local objective or an external post-process. A direct `polar_rotation(affine)` after solving can invalidate the just-satisfied AL contact constraints and inject energy. Prefer local/global SO(3) proximal terms or line-search-controlled projection over unconstrained post-projection.
- Whether the local step has an energy/trust mechanism. ABD can look algebraically stronger than PABD but still be less stable if its affine update has no monotone energy check.

## Useful Scenes

- `ABD: Stacking`: stresses box-box and ground contacts, useful for detecting energy injection.
- `AVBD: Stacking`: same 50-box simple stack geometry as `ABD: Stacking`, useful as the fair SOTA baseline.
- `PABD: Simple Stacking`: same 50-box simple stack geometry as `ABD: Stacking`, useful as the fair PABD comparison.
- `ABD: Tilted Drop`: checks whether an affine rigid body falling at an angle settles flat without exploding.
- Compare with `PABD` and `AVBD` scenes only as behavioral references; do not assume their contact model should be copied into ABD.

## Batch Metrics

`chysx_scene.exe` prints `[METRICS ...]` lines for scenes that expose solver metrics. Use these for controlled comparisons:

```text
maxSpeed
maxUp
maxAng
minZ
finalMaxSpeed
finalMinZ
nan
```

For fair 50-box stacking comparisons, run:

```text
chysx_scene.exe --scene "ABD: Stacking" --frames 500
chysx_scene.exe --scene "PABD: Simple Stacking" --frames 500
chysx_scene.exe --scene "AVBD: Stacking" --frames 500
```

Current useful ABD ablation scenes:

```text
ABD: Stacking              # current best, betaLin=100
ABD: Stacking Incremental  # velocity recovery ablation
ABD: Stacking Beta100      # explicit beta=100 alias
ABD: Stacking Beta1000
ABD: Stacking Beta10000
ABD: High Rotation Rods    # high-angular-velocity contact stress test
AVBD: High Rotation Rods
PABD: High Rotation Rods
ABD: Rotating Gate         # controlled high-rotation narrow-obstacle benchmark
ABD: Rotating Gate Mu10
ABD: Rotating Gate Iter20 Mu5/Mu10/Mu200/Mu500
AVBD: Rotating Gate
PABD: Rotating Gate
```

Recent experimental takeaway: AVBD-style residuals `C = C0*(1-alpha) + J*dq` significantly improve ABD damping. ABD prefers much slower penalty growth than AVBD in the 50-box frictionless stack; `betaLin=100` is currently better than `1000` or `10000`.

Current ABD-favorable evidence: `Rotating Gate` is the more convincing benchmark. It uses a single high-speed rotating thin rod interacting with static narrow gate obstacles and ground. In iteration sweeps at 2/5/10/20 iterations, ABD consistently has lower body-body penetration (`maxPairPen`) than AVBD. At 2 and 5 iterations ABD also has lower final residual speed; at 10 and 20 iterations AVBD damps residual motion better. Frame the current ABD advantage as low-iteration high-rotation contact accuracy, not universal damping superiority. `High Rotation Rods` is still useful as a stress probe but is less controlled.

`muSO3` controls an important trade-off. In `Rotating Gate`, lower values such as `muSO3=10` improve final damping, while stronger values such as `muSO3=50` can reduce pair penetration. Future ABD work should consider adaptive SO(3) proximal strength rather than a fixed global value.

Dynamic `muSO3` ramping is promising but not solved. A linear ramp from low to high `muSO3` can dramatically reduce `maxPairPen` in `Rotating Gate Iter20` (e.g. down to `0.0363`, far below AVBD), while keeping final speed close to AVBD. Its main failure mode is a large transient upward velocity spike, so future ramp schedules should be kinetic-energy-aware or line-search-aware rather than purely linear.

Important correction from the ABD paper: Section 3.2 uses the orthogonality potential `V_perp = kappa * volume * ||A A^T - I||_F^2`, not an Augmented Lagrangian equality constraint `A^T A = I`. A naive OrthoAL implementation diverged badly. The paper-style `V_perp` with exact gradient/Hessian is the correct research direction.

Current `V_perp` evidence: `ABD: Rotating Gate Iter20 Vperp1` is strong. It has `maxPairPen=0.3596`, `minZ=-0.0180`, `maxUp=1.0252`, and `finalMaxSpeed=0.0009`, compared with AVBD Iter20 `maxPairPen=0.8434`, `minZ=-0.0783`, `maxUp=1.3002`, and `finalMaxSpeed=0.0084`. However `ABD: Stacking Vperp1` is unstable, so `V_perp` currently helps high-rotation contact but is not yet a universal replacement.

## Reporting

When reporting findings:

- Separate observed runtime evidence from hypotheses.
- Compare ABD against AVBD/PABD by solver mechanism, not only by scene output.
- Prefer concise explanations of why instability happens: where energy enters, which state variables drift, and which correction step amplifies it.
