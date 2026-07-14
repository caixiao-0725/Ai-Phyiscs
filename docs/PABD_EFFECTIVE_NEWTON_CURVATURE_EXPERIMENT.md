# PABD Effective Newton Curvature Experiment

Date: 2026-07-13

## Goal

Test whether the current ARAP elastic energy can use a state-dependent,
positive-semidefinite Newton-like curvature without a per-body 9x9
eigendecomposition, and measure the effect on deformation and GPU time.

Production scene settings are left unchanged. In particular, each scene keeps
its existing time step, global-local iteration count, global solver, and PCG
iteration count.

## Exact structure

The elastic energy per body is

```text
E(F) = 0.5 * k * V * ||F - polar(F)||_F^2.
```

Write the polar decomposition as `F = R S`, and express a perturbation in the
polar frame as `A = R^T dF`. Split `A` into symmetric and skew parts:

```text
A = sym(A) + hat(w),  w = vex(skew(A)).
```

The exact Hessian-vector product is

```text
R^T H[dF] = sym(A) + hat(G w),
G = I - 2 * (tr(S) I - S)^-1.
```

Therefore the 9x9 Hessian has six constant unit-curvature symmetric modes and
only three state-dependent rotational modes. Relative to the rest co-rotated
Hessian, the exact Newton correction has rank at most three. A 9x9
eigendecomposition is unnecessary for this energy.

The implementation regularizes the three diagonal entries of
`tr(S) I - S` by `1e-7 * max(abs(tr(S)), 1)` before inversion.

## Curvature modes

| Mode | Local Hessian-vector product | PSD | Per-body eigensolve |
|---|---|---:|---:|
| `pd` | `A` | yes | none |
| `corotated_rest` | `sym(A)` | yes | none |
| `polar_gn` | `sym(A) + hat(G^2 w)` | yes | none |
| `projected_newton3` | `sym(A) + hat(PSD(G) w)` | yes | fixed 3x3 Jacobi |

`polar_gn` is the Gauss-Newton curvature of the polar residual. It is not the
same as clipping the exact Hessian: a negative exact rotational eigenvalue is
squared instead of discarded.

For all four modes, `H[F] = F`. Since `grad E = k V (F - R)`, the linearized
right-hand side remains exactly `k V R`; only the Hessian changes.

## Validation

`pabd_curvature_experiment` compares the analytic 9x9 Hessian to central finite
differences of the ARAP gradient. The relative Frobenius error is
`1.891713e-11`. It also compares the assembled four-control block product with
the matrix-free body operator; their relative error is `5.216970e-8`.

Selected exact rotational eigenvalues are:

| Principal stretches | Exact rotational eigenvalues | Positive rank |
|---|---|---:|
| `(0.55, 0.55, 0.55)` | `(-0.8182, -0.8182, -0.8182)` | 0 |
| `(0.95, 0.95, 0.95)` | `(-0.0526, -0.0526, -0.0526)` | 0 |
| `(1.05, 1.05, 1.05)` | `(0.0476, 0.0476, 0.0476)` | 3 |
| `(1.60, 1.60, 1.60)` | `(0.3750, 0.3750, 0.3750)` | 3 |
| `(0.70, 1.00, 1.30)` | `(0.1304, 0.0000, -0.1765)` | 1 |
| `(0.55, 1.15, 1.70)` | `(0.2982, 0.1111, -0.1765)` | 2 |

This is the useful structural result: indefiniteness is isolated to a 3x3
rotational block, and its positive part has state-dependent rank 0 to 3.

## GPU microbenchmark

Workload: 4,194,304 bodies, 80 repetitions, one curvature HVP per body.

| Mode | Total time | Time per body | Throughput |
|---|---:|---:|---:|
| `pd` | 36.946 ms | 0.1101 ns | 9.082 Gbody/s |
| `corotated_rest` | 36.887 ms | 0.1099 ns | 9.097 Gbody/s |
| `polar_gn` | 34.860 ms | 0.1039 ns | 9.626 Gbody/s |
| `projected_newton3` | 35.205 ms | 0.1049 ns | 9.531 Gbody/s |

The kernel is bandwidth-bound, so the small ordering differences are noise,
not evidence that the more complex modes are faster. The supported conclusion
is that the isolated curvature HVP arithmetic is effectively hidden by memory
traffic on the tested GPU.

## Scene measurements

### Rigid-IPC Chain Net 32x32

Settings: 2,880 bodies, stiffness `1e5`, the scene's existing PCG solver and
30 PCG iterations, 240 frames, 40 warmup frames.

| Mode | Mean step | p95 | Stretch error at frame 240 | Finite |
|---|---:|---:|---:|---:|
| `pd` | 5.988 ms | 6.492 ms | `6.406e-3` | yes |
| `corotated_rest` | 5.984 ms | 6.526 ms | `6.130e-3` | yes |
| `polar_gn` | 5.962 ms | 6.498 ms | `6.230e-3` | yes |
| `projected_newton3` | 6.206 ms | 6.819 ms | `6.061e-3` | yes |

At frame 120, independently recomputed PCG relative residuals are
`1.143e-7` to `1.178e-7` for all four modes. No body preconditioner factorization
failed. The similar recursive residuals are therefore not false convergence
caused by recursive residual underflow.

`polar_gn` is effectively free at this scale. `projected_newton3` costs about
3.6 percent over `pd` in the 240-frame run.

### Vertical Link Chain 16

This scene currently uses `BlockJacobi12`, not PCG. The table reports the peak
sampled stretch error over 600 frames; contacts and trajectories diverge after
the mode switch, so this is a stability/stiffness probe rather than a paired
trajectory error.

| Stiffness | PD | Rest | Polar GN | Projected Newton 3 |
|---:|---:|---:|---:|---:|
| `1e4` | `2.582e-1` | `2.294e-1` | `2.322e-1` | `3.304e-1` |
| `1e5` | `3.804e-2` | `6.363e-2` | `3.394e-2` | `4.286e-2` |
| `5e5` | `2.563e-3` | `5.772e-3` | `5.121e-3` | `5.762e-3` |

At stiffness `5e5`, mean step times were 0.578 ms (`pd`), 0.605 ms (rest),
0.609 ms (`polar_gn`), and 0.617 ms (projected Newton 3). The final minimum Y
was about -11.5 for PD and -13.2 to -13.3 for the rotation-aware modes. This is
consistent with PD's artificial unit curvature resisting rotational increments,
but a controlled reference is still required before labeling either trajectory
more accurate.

### Single torque gear

The isolated motor scene has no gear-gear contact. Across stiffness `1e3` to
`5e5`, `polar_gn` was generally within 0 to 4 percent of PD step time, while
`projected_newton3` was roughly 5 to 7 percent slower. Angular velocity differs
substantially among modes because the current integrator performs one
global-local pass and the curvature directly changes the hinge/motor response.
That difference demonstrates effective rotational stiffness, but is not an
accuracy result without a converged reference solution.

## Findings

1. The expensive 9x9 PSD projection can be replaced exactly by a 3x3
   rotational PSD projection for the current ARAP energy.
2. `polar_gn` is the best current performance candidate. Its local arithmetic
   is negligible and its full-scene overhead is small.
3. At high stiffness, measured `||S-I||` and `||G||` become small. Consequently,
   all rotation-aware corrections approach the rest co-rotated Hessian, so a
   large stiffness-preservation gain should not be expected from curvature alone.
4. PD often appears stiffer because it assigns unit curvature to pure rotational
   increments. That can suppress deformation, but it is not physical ARAP
   curvature and can inhibit articulation.
5. The visible full-scene cost comes from packed 12x12 preconditioner work and
   contact processing, not from evaluating the 3x3 curvature formula.

## Matrix-free rank-3 backend

The PCG path now has two switchable Polar-GN backends:

| Backend | Elastic CSR assembly | PCG elastic application | Body12 base |
|---|---|---|---|
| `assembled12` | full dynamic 3x3 blocks | CSR SpMV | dynamic world frame |
| `matrix_free_rank3` | omitted | co-rotated body sidecar | cached local rest frame |

The matrix-free operator evaluates

```text
H_eff = H_rest + B_rot^T C(S) B_rot,
```

with `C(S) = G^2`. Static gradients represent `H_rest`; each global-local pass
updates only `R`, `G^2`, and `k V`. These arrays keep stable addresses, so the
PCG CUDA Graph remains reusable. The Body12 preconditioner caches its
mass/rest/hinge/fixed base in the local frame, adds the same rank-3 correction
and local-frame contact blocks, then rotates residuals and solutions around the
triangular solve.

The backend requires canonical disjoint groups of four control rows and PCG.
If BlockJacobi12 is selected, the solver intentionally falls back to the
assembled curvature path.

On Chain Net 32x32 at stiffness `1e5` and the scene's existing 30 PCG
iterations, a 240-frame A/B run measured:

| Backend | Mean step | p95 | Frame-240 stretch | Finite |
|---|---:|---:|---:|---:|
| `assembled12` | 6.114 ms | 6.579 ms | `6.099e-3` | yes |
| `matrix_free_rank3` | 6.098 ms | 6.478 ms | `5.962e-3` | yes |

The 0.27 percent mean difference is within run-to-run noise, so this is not yet
a demonstrated speedup. The matrix-free true relative PCG residual at frame
120 was `9.225e-8`, with no factorization failure. On a one-body scene the
extra sidecar launch is slower, as expected.

The next performance step is to fuse the body elastic sidecar with a PCG vector
kernel or replace the remaining per-body Cholesky rebuild with a three-column
Woodbury update. The controlled pure-rotation and torque-equilibrium reference
fixtures are still needed for an accuracy claim.

For accuracy, compare against a separate converged reference in two controlled
fixtures: prescribed pure rotation (exact energy and torque should remain zero)
and a cantilever or torque equilibrium with no contact. Keep production scene
iteration counts unchanged; the high-accuracy solve is only the offline
reference.

## Reproduction

```powershell
cmake --build build --config Release --target pabd_curvature_experiment
.\build\Release\pabd_curvature_experiment.exe --samples 4194304 --repeats 80

$env:CHYSX_PABD_ELASTIC_CURVATURE = 'polar_gn'
$env:CHYSX_PABD_POLAR_GN_BACKEND = 'matrix_free_rank3'
$env:CHYSX_PABD_STIFFNESS = '100000'
$env:CHYSX_PABD_CURVATURE_DIAGNOSTICS = '1'
.\build\Release\chysx_scene.exe `
  --scene 'CUDA PABD: Rigid-IPC Chain Net 32x32' `
  --frames 240 --timing-warmup 40 --no-export
```

The viewer exposes both `elastic curvature` and `Polar GN backend` combos.
