# CUDA PABD Block-Jacobi Global-Local Research Plan

## Goal

Replace the PCG global solve in CUDA PABD with a cheaper block-Jacobi style
global update:

- build only per-body / per-ABD-element diagonal Hessian blocks;
- invert one local 12x12 block per body, corresponding to four 3D control
  points;
- apply this once per global solve pass;
- recover accuracy and stability by running multiple global-local outer
  iterations.

The first target is the interpolated ABD scenes, especially:

- `CUDA PABD: PD+ABD Boxes`
- `CUDA PABD: Tetra EE Cross`
- `CUDA PABD: Rigid-IPC Chain Net 4x4/8x8/32x32`
- `CUDA PABD: Vertical Link Chain`

## Current Solver Shape

The current interpolated path assembles:

- fixed topology elastic/mass terms into `BlockCSR3 H`;
- ground/self contact RHS plus contact diagonal baking;
- off-diagonal contact action through `WideContactSpMVOp` inside PCG.

The relevant path is:

- `PabdCudaSolver::step_interpolated`
- `PabdCudaSolver::assemble_interpolated_system_gpu`
- `PCGSolver::solve`
- `collision::WideContactSpMVOp`

The proposed experiment should not remove PCG immediately. It should add a
separate solver mode so PCG and block-Jacobi can be compared frame by frame.

## Proposed Algorithm

For each global-local outer iteration:

1. Predict inertial positions as today.
2. Run local step:
   - update per-element rotations from current control positions;
   - update interpolated surface positions;
   - detect ground/self contacts.
3. Assemble a body-local system for each ABD element/body:
   - local Hessian block `H_i` is 12x12;
   - local RHS block `b_i` is 12-vector;
   - include mass, elasticity, fixed constraints, and diagonalized contact terms.
4. Invert each `H_i` independently on GPU.
5. Update the four control points of that body once:
   - direct solve form: `x_i = inv(H_i) * b_i`;
   - or residual form: `x_i += omega * inv(H_i) * (b_i - H_i x_i)`.
6. Apply fixed constraints.
7. Repeat global-local outer iterations several times.
8. Update velocity from the final positions.

The residual form is the safer first implementation because it behaves like a
damped block-Jacobi iteration and gives a relaxation knob `omega`.

## 12x12 Block Definition

Each ABD body is represented by four control vertices:

```text
q_i = [x0.yz..., x1, x2, x3] in R^12
```

The block Hessian layout should be:

```text
H_i = 4x4 blocks of 3x3 matrices
```

Sources:

- mass block: existing `tet_mass_blocks_dev_`, scaled by `1 / dt^2`;
- elastic block: current PD term `stiffness * volume * grad_a dot grad_b * I3`;
- fixed point block: add `fixed_weight * I3` on fixed local vertices;
- ground contact block: for each surface sample mapped to body `i`, add
  `k * w_a * w_b * n n^T`, with ground normal `n = (0,1,0)`;
- self contact block: for contact coefficients that touch body `i`, accumulate
  only the local-local part into `H_i`.

For contacts touching two bodies, first version should add only each body's
diagonal block and ignore cross-body off-diagonal terms. That is exactly the
block-Jacobi approximation. Later we can compare against PCG by measuring the
missing cross term residual.

## Implementation Steps

1. Add a solver mode enum.

```cpp
enum class PabdGlobalSolverMode {
    PCG,
    BlockJacobi12
};
```

Expose it in `PabdCudaParams` and the ImGui UI.

2. Add GPU buffers.

```text
body_hessian_12x12_dev   [body_count * 144 floats]
body_rhs_12_dev          [body_count * 12 floats]
body_delta_12_dev        [body_count * 12 floats]
body_control_ids_dev     [body_count * 4 ints]
```

For current ABD scenes, `body_count` can be derived from the surface maps or
from the control tet list.

3. Add assembly kernels.

Start with separate focused kernels:

- `clear_body_system_kernel`
- `assemble_body_mass_elastic_kernel`
- `assemble_body_fixed_kernel`
- `assemble_body_ground_contacts_kernel`
- `assemble_body_self_contacts_kernel`

Keep the existing PCG assembly untouched.

4. Add one local 12x12 solve kernel.

Use a small dense solver per body:

- first version: Cholesky LDLT if the block is SPD enough;
- fallback: Gauss-Jordan with diagonal regularization;
- always add a small `diag_epsilon` before inversion.

Recommended first regularization:

```text
diag_epsilon = max(1e-6, 1e-6 * average(diag(H_i)))
```

5. Apply update.

Use relaxation initially:

```text
x_new = lerp(x_old, x_solved, omega)
```

Start with:

```text
omega = 0.2 to 0.8
outer_iterations = 4 to 20
```

6. Add diagnostics.

Print or expose:

- max block condition proxy: `max_diag / min_diag`;
- failed 12x12 inversions;
- max update length;
- max contact count;
- minY and penetration summary;
- optional residual estimate against the existing PCG operator.

## Validation Plan

Phase 1: no collision

- Single tetra pinned
- PD+ABD Boxes with ground/self disabled
- Compare one frame against PCG with same mass/elastic/fixed system

Phase 2: ground only

- Single raised box onto ground
- Check no underground drift after 5000 frames
- Sweep `outer_iterations` and `omega`

Phase 3: self contact only

- Tetra EE Cross
- Vertical Link Chain 2, 4, 10
- Check PF/EE contacts generate visible response

Phase 4: chain nets

- Chain Net 4x4
- Chain Net 8x8
- Chain Net 32x32 only after 4x4 and 8x8 agree qualitatively

## Expected Differences From PCG

Block-Jacobi will be cheaper per iteration but less globally coupled:

- contacts between two bodies lose cross-body Hessian blocks;
- stiff chains may look softer unless outer iterations are increased;
- high contact stiffness may need damping or relaxation;
- convergence will depend strongly on body ordering only if we later switch to
  Gauss-Seidel; pure Jacobi is order-independent but slower.

This is acceptable for the research version because the point is to test
whether repeated global-local passes can replace a clean PCG solve.

## First Milestone

Implement `BlockJacobi12` only for interpolated ABD bodies with:

- mass + elastic + fixed;
- ground contact;
- self contact diagonal blocks;
- no cross-body off-diagonal contact terms;
- no removal of the existing PCG path.

Success criterion:

- `CUDA PABD: PD+ABD Boxes` and `Vertical Link Chain 2` run without NaNs;
- frame timing shows PCG launch sequence is gone in this mode;
- increasing outer iterations visibly improves contact stiffness.

## Open Questions

- Should the local update solve absolute positions `H x = b` or residual
  correction `H dx = r`? Start with residual correction for damping control.
- Should self contacts be rebuilt every outer iteration or fixed for one frame?
  Start with rebuild every outer iteration for correctness, then test reuse.
- Should contact stiffness be scaled by outer iteration count? Keep it unchanged
  first, then introduce scaling only if stacks become too soft.
- Should the 12x12 block be per tet or per logical rigid body? For the current
  ABD scenes they are the same idea: four control points define one body.
