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
   - include mass, elasticity, fixed constraints, and the local-local contact
     block;
   - evaluate cross-body contact blocks at the previous Jacobi iterate and
     move them to the RHS: `b_i_eff = b_i - sum_j!=i A_ij x_j_old`.
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

For contacts touching two bodies, store only each body's diagonal 12x12 block.
The cross-body blocks are not stored, but they must still be evaluated using
the previous Jacobi iterate and subtracted from the local RHS. Dropping them
from both the block and RHS changes a relative contact target into an absolute
target around the origin and can produce a large one-step jump.

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
body_hessian_12x12_dev   [body_count * 78 floats]   // packed symmetric 12x12
body_rhs_12_dev          [body_count * 12 floats]
body_delta_12_dev        [body_count * 12 floats]
body_control_ids_dev     [body_count * 4 ints]
```

The 12x12 block is symmetric, so the research implementation stores only the
lower triangular part:

```text
packed_index(r, c) = r * (r + 1) / 2 + c,  r >= c
```

This reduces Hessian storage from 144 floats/body to 78 floats/body.  A more
specialized ABD format could store 10 symmetric 3x3 sub-blocks, i.e. 60
floats/body, but the 78-float format is the first target because it maps
directly to dense Cholesky/LDLT and is easier to compare against PCG.

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

## CUDA Optimization Audit

This section records the optimization audit of the current research
implementation. The main conclusion is that the 12x12 factorization is not the
first bottleneck. The current contact path launches work from buffer capacity
and scatters a large number of floating-point atomics before the body solve.

### Measured Baseline

The measurements below were collected from the Release build on an RTX 5090
(SM 12.0, CUDA 13.0) with Nsight Systems. They are one-run timing samples, not
final benchmark numbers. Nsight Compute hardware-counter collection was
blocked by `ERR_NVGPUCTRPERM`, so memory-throughput and occupancy counters still
need to be collected after enabling GPU performance-counter access.

The most useful probe was the first frame of `Rigid-IPC Chain Net 32x32` forced
temporarily onto Block-Jacobi:

```text
bodies                 = 2,880
surface vertices       = 633,600
wide-contact capacity  = 10,137,600
valid wide contacts    = 0
```

Even with zero valid contacts, kernels launched from maximum capacity still
did substantial work:

| Kernel | Time (us) |
| --- | ---: |
| `append_interpolated_self_contacts_kernel` | 38.080 |
| `bake_wide_contact_diag_kernel` | 20.416 |
| `add_local_wide_contacts_to_body_hessian_kernel` | 40.161 |
| `subtract_cross_body_wide_contacts_from_rhs_kernel` | 40.192 |
| Contact-path subtotal with zero contacts | 138.849 |
| `assemble_interpolated_mass_elastic_kernel` | 15.968 |
| `pack_interpolated_body_hessian_kernel` | 10.689 |
| `solve_interpolated_block_jacobi_kernel` | 10.336 |

This does not mean collision detection is cheap; broad phase and narrow phase
have their own costs. It means that, inside the Block-Jacobi assembly/solve
route, empty capacity-sized contact passes already cost more than the actual
2,880-body solve.

`cuobjdump --dump-resource-usage` reports:

| Kernel | Registers/thread | Stack | Local |
| --- | ---: | ---: | ---: |
| 12x12 solve | 130 | 0 | 0 |
| body Hessian pack | 40 | 0 | 0 |
| local contact Hessian scatter | 36 | 0 | 0 |
| cross-body RHS scatter | 40 | 0 | 0 |

The compiler scalarizes the current `float A[12][12]` solve, so it does not
spill to local memory in this build. The cost is high register pressure, which
can limit occupancy. This should be optimized after the capacity-sized launch
and contact-scatter problems.

### Current Atomic Cost

For one active self contact with two four-control bodies, the current upper
bound is approximately:

| Pass | Float atomics/contact |
| --- | ---: |
| target RHS in `append_interpolated_self_contacts_kernel` | 24 |
| per-control 3x3 diagonal in `bake_wide_contact_diag_kernel` | 72 |
| local off-diagonal 12x12 blocks | 108 |
| lagged cross-body RHS | 24 |
| Total | 228 |

The exact number is lower when coefficients are zero. Counter atomics are not
included. A ground contact has only one body side but can still issue roughly
102 float atomics through the corresponding passes.

The problem is data ownership: contact threads scatter into body matrices and
body RHS vectors. Block-Jacobi naturally wants the opposite ownership model:
one body worker should gather all contacts belonging to that body and write its
local system exactly once.

### P0: Stop Launching From Capacity

All kernels whose valid count lives on the device should use a bounded
persistent grid instead of one thread per reserved slot:

```cpp
const int n = min(*count_dev, capacity);
for (int c = global_thread_id; c < n; c += total_threads) {
    process(c);
}
```

Choose a fixed launch size from an occupancy query or a small multiple of the
SM count, for example 2 to 4 resident blocks per SM. The launch shape remains
constant and CUDA-Graph friendly, while a zero contact count executes only the
bounded grid instead of millions of empty threads.

Apply this first to:

- `append_interpolated_self_contacts_kernel`;
- `bake_wide_contact_diag_kernel`;
- `add_local_wide_contacts_to_body_hessian_kernel`;
- `subtract_cross_body_wide_contacts_from_rhs_kernel`.

This is a low-risk scheduling change. It does not alter the contact set,
matrix, RHS, time step, or iteration counts.

Implementation status (2026-07-10): completed for these four kernels with a
maximum of four blocks per SM (680 blocks on the RTX 5090) and a
device-count-driven grid-stride loop. On the same
three-frame 32x32 zero-contact probe, their combined average GPU time changed
from 138.849 us to approximately 3.477 us, about a 40x reduction. The
180-frame `Vertical Link Chain 2` trace retained the same PF/EE counts and the
same reported minY values at frames 60, 120, and 180.

As an intermediate Block-Jacobi-only optimization, skip
`bake_wide_contact_diag_kernel` and let the local body-Hessian pass add both the
diagonal and off-diagonal parts directly to the packed body block. PCG still
needs its existing diagonal bake. This removes one kernel and avoids writing
contact data into `H_.diag` only to read and repack it immediately.

### P1: Bind Contacts To Bodies

Yes, contacts should be bound to rigid bodies for the Block-Jacobi route. The
binding has two forms:

- ground samples have fixed topology and can be bound to bodies once at setup;
- self contacts are dynamic and must be rebound every collision/local pass.

The implementation uses a body-contact ELL adjacency instead of CSR. Memory is
intentionally traded for one-pass construction and fixed-stride access:

```text
body_contact_counts  [body_count]
body_contact_refs    [ell_width * body_count], slot-major

BodyContactRef = uint32(contact_id) + one side bit
```

Build it entirely on the GPU:

1. Zero `body_contact_counts[body_count]`.
2. For every valid `WideContact`, reserve one ELL slot on body 0 and body 1.
   This is at most two integer atomics per contact.
3. Write each encoded reference directly to
   `body_contact_refs[slot * body_count + body]`.
4. Launch one body worker per body; it gathers its contact references and owns
   all writes to its 12x12 block and 12-vector RHS.

This removes the CSR count/scan/scatter sequence and replaces up to hundreds
of contended float atomics per contact with at most two integer atomics. The
slot-major layout makes equal-slot reads contiguous across neighboring body
workers. The width policy reserves four times the capacity-derived maximum
average reference degree:

```text
ell_width = min(wide_capacity,
                round_up_pow2(max(64,
                                  ceil(8 * wide_capacity / body_count))))
```

An overflow counter reports dropped references; overflow is never silent. The
32x32 chain scene uses width 32768 and 360 MiB for ELL references, which is an
intentional trade on the 32 GiB target GPU.

Do not permanently attach a self contact to a body. The body IDs are fixed,
but contact existence, weights, normal, and target are dynamic. Rebuild only
the compact adjacency, not the geometry topology.

Ground contacts currently enter the same dynamic ELL as self contacts. A later
step can prebind `body_surface_ids` and let the body worker evaluate ground
samples directly, preserving the existing separation between ground and
self-collision sampling while removing the ground append pass.

Implementation status (2026-07-10): the ELL builder and overflow diagnostics
are active in the Block-Jacobi branch. Contact target RHS, the complete
same-body contact Hessian, and lagged cross-body RHS are gathered directly into
the register-resident 12x12 solve. A device-to-device `x_lagged` snapshot keeps
strict Jacobi read-before-write semantics. PCG retains its existing
BlockCSR3/WideContact path.

On the 24-link active-contact probe, the ELL builder averaged about 1.01 us.
An initial separate body-gather kernel averaged 14.08 us and was rejected. The
fused contact-aware solve averaged about 10.93 us versus 7.77 us for the prior
contact-free solve kernel, eliminating the separate gather and its global
78-float body-matrix traffic. The fused solve uses 150 registers/thread with
zero stack and zero local memory.

### P1: Add A Dedicated Block-Jacobi Assembly Path

The current Block-Jacobi path inherits a PCG-oriented pipeline:

```text
clear BlockCSR3
assemble mass/elastic/fixed into BlockCSR3
bake contact diagonal into BlockCSR3
copy RHS
pack BlockCSR3 into 78 floats/body
scatter local contacts
scatter cross-body RHS
solve
```

For interpolated ABD bodies, four controls belong to one body and the body
worker can assemble directly. The target pipeline should be:

```text
predict controls
update surface positions
detect/filter contacts and write WideContact
build body-contact ELL
assemble-and-solve one local body system
write four controls
```

Keep the current `BlockCSR3 + WideContactSpMVOp` path unchanged for PCG. The
Block-Jacobi branch should not clear, write, read, and repack the sparse PCG
matrix.

The mass/elastic/fixed Hessian is constant while `dt`, density, stiffness,
fixed flags, and fixed weight remain unchanged. Define the scalar 4x4 base
matrix

```text
K_ab = M_ab / h^2
     + stiffness * volume * dot(grad_a, grad_b)
     + fixed(a) * fixed_weight * delta_ab

H_base = K tensor I3
```

Cache the ten lower-triangular values of `K`, or its 4x4 Cholesky factor, per
body. Rebuild the cache only when a relevant parameter changes. The local RHS
still changes every outer iteration because it contains predicted positions
and the current polar rotation.

This also removes unnecessary atomics from mass/elastic/fixed assembly for the
one-tet-per-body ABD representation. Add a checked fallback for any future
mesh in which controls are shared between logical bodies.

Implementation status (2026-07-10): completed for the current one-tet-per-body
ABD layout. `block_jacobi_base_k_` stores ten lower-triangular scalar entries
per body. A small rebuild kernel updates this cache only when `h`, stiffness,
or fixed weight changes. The fused solve now builds the dynamic mass,
corotated-elastic, and fixed RHS directly from `predicted`, `x_lagged`, and the
current polar rotation.

The Block-Jacobi branch no longer performs:

- `H_.set_zero()`;
- global mass/elastic/fixed BlockCSR3 assembly;
- global RHS clear and device-to-device RHS copy;
- 78-float body-Hessian clear;
- `pack_interpolated_body_hessian_kernel`.

PCG still uses the original BlockCSR3 assembly. Explicit `__fmul_rn` and
`__fadd_rn` operations reproduce the rounding boundaries previously imposed
by atomic accumulation. The one-pass 440-surface-vertex comparison against the
saved P1 executable then had zero exported-coordinate difference, and the
60/120/180-frame two-link and 24-link minY traces matched again.

On the 24-link profile, the P1 kernels for mass/elastic assembly, fixed
assembly, body packing, contact-aware solve, and ELL build totaled about
20.29 us/frame. P2 reduced this set to about 12.73 us/frame, a 37 percent
reduction. It also removed one device-to-device copy and four memsets per
frame. The final fused solve uses 157 registers/thread with zero stack and zero
local memory.

On the 32x32 zero-contact probe, the P2 fused solve averaged about 10.01 us and
ELL construction about 1.00 us. The cached base-K rebuild cost about 1.82 us
only on the first frame. The removed P1 base assembly and pack kernels had
previously cost about 15.97 us and 10.69 us respectively.

### Using Matrix Symmetry

The current global body buffer already stores the lower triangle of a generic
symmetric 12x12 matrix, reducing 144 floats to 78 floats per body. There are
two additional opportunities.

First, every 3x3 control-pair block in the current model is itself symmetric:

```text
mass/elastic/fixed: scalar * I3
normal contact:     scalar * (n n^T)
```

Therefore, if a global body matrix is still needed, it can be represented as
ten symmetric 3x3 blocks, each with six floats:

```text
10 control pairs * 6 floats = 60 floats/body
```

An equivalent layout stores six symmetric 4x4 matrices for `xx`, `xy`, `xz`,
`yy`, `yz`, and `zz`. This is 23 percent less storage than the current
78-float format. Guard this optimization with an assertion or format flag if a
future nonsymmetric term is introduced.

Second, the best memory optimization is not to store the dynamic body matrix
globally. A body-owned assembly/solve kernel can keep the lower triangle in
registers or shared memory, solve it, and write only four output controls. That
eliminates:

- clearing `body_hessian_12_`;
- writing 60/78 floats per body;
- reading them back in the solve kernel;
- copying `rhs_` into `block_jacobi_rhs_`;
- the separate sparse-to-body pack pass.

For each local contact side, define

```text
u = sqrt(k_contact) * (w tensor n),  u in R^12
H_contact = u * u^T
```

The body worker computes six unique values of `n n^T` once and performs a
rank-one symmetric update to its local block.

### Accelerating The 12x12 Solve

Do not form an explicit 12x12 inverse. The current Cholesky factorization plus
two triangular solves is the correct baseline for an SPD system. The matrix is
SPD/PSD by construction:

- inertia is positive;
- corotated PD stiffness is PSD;
- fixed and contact penalties are PSD rank updates.

The practical solve optimizations, in priority order, are:

1. **Zero-contact fast path.** If `body_contact_count == 0`, solve three
   independent 4x4 systems using the cached factor of `K`. Most bodies in a
   sparse-contact scene can take this path.
2. **Keep only the lower triangle live.** The current solver loads 78 floats
   but expands them into a full `A[12][12]`, producing 130 registers/thread.
   Test an in-place packed Cholesky and check `cuobjdump` after every version;
   dynamic packed indexing can accidentally cause local-memory spills.
3. **Use a cooperative tile per body.** A 16-thread tile is a natural mapping:
   lanes 0..11 own one matrix row/DOF and four lanes assist with contact loads.
   Each active lane holds at most 12 lower-triangular values. Tile shuffles can
   broadcast pivots during Cholesky and triangular solves. This should reduce
   per-thread registers and improve occupancy without global matrix traffic.
4. **Experimental low-rank path.** Contacts are rank-one updates to
   `K tensor I3`. For a very small number of contacts per body, test Woodbury
   or Cholesky rank-one updates from the cached base factor. Select this only
   after measuring the body contact-degree histogram. Around four or more
   contacts, repeated 12x12 rank-one updates may cost as much as one fresh
   factorization.
5. **Reuse reciprocal diagonals.** Store `1 / L_ii` during factorization and
   multiply in the factorization and triangular solves instead of repeating
   divisions. This is a small final-stage optimization.

LDLT without pivoting can remove square roots, but it is not automatically
faster or more stable on this GPU. Batched cuSOLVER and Tensor Core paths are
also poor first choices for a 12x12 FP32 contact solve: padding, pointer-array
traffic, launch overhead, and reduced numerical margin can exceed the saved
arithmetic.

If absolute-position solves retain diagonal regularization, use a proximal
reference consistently:

```text
(H + epsilon I) x = b + epsilon x_reference
```

Otherwise `epsilon I` silently attracts absolute coordinates toward the
origin. A residual solve `(H + epsilon I) dx = r` avoids that bias naturally.

### Recommended Fused Body Kernel

The target body worker should perform the following operations in one kernel:

1. Load cached base `K` or its factor.
2. Build the 12-vector mass/elastic/fixed RHS.
3. Loop over `body_contact_refs[offset[i]..offset[i+1])`.
4. Add target RHS and lagged cross-body RHS using the old Jacobi positions.
5. Add the local rank-one contact Hessian.
6. Solve the local system and apply `omega`.
7. Write the four owned controls; integrate fixed-control handling here.

Both sides may read the same 64-byte `WideContact`, but each body writes only
its own on-chip system and controls. This is preferable to duplicating a large
side-local contact payload. A compact 32-bit contact reference should be the
default; use a workspace arena or reuse dead narrow-phase buffers so the
`2 * contact_capacity` reference buffer does not permanently increase peak
memory more than necessary.

### Implementation Order

**P0: scheduling, no algorithm change**

- [x] convert the four Block-Jacobi count-driven kernels to bounded
  grid-stride loops;
- add NVTX ranges for contact conversion, body binding, assembly, and solve;
- expose valid wide contacts, active bodies, average/max contacts per body;
- keep an error on every capacity overflow.

**P1: remove Block-Jacobi contact scatter**

- [x] add body-contact ELL;
- [ ] statically bind ground samples to bodies;
- [x] make one body worker assemble local contact Hessian and effective RHS;
- [x] skip PCG diagonal baking in Block-Jacobi mode;
- [x] preserve the current PCG route exactly.

**P2: remove the PCG matrix from the Block-Jacobi route**

- [x] cache the scalar 4x4 base matrix;
- [x] assemble local RHS directly per body;
- [x] fuse body assembly and solve;
- [x] remove Block-Jacobi Hessian clear/copy/pack buffers after A/B agreement.

**P3: solve-kernel specialization**

- add zero-contact 4x4 fast path;
- compare one-thread packed Cholesky with 16-thread cooperative Cholesky;
- test low-rank updates only for measured low contact degree;
- capture the final multi-outer-iteration Block-Jacobi path in a CUDA Graph.

### Optimization Validation

Every phase must compare against the current corrected Block-Jacobi result,
not only against visual output. Record:

- per-kernel GPU time and total step GPU time;
- valid contacts versus reserved capacity;
- active body count and contact-degree histogram;
- registers, stack, local memory, and achieved occupancy;
- maximum per-control `abs(x_new - x_reference)` after one outer pass;
- minY, PF/EE counts, inversion failures, NaNs, and overflows;
- 180-frame `Vertical Link Chain 2` trace;
- 4x4 and 8x8 chain stability;
- 32x32 zero-contact and active-contact timing separately;
- 5,000-frame box/ground regression after the fused path is stable.

Suggested acceptance thresholds for a semantics-preserving stage:

```text
same contact IDs/counts before assembly
max control-position difference <= 1e-5 after one pass
no new NaN, inversion failure, or capacity overflow
```

Floating-point reordering from removing atomics can change the last bits. If
repeatability matters, keep body references sorted by contact ID within each
body and compare deterministic runs separately.

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

- contacts between two bodies use lagged cross-body Hessian action on the RHS;
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
- lagged cross-body contact terms on the local RHS;
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
