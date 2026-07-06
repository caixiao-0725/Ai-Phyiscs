# ChysX Rigid-IPC 实现 vs 原版 rigid-ipc 算法对比

## 1. 原版 rigid-ipc 算法流程

每个时间步 `dt` 内，求解以下优化问题：

$$
\min_q \; \frac{1}{\bar{m}} \left[ E_{\text{body}}(q) + \kappa \cdot B(q) \right]
$$

其中：
- $q \in \mathbb{R}^{6n}$，每个刚体 6 DOF（平移 $p \in \mathbb{R}^3$ + 轴角旋转 $\theta \in \mathbb{R}^3$）
- $E_{\text{body}}$ 是惯性能量（BDF1 时间积分）
- $B(q) = \sum_{c} b(d_c^2(q), \hat{d}^2)$ 是 IPC log-barrier 接触势能
- $\bar{m}$ 是自由 DOF 质量矩阵对角元的均值（归一化）
- $\kappa$ 是自适应 barrier 刚度

### Newton 迭代

```
for iter = 0, 1, 2, ...
    1. 组装梯度: g = ∇E_body/m̄ + (κ/m̄) · ∇B
    2. 组装 Hessian:
       H = H_body/m̄ + (κ/m̄) · H_barrier
       
       H_body 是 block-diagonal: 每个 body 一个 6×6 块
       H_barrier 是 SPARSE: 每个 contact (body_A, body_B) 贡献一个 12×12 local Hessian
       
       local Hessian 结构:
       ┌─────────┬─────────┐
       │ H[A,A]  │ H[A,B]  │   ← 6×6 blocks
       ├─────────┼─────────┤
       │ H[B,A]  │ H[B,B]  │
       └─────────┴─────────┘
       
       H[A,B] = Σ_verts coeff · (J_A^T · ∇d/∂x_a) · (J_B^T · ∇d/∂x_b)^T
       
       这些 off-diagonal blocks 提供了 body 间的接触耦合
       
    3. PSD projection: H_barrier_local = project_to_psd(H_barrier_local)
    4. 线性求解: SimplicialLDLT (Eigen 稀疏直接法)
       H · Δq = -g
       如果 LDLT 失败 → Tikhonov 正则化: H + λI, λ 倍增直到成功
    5. 收敛判断: max_vertex_speed(Δq) ≤ 0.01 × bbox_diagonal
    6. CCD: α_ccd = min ToI (防穿透)
    7. Line search: 回溯直到 f(q + αΔq) < f(q), α 从 α_ccd 开始每次减半
    8. 自适应 κ 更新
```

### 关键特征

- **完整稀疏 Hessian**: 包含 body 间 off-diagonal coupling
- **CPU 直接求解器**: Eigen SimplicialLDLT
- **双精度**: 全程 double

## 2. ChysX 当前实现（修改前的 block-diagonal 版本）

### 算法流程

```
for iter = 0, 1, 2, ...
    1. 碰撞检测 (GPU):
       - Broadphase: QuantBvh / OptiX → EF 候选对
       - Narrowphase: GPU kernel 将 EF 对分类为 PP/PE/PT/EE contacts
       - 补充相邻 VF/EE 检测
       
    2. Body energy 组装 (CPU):
       对每个 body i:
         g_body[i] = body_gradient(body_i)   // 6维
         H_body[i] = body_hessian(body_i)    // 6×6
       缩放: g /= m̄,  H /= m̄
       
    3. Barrier 梯度 + Hessian (GPU):
       kernel_assemble_contacts:
         对每个 contact c:
           计算 d² 和 ∇d/∂x (每个顶点)
           对每个顶点 k:
             J_k = RigidJacobi(x_bar_k)   // 3×6 Jacobian
             jt_g_k = J_k^T · ∇d/∂x_k    // 6维
             
             ★ grad[body_k] += -κ/m̄ · dB/dD · jt_g_k    (atomicAdd)
             ★ H_diag[body_k] += κ/m̄ · d²B/dD² · jt_g_k · jt_g_k^T  (atomicAdd 6×6)
             
       ↑ 注意: 只累加到 H_diag[body_k] (对角块)
         缺失: H_offdiag[body_A, body_B] (cross-body coupling)
       
    4. DOF clamping:
       对每个 locked DOF d: rhs[d]=0, H_diag[d,d]=1e10
       
    5. 线性求解 (CPU):
       PCG (block-diagonal preconditioner)
       每个 body 独立 6×6 块 → 实际上退化为 per-body 直接求解
       
    6. CCD (GPU): 计算最大安全步长
    7. Line search: 回溯
    8. 收敛判断 + 自适应 κ
```

### 本质区别

| | 原版 rigid-ipc | ChysX 当前实现 |
|---|---|---|
| **Hessian 结构** | 完整稀疏 (含 off-diagonal blocks) | **仅 block-diagonal** |
| **线性求解** | 稀疏 LDLT (CPU, double) | block-diagonal PCG (CPU, float) |
| **Contact coupling** | body 间完全耦合 | **body 间完全解耦** |
| **精度** | double | float (body_energy 用 double) |
| **碰撞检测** | CPU | GPU (QuantBvh + CUDA narrow phase) |
| **Contact 类型** | PP/PE/PT/EE | PP/PE/PT/EE |

### 为什么 block-diagonal 会卡住

chain 场景中 rail(body 0) 受重力下落，齿与 gear(body 1) 啮合。

**原版**: Newton 步同时考虑「rail 下移 → barrier 推 gear 旋转」的耦合。
Hessian 的 off-diagonal block `H[rail, gear]` 告诉求解器：
"rail 的 y 位移和 gear 的 rz 旋转是强耦合的"

**ChysX**: Newton 步只看每个 body 自己的对角块。
- rail 看到 barrier gradient 阻止它下移 → dq[rail] ≈ 0
- gear 看到的 barrier gradient 在 dof[5](rz) 上非常小（因为 barrier 主要是法向力，而齿轮旋转是切向的）→ dq[gear] ≈ 0
- 结果：**两个 body 都几乎不动**

这不是"创新"，这是一个 **bug** —— 缺失了 off-diagonal coupling blocks。

## 3. 正确做法

barrier Hessian 的完整组装需要：

对每个 contact 涉及的顶点对 (a, b)，其中 a ∈ body_A, b ∈ body_B:

$$
H[A,B] \mathrel{+}= \frac{\kappa}{\bar{m}} \cdot \frac{\partial^2 b}{\partial D^2} \cdot (J_A^T \nabla_a d) \cdot (J_B^T \nabla_b d)^T
$$

这个 6×6 block 需要添加到全局矩阵的 off-diagonal 位置。

### GPU 实现方案

原版在 CPU 用 Eigen 稀疏 LDLT。要在 GPU 上做，有几个选项：

**方案 A: GPU 组装 + cuSolver 稀疏 LDLT**
- 在 GPU 上组装完整稀疏矩阵（COO/CSR）
- 用 cuSolverSp 或 cuSparse 求解
- 适合 body 数较多的场景

**方案 B: GPU 组装 + cuSolver dense Cholesky**
- 对于小系统（≤几十个 body, ≤200 DOF），直接用 dense 矩阵
- 用 cuSolverDn 的 `potrf/potrs` (Cholesky)
- 简单且高效

**方案 C: GPU 组装 off-diagonal + GPU PCG**
- 在 GPU 上做 block-sparse SpMV
- 需要 PSD 保证或正则化
- 最 GPU-native 但实现复杂

当前 chain 场景 6 body = 36 DOF，方案 B 最简单。
