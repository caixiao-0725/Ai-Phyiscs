# ABD 增广拉格朗日框架：仿射矩阵下的 Dual Update 推导

## 1. 问题背景

在 Affine Body Dynamics (ABD) 中，刚体状态用 12 个自由度表示：

$$q = (c, A) \in \mathbb{R}^{12}, \quad c \in \mathbb{R}^3, \; A \in \mathbb{R}^{3 \times 3}$$

世界坐标中某 body-local 点 $r$ 的位置：

$$x(q) = c + A \cdot r$$

当前实现中，primal update 完全在仿射矩阵空间进行（12×12 Schur 系统），但 dual update 却转回了轴角表示来计算约束值。这在概念上不自洽——我们应该**直接在仿射空间定义约束函数并推导 dual update**。

---

## 2. 增广拉格朗日总框架

### 2.1 优化目标

每个时间步求解：

$$\min_{q} \; \frac{1}{2}(q - \tilde{q})^T M (q - \tilde{q}) + \sum_i \left[ \lambda_i \, C_i(q) + \frac{\kappa_i}{2} \, C_i(q)^2 \right]$$

其中：
- $\tilde{q}$ 为 symplectic Euler 惯性预测
- $C_i(q)$ 为第 $i$ 个约束函数的约束值
- $\lambda_i$ 为对偶变量（Lagrange 乘子）
- $\kappa_i$ 为罚参数（penalty）

### 2.2 求解流程

```
for iter = 0 to N:
    1. Primal update: 固定 (λ, κ), 对 q 最小化增广拉格朗日量
    2. Dual update:   固定 q, 更新 (λ, κ)
```

**关键**：primal update 求解的是**完整能量**（惯性 + penalty + 乘子项），而 dual update 只需要**评估当前约束值**并更新乘子。

---

## 3. 仿射空间中的约束定义

### 3.1 地面碰撞约束

刚体上 body-local 点 $r_v$ 的世界坐标为 $x_v = c + A r_v$。地面法线为 $n = (0,0,1)^T$，地面高度为 $z_g$。

**约束函数**（穿透深度，负值表示穿透）：

$$C_v^{\text{gnd}}(q) = n^T (c + A r_v) - z_g = n^T x_v - z_g$$

这是一个标量约束。注意它**直接定义在仿射坐标 $(c, A)$ 上**，不需要任何轴角表示。

约束条件为 $C_v^{\text{gnd}} \geq 0$（单侧不等式约束）。

### 3.2 刚体-刚体碰撞约束

设两个刚体 $I, J$，碰撞对的接触点为：

$$x_I = c_I + A_I \, r_I, \quad x_J = c_J + A_J \, r_J$$

碰撞法线为 $n$（从 $J$ 指向 $I$），接触基底为 $\{n, t_1, t_2\}$。

**约束函数**（3D，包含法线和切向分量）：

$$\mathbf{C}^{\text{box}}(q_I, q_J) = B^T (x_I - x_J) + d_0$$

其中 $B = [n \; t_1 \; t_2]^T \in \mathbb{R}^{3 \times 3}$ 为接触基底矩阵，$d_0 = (\delta_{\text{margin}}, 0, 0)^T$。

展开：

$$C_d = e_d^T \left[ B^T \big( (c_I + A_I r_I) - (c_J + A_J r_J) \big) + d_0 \right], \quad d \in \{0, 1, 2\}$$

同样，**完全用仿射矩阵 $A$ 表达**，无需轴角转换。

---

## 4. Primal Update（回顾）

primal update 已在 `PABD_12x12_Schur_Derivation.md` 中详细推导。简要回顾核心结构：

### 4.1 局部帧

在惯性预测 $\tilde{q} = (\tilde{c}, \tilde{A})$ 的参考帧中求解：

$$c = \tilde{c} + \tilde{A} c_L, \quad A = \tilde{A} A_L$$

### 4.2 约束对 Hessian 的贡献

对单个约束 $C_i$ 以及 penalty $\kappa_i$，增广拉格朗日目标中的二次罚项为：

$$\frac{\kappa_i}{2} C_i^2 = \frac{\kappa_i}{2} \|n^T(c + A r) - z_g\|^2$$

Gauss-Newton 近似的 Hessian 对系统的贡献为 $\kappa_i J_i^T J_i$，与纯罚函数形式相同。

乘子项 $\lambda_i C_i$ 是线性的，只影响 RHS（梯度），不影响 Hessian。

### 4.3 约束对 RHS 的贡献

在局部帧中，约束对 RHS 的贡献分为两部分：

**罚函数部分**（将点推向目标位置）：

$$\text{src}_c \mathrel{+}= \kappa_i \cdot x_i^L, \quad \text{src}_A \mathrel{+}= \kappa_i \cdot (x_i^L \otimes r_i)$$

**乘子部分**（额外的恢复力）：

$$\text{src}_c \mathrel{+}= -\lambda_i \cdot n_i^L, \quad \text{src}_A \mathrel{+}= -\lambda_i \cdot (n_i^L \otimes r_i)$$

其中 $n_i^L = \tilde{A}^T n_i$ 为约束法线在局部帧中的表示，$x_i^L = \tilde{A}^T(x_i^{\text{target}} - \tilde{c})$。

---

## 5. Dual Update 推导（核心部分）

### 5.1 标准增广拉格朗日 Dual Update 公式

增广拉格朗日方法的 dual update 规则为：

$$\lambda_i^{(k+1)} = \lambda_i^{(k)} + \kappa_i \, C_i(q^{(k+1)})$$

其中 $q^{(k+1)}$ 是 primal update 之后的新状态。

这是增广拉格朗日方法的标准形式。直觉上：如果 primal update 之后约束仍然违反（$C_i \neq 0$），乘子累积一个正比于违反量的修正，使得下一次 primal update 时对该约束施加更强的力。

### 5.2 地面碰撞的 Dual Update

primal update 完成后，直接在仿射坐标中评估约束值：

$$C_v = n^T (c^{(k+1)} + A^{(k+1)} r_v) - z_g$$

其中 $c^{(k+1)}$ 和 $A^{(k+1)}$ 是 primal update 后的新值（经过极分解投影）。

注意这里**不需要任何轴角或四元数转换**。约束值 $C_v$ 就是世界坐标中该点的 z 坐标减去地面高度。

**乘子更新**：

$$F_v = \kappa_v \, C_v + \lambda_v$$

对于单侧接触约束（只抵抗穿透，不抵抗分离）：

$$\lambda_v^{(k+1)} = \min(F_v, \; 0)$$

$\min(\cdot, 0)$ 投影确保乘子始终 $\leq 0$，即约束力只能是**压缩力**（推离地面），不能是拉力。

**罚参数更新**（自适应）：

$$\kappa_v^{(k+1)} = \begin{cases} \min(\kappa_v + \beta |C_v|, \; \kappa_{\max}) & \text{if } F_v < 0 \\ \kappa_v & \text{otherwise} \end{cases}$$

当约束仍活跃（压缩力 $F_v < 0$）时，增大罚参数以加速收敛。

### 5.3 刚体-刚体碰撞的 Dual Update

primal update 后，评估 3D 约束值：

$$\text{diff} = (c_I^{(k+1)} + A_I^{(k+1)} r_I) - (c_J^{(k+1)} + A_J^{(k+1)} r_J)$$

$$C_d = e_d^T (B^T \text{diff}) + d_{0,d}, \quad d \in \{0, 1, 2\}$$

同样，完全在仿射空间计算，无需转换。

**3D 乘子更新**：

$$F_d = \kappa_d \, C_d + \lambda_d, \quad d \in \{0, 1, 2\}$$

三个分量的投影规则不同：

**法线分量** ($d = 0$)——单侧接触：

$$F_0 \leftarrow \min(F_0, \; 0)$$

**切向分量** ($d = 1, 2$)——Coulomb 摩擦锥：

$$\text{tangForce} = \sqrt{F_1^2 + F_2^2}$$

$$\text{bound} = \mu |F_0|$$

$$\text{if } \text{tangForce} > \text{bound}: \quad F_1 \leftarrow F_1 \cdot \frac{\text{bound}}{\text{tangForce}}, \quad F_2 \leftarrow F_2 \cdot \frac{\text{bound}}{\text{tangForce}}$$

**罚参数更新**：

$$\kappa_0^{(k+1)} = \begin{cases} \min(\kappa_0 + \beta |C_0|, \; \kappa_{\max}) & \text{if } F_0 < 0 \text{ (active normal)} \\ \kappa_0 & \text{otherwise} \end{cases}$$

$$\kappa_d^{(k+1)} = \begin{cases} \min(\kappa_d + \beta |C_d|, \; \kappa_{\max}) & \text{if } \text{tangForce} \leq \text{bound} \text{ (static friction)} \\ \kappa_d & \text{otherwise (sliding)} \end{cases}, \quad d \in \{1, 2\}$$

---

## 6. 为什么不应该转回轴角？

### 6.1 概念一致性

整个 ABD 框架选择仿射矩阵 $A \in \mathbb{R}^{3\times 3}$ 作为广义坐标，primal update 在 12D 空间求解。如果 dual update 把 $A$ 转回轴角 $\theta \in \mathbb{R}^3$ 来计算约束，就引入了坐标系混淆：

- Primal 在 $(c, A)$ 空间走了一步
- Dual 却在 $(c, \theta)$ 空间评估约束

约束函数 $C$ 应该对**同一组广义坐标**评估。

### 6.2 数学上的必要性

仔细看约束函数：

$$C_v = n^T (c + A r_v) - z_g$$

这是 $(c, A)$ 的**线性函数**！其值完全由 $c$ 和 $A$ 确定，根本不需要知道"对应的轴角是多少"。

对于 box-box 约束：

$$\text{diff} = (c_I + A_I r_I) - (c_J + A_J r_J)$$

同样是仿射坐标的线性组合。

### 6.3 精度和效率

转回轴角需要：
1. 从旋转矩阵提取轴角：`mat_to_angular(A_new, A_old)` —— 涉及 $\arccos$、除法、奇异性处理
2. 用轴角构造旋转矩阵来旋转杠臂：`rotateVec(r, AxisAngle)` —— 又需要 Rodrigues 公式

这既增加了计算量，又引入了数值误差（$\arccos$ 在接近 0 和 $\pi$ 时不精确）。而直接用 $A$ 计算则是简单的矩阵-向量乘法。

### 6.4 极分解后的 A 已经是旋转矩阵

primal update 的最后一步是极分解投影：

$$A^{(k+1)} = \text{polar}(\tilde{A} \cdot A_L)$$

此时 $A^{(k+1)} \in SO(3)$ 已经是一个合法的旋转矩阵，直接用它计算 $x = c + A r$ 就得到了正确的世界坐标。

---

## 7. 完整的单次迭代流程

```
for iter = 0 to N_iter:

    // ---- 碰撞检测 (可选: 每次迭代内重检) ----
    detect_contacts(bodies, ground, pairs)

    // ---- Primal Update ----
    for each body i:
        // 1. 构建局部帧
        c̃ = body.inertialLin    // symplectic Euler 预测质心
        Ã = body.inertialAff    // symplectic Euler 预测旋转
        ÃᵀT = transpose(Ã)

        // 2. 累积 Hessian (AffineSchurSolver)
        schur.reset()
        for each ground contact gc touching body i:
            schur.accumulate(κ_gc, r_gc)
        for each box contact bc touching body i:
            schur.accumulate(κ_avg, r_local)
        schur.factor(mass, I₀)

        // 3. 构建 RHS
        srcC = 0,  srcA = I₀

        // Ground contacts:
        for each gc:
            x_target = project_to_ground(c + A·r)
            x_L = ÃᵀT · (x_target - c̃)
            n_L = ÃᵀT · n
            srcC += κ · x_L       // penalty
            srcA += κ · (x_L ⊗ r) // penalty
            srcC += (-λ) · n_L     // 乘子
            srcA += (-λ) · (n_L ⊗ r) // 乘子

        // Box contacts:
        for each bc:
            x_target = compute_target(other_body, normal, margin)
            x_L = ÃᵀT · (x_target - c̃)
            srcC += κ_avg · x_L
            srcA += κ_avg · (x_L ⊗ r)
            for d = 0,1,2:
                n_d_L = ÃᵀT · basis[d]
                srcC += (-λ_d · sign) · n_d_L
                srcA += (-λ_d · sign) · (n_d_L ⊗ r)

        // 4. 求解
        (c_L, A_L) = schur.solve_update(srcC, srcA)

        // 5. 全局坐标恢复
        c = c̃ + Ã · c_L
        A = polar(Ã · A_L)       // 极分解投影到 SO(3)

    // ---- Dual Update ----

    // 地面碰撞
    for each gc:
        x_world = gc.body.c + gc.body.A · gc.r     ← 直接用仿射矩阵!
        C = x_world.z - z_ground                    ← 标量约束值

        F = κ · C + λ
        F = min(F, 0)        // 单侧投影
        λ ← F

        if F < 0:
            κ ← min(κ + β|C|, κ_max)

    // 刚体-刚体碰撞
    for each bc:
        diff = (bodyA.c + bodyA.A · rA) - (bodyB.c + bodyB.A · rB)  ← 直接用仿射矩阵!
        C = Bᵀ · diff + d₀                          ← 3D 约束值

        F_d = κ_d · C_d + λ_d,  d ∈ {0,1,2}
        F₀ = min(F₀, 0)                             // 法线: 单侧
        (F₁, F₂) = Coulomb_project(F₀, F₁, F₂, μ)  // 切向: 摩擦锥
        λ ← F

        if F₀ < 0:  κ₀ ← min(κ₀ + β|C₀|, κ_max)
        if static:   κ_d ← min(κ_d + β|C_d|, κ_max), d=1,2

    // 其他约束 (joints, springs)
    for each force: force.updateDual(α)
```

---

## 8. 对比：仿射 vs 轴角 Dual Update

| | 轴角 Dual Update（旧） | 仿射 Dual Update（正确） |
|---|---|---|
| **约束值计算** | $C = J \cdot \Delta\theta + \ldots$ (需 `mat_to_angular`) | $C = n^T(c + Ar) - z_g$ (直接矩阵乘) |
| **杠臂旋转** | `rotateVec(r, axisAngle)` (Rodrigues) | $A \cdot r$ (矩阵乘) |
| **奇异性** | $\theta \to 0$ 或 $\theta \to \pi$ 时 `mat_to_angular` 不精确 | 无奇异性 |
| **概念一致性** | Primal 在 $(c,A)$, Dual 在 $(c,\theta)$ — 混合 | 全程在 $(c,A)$ 空间 |
| **计算量** | 需要矩阵→轴角→旋转向量→约束值 | 一次矩阵-向量乘法 |
| **Jacobian** | $J_\theta = [n]_\times r$ (3×3 叉积) | 不需要 Jacobian, 直接求值 |

---

## 9. 代码对应

当前 `avbd_solver.cpp` 中的 dual update 实际上**已经是仿射空间的做法**：

```cpp
// Ground contact dual update
float3 worldPos = gc.body->positionLin + gc.body->affine * gc.rLocal;  // x = c + A·r
float d = worldPos.z - ground_z;                                       // C = n^T·x - z_g

float F = gc.penalty * d + gc.lambda;   // F = κ·C + λ
F = std::min(F, 0.0f);                  // 单侧投影
gc.lambda = F;

// Box-box contact dual update
float3 diff = bc.bodyA->positionLin + bc.bodyA->affine * bc.rA
            - bc.bodyB->positionLin - bc.bodyB->affine * bc.rB;  // 直接用 A
float3 C = bc.basis * diff + float3{AVBD_COLLISION_MARGIN, 0, 0};  // B^T·diff + d₀
```

这段代码是正确的仿射空间做法。需要修正的是 `Manifold::updateDual()` 中的轴角转换路径——即 AVBD 原始 Manifold 约束在 Affine 模式下不应调用 `mat_to_angular`。

---

## 10. 增广拉格朗日收敛性

### 10.1 乘子的物理意义

$\lambda_i$ 在物理上等价于**约束力的时间积分**。在每次迭代中：

- 如果 primal 没有完全消除穿透 ($C \neq 0$)，乘子累积一个额外的力
- 经过多次迭代，$\lambda$ 收敛到精确的 Lagrange 乘子，此时 $C \to 0$

### 10.2 帧间的 warm start

跨帧时，上一帧的 $(\lambda, \kappa)$ 提供了很好的初始估计：

$$\lambda^{(0)}_{\text{new frame}} = \alpha \gamma \cdot \lambda^{\text{end}}_{\text{prev frame}}$$

$$\kappa^{(0)}_{\text{new frame}} = \text{clamp}(\gamma \cdot \kappa^{\text{end}}_{\text{prev frame}}, \; \kappa_{\min}, \; \kappa_{\max})$$

其中 $\alpha$ 为约束保留率，$\gamma$ 为罚参数衰减/增长因子。这使得帧间约束变化平滑，减少"弹跳"。

### 10.3 与 PABD 的关系

PABD 原文使用 `positivePart(d)` 和 `negativePart(d)` 函数：
- `positivePart(d)` $\leftrightarrow$ 我们的 $\kappa$（Hessian 贡献系数）
- `negativePart(d)` $\leftrightarrow$ 我们的 $-\lambda$（RHS 恢复力）

PABD 将两者耦合在能量函数的导数中（如 barrier 函数的正部和负部），而我们显式分离了 $\kappa$（penalty）和 $\lambda$（乘子），两者在数学上等价，但显式形式更便于理解和调参。

---

## 11. 总结

- ABD 的 dual update **应当直接用仿射矩阵** $A$ 计算约束值，即 $x = c + Ar$
- 约束函数 $C(q) = n^T(c + Ar) - z_g$ 对仿射坐标 $(c, A)$ 是线性的，无需任何坐标转换
- 当前 `stepCpuAffine()` 中地面和 box-box 的 dual update 已经是正确的仿射做法
- `Manifold::updateDual()` 中的 `mat_to_angular` 路径是为兼容轴角模式保留的，ABD 模式不应使用它
- 增广拉格朗日的收敛性不依赖旋转表示——只要约束值 $C$ 计算正确，$\lambda$ 更新正确，就能收敛
