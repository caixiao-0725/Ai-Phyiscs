# AVBD vs PABD 旋转表示对比报告

## 1. 算法概述

| 维度 | AVBD (Axis-angle Velocity-driven Block Descent) | PABD (Projective Affine Body Dynamics) |
|------|------------------------------------------------|----------------------------------------|
| 旋转存储 | 四元数 `quat` (4 float) | 3×3 仿射矩阵 `Matrix` (9 float) |
| 旋转增量 | 3D 轴角向量 `δθ` | 3×3 矩阵增量 `δA` |
| 广义坐标 | `q = (x, θ)` ∈ ℝ⁶ | `q = (c, A)` ∈ ℝ¹² |
| 求解器 | 6×6 LDLᵀ 逐体块下降 | 12×12 Schur 补分解全局求解 |
| 约束方法 | 增广拉格朗日 (自适应 penalty + dual λ) | 带 barrier 的投影位置求解 |

---

## 2. 旋转表示详解

### 2.1 AVBD: 轴角 + 四元数

AVBD 使用四元数存储旋转状态，但所有增量和差异运算在 **轴角切空间** 进行：

```cpp
// 四元数差 → 轴角误差 (avbd_maths.h:122-123)
float3 operator-(quat a, quat b) {
    return (a * inverse(b)).vec() * 2.0f;
}

// 四元数 + 轴角增量
quat operator+(quat a, float3 b) {
    return normalize(a + quat{b.x, b.y, b.z, 0} * a * 0.5f);
}
```

数学表达：
- **差异**：$\delta\boldsymbol{\theta} = 2\,\mathrm{vec}(q_a q_b^{-1})$（小角度近似下的轴角向量）
- **更新**：$q_{\text{new}} = \mathrm{normalize}(q + \frac{1}{2}[\delta\boldsymbol{\theta}, 0] \cdot q)$

关键特性：
- 旋转自由度为 **3 维**（紧凑）
- 自然保证旋转约束（四元数归一化）
- 奇异性只在 $\|\delta\theta\| = 2\pi$ 附近

### 2.2 PABD: 3×3 仿射矩阵

PABD 使用 3×3 矩阵 $A$ 直接表示旋转，刚体约束由 **极分解投影** 保证：

```cpp
// 刚体世界坐标 (PABDConstraintSolver.cu:86)
Coord x0 = qPartAGlobal[id0] + qPartBGlobal[id0] * r0;
//    = center + A * r_local

// 极分解投影 (PABDConstraintSolver.cu:1077)
polarDecomposition(mat, R, Real(0.00001));
rotLocal[pId] = R;
```

数学表达：
- **世界坐标**：$\mathbf{x} = \mathbf{c} + A \cdot \mathbf{r}_{local}$
- **仿射更新后投影**：$A \leftarrow \text{polar}(A_{update})$，提取旋转分量 $R$

关键特性：
- 旋转自由度为 **9 维**（冗余但无奇异性）
- 需要极分解强制正交性
- 矩阵-向量乘法直接可用，无需四元数旋转公式

---

## 3. 时间积分对比

### 3.1 AVBD 时间积分

#### 预测步 (Predict)

```
x̃ = x + v·Δt + g·Δt²         (平移)
q̃ = q ⊕ ω·Δt                 (旋转，轴角增量)
```

代码位置：`avbd_gpu_solver.cu:190-219`

```cpp
// 惯性目标
V3 inertLin = posLin + velLin * dt;
if (m > 0.0f) inertLin = inertLin + v3(0, 0, gravity) * (dt * dt);
Q4 inertAng = qaddv(posAng, velAng * dt);

// 预测位姿
V3 newPos = posLin + velLin * dt + v3(0, 0, gravity) * (accelWeight * dt * dt);
Q4 newAng = qaddv(posAng, velAng * dt);
```

#### 速度恢复 (BDF1)

```
v^{n+1} = (x^{n+1} - x^n) / Δt           (平移)
ω^{n+1} = 2·vec(q^{n+1}·(q^n)^{-1}) / Δt  (旋转)
```

代码位置：`avbd_gpu_solver.cu:500-510`

### 3.2 PABD 时间积分

#### 预测步

```
c̃ = c + v·Δt                   (平移，重力在速度层处理)
q̃ = q ⊕ 0.5·Δt·[ω,0]·q        (四元数用于生成初始A)
Ã = q̃.toMatrix3x3()            (转为矩阵)
```

代码位置：`PABDConstraintSolver.cu:702-713`

```cpp
center[pId] += velocity[pId] * dt;

Matrix inertia = affine[pId] * initialInertia[pId] * affine[pId].inverse();
quaternion[pId] += 0.5f * dt * (Quat(ω.x, ω.y, ω.z, 0.0) * quaternion[pId]);
quaternion[pId] = quaternion[pId].normalize();
affine[pId] = quaternion[pId].toMatrix3x3();
```

#### 速度恢复

PABD 从旋转矩阵差中提取角速度（方案1，反对称部分）：

```
dR = (R_new - R_old) · R_old^T
ω = 0.5 · [dR₃₂ - dR₂₃, dR₁₃ - dR₃₁, dR₂₁ - dR₁₂] / Δt
```

代码位置：`PABDConstraintSolver.cu:746-749`

```cpp
Matrix R_n = affine[pId];
Matrix R_new = qPartBGlobal[pId];
Matrix dR = (R_new - R_n) * R_n.transpose();
Coord dw = 0.5 * Coord(dR(2,1) - dR(1,2), dR(0,2) - dR(2,0), dR(1,0) - dR(0,1)) / dt;
```

### 3.3 核心差异

| | AVBD | PABD |
|---|------|------|
| 预测自由度 | 6D (3 平移 + 3 轴角) | 12D (3 平移 + 9 矩阵) |
| 重力处理 | 带加速度权重的位置预测 | 速度层 substepping |
| 角速度恢复 | 四元数差 → 轴角 | 矩阵差 → 反对称提取 |
| 旋转积分 | 指数映射近似 | 四元数积分 → 转矩阵 |

---

## 4. 质量计算对比

### 4.1 AVBD

长方体质量（`avbd_rigid.cpp:18`）：

$$m = s_x \cdot s_y \cdot s_z \cdot \rho$$

存储为标量 `Rigid::mass`。静态物体 `mass ≤ 0`，动态物体 `mass > 0`。

### 4.2 PABD

通过外部 `RigidBodySystem` 计算，存储于 `DArray<Real> Mass`。计算方式类似，取决于形状类型（球、盒、胶囊等）。

### 4.3 差异

两者在平移质量的计算和使用上基本一致。差异在于 PABD 的 12D 广义质量矩阵中，平移部分和旋转部分通过 Schur 补耦合，而 AVBD 的 6×6 Hessian 也包含耦合项 `lhsCross`。

---

## 5. 转动惯量对比

### 5.1 AVBD

对角惯性张量（体坐标系，`avbd_rigid.cpp:19-22`）：

$$I_{xx} = \frac{m}{12}(s_y^2 + s_z^2), \quad I_{yy} = \frac{m}{12}(s_x^2 + s_z^2), \quad I_{zz} = \frac{m}{12}(s_x^2 + s_y^2)$$

存储为 `float3 moment = (I_{xx}, I_{yy}, I_{zz})`。

**关键简化**：惯性张量**不旋转**到世界坐标系。即使用 $\text{diag}(I_{xx}, I_{yy}, I_{zz})$ 直接作为世界坐标系惯性，忽略 $R I R^T$。

```cpp
// avbd_gpu_solver.cu:296-300
M33 MAng = m33diag(mom_x[body], mom_y[body], mom_z[body]);
M33 lhsAng = m33scale(MAng, invDt2);  // I / Δt²
```

### 5.2 PABD

初始惯性张量 `InitialInertia` 为 3×3 矩阵（可以是非对角的）。

在 PABD 框架中，惯性以 **Frobenius 范数** 形式出现在优化目标中：

$$E_{\text{inertia}} = \frac{1}{2} \|A - A^*\|_{I_0}^2$$

其中 $I_0$ 是初始惯性矩阵，$A^*$ 是仿射预测值。

```cpp
// PABDConstraintSolver.cu:649-653
Matrix I = initialI[tId];
// Schur 补中使用
Matrix SPart = (I + sumD[tId]) / k_i - dyadic(C, B);
```

仿射矩阵自然支持世界坐标系的惯性变换：

$$I_{world} = A \cdot I_0 \cdot A^{-1}$$

### 5.3 核心差异

| | AVBD | PABD |
|---|------|------|
| 惯性存储 | 对角 `float3` | 完整 `Matrix 3×3` |
| 世界坐标系转换 | **忽略**（近似） | **精确** $AI_0A^{-1}$ |
| Hessian 中的角色 | $I/\Delta t^2$ 对角块 | Schur 补矩阵 $(I + \sum D) / k$ |

---

## 6. 碰撞 Hessian 对比

### 6.1 AVBD 碰撞 Hessian

#### 6×6 块结构

$$\mathbf{H} = \begin{bmatrix} H_{ll} & H_{lc}^T \\ H_{lc} & H_{aa} \end{bmatrix}$$

#### 惯性贡献

$$H_{ll}^{inertial} = \frac{m}{\Delta t^2} \mathbf{I}_{3\times3}, \quad H_{aa}^{inertial} = \frac{I}{\Delta t^2}$$

#### 碰撞约束 Jacobian

接触坐标系 `basis` = $[n, t_1, t_2]$（法线 + 两切线）：

$$J_{A,lin} = B, \quad J_{B,lin} = -B$$
$$J_{A,ang}[k] = \mathbf{r}_A^{world} \times B[k], \quad J_{B,ang}[k] = \mathbf{r}_B^{world} \times B[k]$$

其中力臂旋转到世界坐标系：
$$\mathbf{r}^{world} = \text{qrot}(q, \mathbf{r}^{local})$$

#### Hessian 累积 (Gauss-Newton)

$$H_{ll} \mathrel{+}= J_l^T K J_l, \quad H_{aa} \mathrel{+}= J_a^T K J_a, \quad H_{lc} \mathrel{+}= J_a^T K J_l$$

代码位置：`avbd_manifold.cpp:102-104`

```cpp
lhsLin += jLinT * K * jLin;
lhsAng += jAngTk * jAng;
lhsCross += jAngTk * jLin;
```

#### RHS (右端项)

$$\mathbf{rhs}_l = J_l^T F, \quad \mathbf{rhs}_a = J_a^T F$$

其中 $F = KC + \lambda$，经过法线方向单侧约束和库仑摩擦锥投影。

### 6.2 PABD 碰撞 Hessian

#### 12×12 块结构（通过 Schur 补分解为 3+9）

PABD 的广义坐标为 $q = (c, A)$，其中 $c \in \mathbb{R}^3$，$A \in \mathbb{R}^{3\times3}$。

约束值为标量距离：$d = (x_1 - x_0) \cdot n + 2 d_{hat}$

#### 第一类求和（用于构建 $M^{-1}$ 的 Schur 补）

对每个约束 $c$，计算权重 $\alpha_c = h^2 \cdot B'(d)^+/d$：

$$\text{sum0} \mathrel{+}= \alpha, \quad \text{sumB} \mathrel{+}= \alpha \cdot r, \quad \text{sumD} \mathrel{+}= \alpha \cdot r \otimes r$$

#### Schur 补逆矩阵

$$k = m + \sum \alpha_c$$

$$S = \frac{I_0 + \text{sumD}}{k} - C \otimes B$$

$$M^{-1} = \begin{bmatrix} a & b^T \\ c & D \end{bmatrix}, \quad a = \frac{1 + B^T S^{-1} C}{k}, \quad D = \frac{S^{-1}}{k}$$

代码位置：`PABDConstraintSolver.cu:267-327`

```cpp
Real k_i = m_i + alpha_i;
Coord B = sumB[tId] / k_i;
Coord C = sumC[tId] / k_i;
Matrix SPart = (I + sumD[tId]) / k_i - dyadic(C, B);
Matrix SPartInv = (SPart * k_i).inverse() * k_i;
mat1PartD[tId] = SPartInv / k_i;
mat1PartC[tId] = -SPartInv * C / k_i;
mat1PartB[tId] = -SPartInv.transpose() * B / k_i;
mat1PartA[tId] = (1 + B.dot(SPartInv * C)) / k_i;
```

#### 第二类求和（RHS 力项）

$$\text{sum3} \mathrel{+}= \beta \cdot n^{local}, \quad \text{sum4} \mathrel{+}= \alpha \cdot x_j^{local}$$

$$\text{sum5} \mathrel{+}= \beta \cdot n^{local} \otimes r, \quad \text{sum6} \mathrel{+}= \alpha \cdot x_j^{local} \otimes r$$

最终的 RHS 力向量：

$$\text{src}_c = m \cdot 0 + \text{sum3} + \text{sum4}$$

$$\text{src}_A = I_0 \cdot \mathbf{I} + \text{sum5} + \text{sum6}$$

#### 位移计算

$$\Delta c = a \cdot \text{src}_c + \text{src}_A \cdot b$$

$$\Delta A = \text{src}_c \otimes c + \text{src}_A \cdot D^T$$

### 6.3 核心差异

| | AVBD | PABD |
|---|------|------|
| Hessian 维度 | 6×6 (3 平移 + 3 旋转) | 12×12 (3 平移 + 9 矩阵), Schur 补分解 |
| 旋转 Jacobian | $J_a = r^{world} \times B$ (叉积) | $J_A = r^{local} \otimes n^{local}$ (张量积) |
| 求解方式 | 逐体 LDLᵀ 分解 | 全局原子累加 + Schur 补矩阵逆 |
| 约束函数 | 线性约束 + 自适应 penalty | Barrier 函数 $B(d)$ (多种可选) |
| 接触坐标系 | 法线+切线 basis 变换 | 法线投影到局部坐标系 |
| 摩擦处理 | 力投影到库仑锥 | 额外摩擦约束求和 |
| 线搜索 | 无 | 有 (能量下降步长限制) |

---

## 7. 碰撞响应处理对比

### 7.1 AVBD 碰撞响应

#### 流水线

```
BroadphaseGPU::query_gpu()     → 粗筛 (AABB + BVH)
NarrowphaseGPU::query_gpu()    → SAT 精筛 → GpuManifold + GpuContact
NarrowphaseGPU::warmstart_gpu() → 跨帧 λ/penalty 传递
GraphColoringGPU::color_jp()   → 图着色 (JP/Luby/Vivace/LDF)
GpuSolver::solve()             → 着色 Gauss-Seidel
```

#### 约束公式

$$C = C_0(1-\alpha) + J_A \Delta q_A + J_B \Delta q_B$$

$$F = KC + \lambda$$

法线：$F_n = \min(F_n, 0)$（仅压缩）

摩擦：$|F_t| \leq \mu |F_n|$（库仑锥投影）

#### Dual 更新 (增广拉格朗日)

$$\lambda \leftarrow F$$

$$k_n \leftarrow \min(k_n + \beta |C_n|, k_{max}) \quad \text{if } F_n < 0$$

粘滞检测：$|C_t| < 10^{-5}$

### 7.2 PABD 碰撞响应

#### 流水线

```
外部碰撞检测                    → ContactPair
PABDS2_UpdateContactConstraints → PABDConstraint (局部坐标系)
位置求解器 (global iteration)   → 仿射矩阵投影
速度求解器 (Jacobi iteration)   → 脉冲修正
```

#### 约束公式

距离约束：$d = \|x_1 - x_0\|$

接触约束：$d = (x_1 - x_0) \cdot (A \cdot n_0) + 2d_{hat}$

其中 $x_i = c_i + A_i \cdot r_i^{local}$

#### Barrier 能量

$$B(d) = k \left[(1-\gamma) + \frac{1}{2}(1-\gamma)^2 + \frac{1}{3}(1-\gamma)^3\right], \quad \gamma = d/d_{hat}$$

梯度分解为正部和负部：$B'(d) = B'^+(d) + B'^-(d)$

正部用于构建 Hessian 近似（保证半正定），负部用于力项。

#### 静摩擦

```cpp
Coord dP_t = x10 - (x10.dot(normal_global) * normal_global);  // 切向位移
Real d = dP_t.norm();
```

作为独立约束求解，与法线约束分开。

### 7.3 核心差异

| | AVBD | PABD |
|---|------|------|
| 碰撞检测 | 内置 GPU SAT + BVH | 外部模块提供 |
| 接触几何 | 8 点 manifold + feature key | 逐对 contact pair |
| 约束类型 | 增广拉格朗日 (penalty + dual) | Barrier 投影 |
| 迭代方式 | 着色 Gauss-Seidel (串行颜色) | 全局 Jacobi (全并行) |
| 温启动 | feature key 跨帧匹配 | 无显式温启动 |
| penalty 自适应 | $k \leftarrow k + \beta|C|$ | $k \leftarrow k + 1000|d - d_0|$（仅 distance） |
| 摩擦模型 | Coulomb 锥投影 (力层) | 静摩擦约束 + 动摩擦系数 |
| 速度修正 | BDF1 位置差 | 双层：位置求解 + 速度 Jacobi |
| 极分解 | 不需要 | 每次仿射更新后需要 |

---

## 8. 在 AVBD 中集成仿射矩阵旋转 — 已实现

### 8.1 设计原则

1. **运行时切换**：通过 `Solver::rotation_mode` 枚举在运行时选择旋转表示
2. **最小侵入**：仅替换旋转相关计算，保留碰撞检测、warm-start、graph coloring、penalty 等基础设施
3. **统一接口**：`Force::updatePrimal/updateDual` 接口不变，内部根据模式分支
4. **双重存储**：`Rigid` 同时持有 `quat positionAng` 和 `float3x3 affine`，通过 `syncFromQuat()`/`syncFromAffine()` 保持一致

### 8.2 使用方法

```cpp
Solver solver;
solver.rotation_mode = RotationMode::Affine;  // 使用仿射矩阵
// 或
solver.rotation_mode = RotationMode::AxisAngle;  // 默认，使用轴角
```

### 8.3 已修改的文件

| 文件 | 修改内容 |
|------|---------|
| `avbd_maths.h` | 新增 `RotationMode` 枚举、极分解 `polar_rotation()`、矩阵逆 `inverse3x3()`、`mat_to_angular()`、`mat_to_quat()`、`AffineSchurSolver` |
| `avbd_solver.h` | `Rigid` 新增 `affine`/`initialAff`/`inertialAff`/`inertiaMatrix` 字段、`syncFromQuat()`/`syncFromAffine()`/`rotateVec()`/`transformVec()` 方法；`Solver` 新增 `rotation_mode` |
| `avbd_rigid.cpp` | 构造函数初始化仿射矩阵字段 |
| `avbd_solver.cpp` | CPU 路径增加 Affine 模式的预测/求解/速度恢复分支；`unpack()` 调用 `syncFromQuat()` |
| `avbd_manifold.cpp` | `initialize()`/`updatePrimal()`/`updateDual()` 使用 `rotateVec()`/`transformVec()` 和模式感知的角位移计算 |
| `avbd_joint.cpp` | `initialize()`/`updatePrimal()`/`updateDual()` 全部使用模式感知的旋转和角度差异 |
| `avbd_spring.cpp` | 使用 `transformVec()`/`rotateVec()` 替代硬编码的四元数旋转 |
| `avbd_collide.cpp` | 无需修改 — 碰撞检测通过 `positionAng`（始终同步）工作 |

### 8.4 仿射模式的关键实现细节

| 模块 | 轴角模式（原始） | 仿射矩阵模式（新增） |
|------|---------|-------------|
| 旋转存储 | `quat positionAng` (4 float) | `float3x3 affine` (9 float) + 自动同步到 quat |
| 旋转变换 | `rotate(q, v)` | `affine * v` |
| 增量表示 | `δθ` (3D 轴角) | `δθ` → `skew(δθ)` → `(I + skew) * A` → `polar_rotation()` |
| 角位移 | `q_a - q_b` → 3D | `vee(A_new · A_old^T - I)` → 3D |
| 增量更新 | `q + δθ` 归一化 | `(I + skew(δθ)) · A` 然后极分解 |
| Hessian | 6×6 LDLᵀ（保持） | 同样使用 6×6（角增量仍为 3D） |
| 惯性 Hessian | `diag(I)/Δt²` | `diag(I)/Δt²`（相同） |
| 碰撞 Jacobian | `r^{world} × B` | 同样使用叉积（增量仍在切空间） |
| 速度恢复 | `2·vec(q_{new}·q_{old}^{-1})/Δt` | `vee((R_{new}·R_{old}^T - I))/Δt` |

### 8.5 12×12 Schur 补求解器实现

当前实现已使用 PABD 论文原版的 **12×12 Schur 补分解**。广义坐标为 $q = (c, A) \in \mathbb{R}^{12}$，其中 $c \in \mathbb{R}^3$ 是质心平移，$A \in \mathbb{R}^{3\times 3}$ 是仿射旋转矩阵（9 自由度）。

**12×12 块结构**：

$$
\begin{bmatrix} m + \Sigma\alpha & \Sigma\alpha\cdot r^T \\ \Sigma\alpha\cdot r & I_0 + \Sigma\alpha\cdot r\cdot r^T \end{bmatrix}
\begin{bmatrix} \Delta c \\ \Delta A \end{bmatrix}
=
\begin{bmatrix} \text{src}_c \\ \text{src}_A \end{bmatrix}
$$

其中 $\Sigma\alpha = \sum_c h^2 k_c$ 为约束权重之和，$r$ 为局部接触臂。

**Schur 补分解** (`AffineSchurSolver::factor`)：

$$S = \frac{I_0 + \text{sumD}}{k} - C \cdot B^T, \quad k = m + \text{sum0}$$

$$M^{-1} = \begin{bmatrix} a & b^T \\ c & D \end{bmatrix}$$

其中 $D = S^{-1}/k$, $c = -S^{-1} C / k$, $b = -S^{-T} B / k$, $a = (1 + B^T S^{-1} C) / k$。

**求解流程** (`stepCpuAffine` Phase 3)：

1. **累积 Hessian 和**：对每个约束调用 `schur.accumulate(α_c, r_local)` 累积 `sum0`, `sumB`, `sumD`
2. **分解**：`schur.factor(mass, inertiaMatrix)` 计算 $M^{-1}$
3. **累积 RHS**：碰撞约束的梯度在 $q^*$ 参考帧中累积为 `srcC`（平移分量）和 `srcA`（矩阵分量）
4. **求解**：`schur.solve_update(srcC, srcA, dxLin, dxAff)` 得到局部增量
5. **全局更新**：$c = c^* + A^* \cdot \Delta c_{\text{local}}$, $A = A^* \cdot \Delta A_{\text{local}}$
6. **投影回 SO(3)**：极分解 `polar_rotation(A)` 保证正交性

与之前的 6×6 切空间近似不同，此实现在矩阵的 **9D 空间**中直接求解增量，避免了 `skew`/`mat_to_angular` 的小角度截断误差。
