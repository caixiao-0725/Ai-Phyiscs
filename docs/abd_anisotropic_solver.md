# ABD 各向异性 Schur Complement 求解器：完整推导与实现

## 1. 问题描述

我们需要求解 **仿射体动力学 (Affine Body Dynamics, ABD)** 的隐式时间积分问题。每个刚体的自由度为 12 维：

$$
\mathbf{q} = (\mathbf{c}_L \in \mathbb{R}^3,\; \mathbf{A}_L \in \mathbb{R}^{3\times3})
$$

其中 $\mathbf{c}_L$ 是相对于惯性预测位置 $\tilde{\mathbf{c}}$ 的平移偏移，$\mathbf{A}_L$ 是相对于惯性预测仿射矩阵 $\tilde{\mathbf{A}}$ 的变形矩阵。

世界坐标系中的位置恢复为：

$$
\mathbf{x}^{\text{world}} = \tilde{\mathbf{c}} + \tilde{\mathbf{A}}\,(\mathbf{c}_L + \mathbf{A}_L\,\mathbf{r})
$$

其中 $\mathbf{r}$ 是局部参考坐标。当无约束力时，$\mathbf{c}_L = \mathbf{0}$，$\mathbf{A}_L = \mathbf{I}$，物体保持惯性预测状态。

---

## 2. 能量表达式

### 2.1 惯性能量

惯性能量包含平移惯性和旋转惯性两部分：

$$
E_{\text{inertia}} = \frac{1}{2}\,m\,\|\mathbf{c}_L\|^2 + \frac{1}{2}\sum_{j=0}^{2} (\mathbf{A}_L[j] - \mathbf{e}_j)^\top \mathbf{J}\, (\mathbf{A}_L[j] - \mathbf{e}_j)
$$

其中：

- $m$ 为质量
- $\mathbf{J} = \text{diag}(J_0, J_1, J_2)$ 为惯性矩阵
- $\mathbf{A}_L[j]$ 为 $\mathbf{A}_L$ 的第 $j$ 行（3 维向量）
- $\mathbf{e}_j$ 为第 $j$ 个标准基向量

当 $\mathbf{A}_L = \mathbf{I}$ 时，旋转惯性为零。

### 2.2 约束能量（增广拉格朗日）

每个接触点在 3 个方向 $d \in \{0(\text{法向}),\, 1(\text{切向}_x),\, 2(\text{切向}_y)\}$ 上有独立的约束：

$$
C_d = \tilde{\mathbf{n}}_d^\top (\mathbf{c}_L + \mathbf{A}_L\,\mathbf{r}) + \xi_d
$$

其中：

- $\tilde{\mathbf{n}}_d = \tilde{\mathbf{A}}^\top \mathbf{n}_d^{\text{world}}$ 为约束方向在局部坐标中的表示
- $\xi_d = \mathbf{n}_d^{\text{world}\top} \tilde{\mathbf{c}} - t_d$ 为惯性预测处的约束偏移量
- $t_d$ 为约束目标值（如地面 $z$ 坐标、锚定切向位置等）

增广拉格朗日能量：

$$
E_{\text{AL}} = \sum_{\text{contacts}} \sum_{d} \left[\frac{\kappa_d}{2}\,C_d^2 + \lambda_d\,C_d\right]
$$

其中 $\kappa_d$ 为 penalty 参数，$\lambda_d$ 为拉格朗日乘子。

### 2.3 总能量

$$
E = E_{\text{inertia}} + E_{\text{AL}}
$$

---

## 3. 一阶导数（梯度）

### 3.1 对 $\mathbf{c}_L$ 求导

$$
\frac{\partial E}{\partial \mathbf{c}_L} = m\,\mathbf{c}_L + \sum_d (\kappa_d\,C_d + \lambda_d)\,\tilde{\mathbf{n}}_d
$$

### 3.2 对 $\mathbf{A}_L[j]$ 求导

$$
\frac{\partial E}{\partial \mathbf{A}_L[j]} = \mathbf{J}\,(\mathbf{A}_L[j] - \mathbf{e}_j) + \sum_d (\kappa_d\,C_d + \lambda_d)\,\tilde{\mathbf{n}}_d\,r_j
$$

其中 $r_j$ 是 $\mathbf{r}$ 的第 $j$ 个分量。

### 3.3 定义约束力

$$
F_d \triangleq \kappa_d\,C_d + \lambda_d
$$

经 Coulomb 摩擦锥投影后（见第 6 节），梯度简化为：

$$
\nabla_{\mathbf{c}_L} E = m\,\mathbf{c}_L + \sum_d F_d\,\tilde{\mathbf{n}}_d
$$

$$
\nabla_{\mathbf{A}_L[j]} E = \mathbf{J}\,(\mathbf{A}_L[j] - \mathbf{e}_j) + \sum_d F_d\,\tilde{\mathbf{n}}_d\,r_j
$$

---

## 4. 二阶导数（Hessian）

使用 Gauss-Newton 近似（对线性约束精确）。单个方向约束 $C_d = \tilde{\mathbf{n}}_d^\top(\mathbf{c}_L + \mathbf{A}_L\,\mathbf{r})$ 的 Jacobian 为 12 维行向量：

$$
\mathbf{J}_d = \begin{bmatrix} \tilde{\mathbf{n}}_d \\ \tilde{\mathbf{n}}_d\,r_0 \\ \tilde{\mathbf{n}}_d\,r_1 \\ \tilde{\mathbf{n}}_d\,r_2 \end{bmatrix}^\top
$$

对应的 Hessian 贡献 $\kappa_d\,\mathbf{J}_d^\top\mathbf{J}_d$ 展开为分块结构：

### 4.1 Hessian 分块

$$
\mathbf{H} = \begin{bmatrix} \mathbf{K} & \mathbf{B}_0 & \mathbf{B}_1 & \mathbf{B}_2 \\ \mathbf{B}_0^\top & \mathbf{D}_{00} & \mathbf{D}_{01} & \mathbf{D}_{02} \\ \mathbf{B}_1^\top & \mathbf{D}_{10} & \mathbf{D}_{11} & \mathbf{D}_{12} \\ \mathbf{B}_2^\top & \mathbf{D}_{20} & \mathbf{D}_{21} & \mathbf{D}_{22} \end{bmatrix}
$$

各分块的表达式（3×3 矩阵）：

$$
\mathbf{K} = m\,\mathbf{I}_3 + \underbrace{\sum_{\text{contacts}}\sum_d \kappa_d\,(\tilde{\mathbf{n}}_d \otimes \tilde{\mathbf{n}}_d)}_{\texttt{sumK}}
$$

$$
\mathbf{B}_j = \underbrace{\sum_{\text{contacts}}\sum_d \kappa_d\,(\tilde{\mathbf{n}}_d \otimes \tilde{\mathbf{n}}_d)\,r_j}_{\texttt{sumB[j]}}
$$

$$
\mathbf{D}_{j_1 j_2} = \delta_{j_1 j_2}\,\mathbf{J} + \underbrace{\sum_{\text{contacts}}\sum_d \kappa_d\,(\tilde{\mathbf{n}}_d \otimes \tilde{\mathbf{n}}_d)\,r_{j_1}\,r_{j_2}}_{\texttt{sumD[j1][j2]}}
$$

**关键性质**：所有分块共享相同的基矩阵 $\mathbf{M} = \kappa_d\,(\tilde{\mathbf{n}}_d \otimes \tilde{\mathbf{n}}_d)$（3×3 对称），乘以 $\mathbf{r}$ 分量的标量积即可得到各分块。

### 4.2 各向异性 vs 各向同性

| | 各向同性近似 | 各向异性精确 |
|---|---|---|
| $\mathbf{K}$ | $(\,m + \sum\kappa\,)\,\mathbf{I}_3$（标量） | $m\,\mathbf{I}_3 + \sum\kappa_d\,(\tilde{\mathbf{n}}_d\otimes\tilde{\mathbf{n}}_d)$（3×3 矩阵）|
| $\mathbf{B}_j$ | $\sum\kappa\,r_j\,\mathbf{I}_3$（标量乘 I） | $\sum\kappa_d\,(\tilde{\mathbf{n}}_d\otimes\tilde{\mathbf{n}}_d)\,r_j$（3×3 矩阵）|
| $\mathbf{D}_{j_1j_2}$ | $\delta\mathbf{J} + \sum\kappa\,r_{j_1}r_{j_2}\,\mathbf{I}_3$ | $\delta\mathbf{J} + \sum\kappa_d\,(\tilde{\mathbf{n}}_d\otimes\tilde{\mathbf{n}}_d)\,r_{j_1}r_{j_2}$|

各向异性精确版在摩擦场景下尤为重要：法向和切向的 penalty $(\kappa_n, \kappa_{t_1}, \kappa_{t_2})$ 通常差异很大。

---

## 5. 线性系统的求解：Schur Complement + Block Cholesky

### 5.1 系统方程

令梯度为零 $\nabla E = 0$，整理为线性系统 $\mathbf{H}\,\mathbf{q} = \mathbf{b}$：

$$
\begin{bmatrix} \mathbf{K} & \mathbf{B} \\ \mathbf{B}^\top & \mathbf{D} \end{bmatrix}
\begin{bmatrix} \mathbf{c}_L \\ \mathbf{a} \end{bmatrix}
= \begin{bmatrix} \mathbf{b}_c \\ \mathbf{b}_A \end{bmatrix}
$$

其中 $\mathbf{a} = [\mathbf{A}_L[0];\,\mathbf{A}_L[1];\,\mathbf{A}_L[2]]$ 是 9 维向量。

**RHS 推导**：从 $\nabla E = 0$ 在 $(\mathbf{c}_L,\,\mathbf{A}_L)$ 处展开，目标 $\mathbf{q}_{\text{target}} = (\mathbf{0},\,\mathbf{I})$：

$$
\mathbf{b}_c = \underbrace{\sum_j \texttt{sumB}[j]\,\mathbf{e}_j}_{\text{Hessian 在目标处}} \;-\; \sum_d F_d\,\tilde{\mathbf{n}}_d
$$

$$
\mathbf{b}_A[j] = \underbrace{\mathbf{J}\,\mathbf{e}_j + \sum_k \texttt{sumD}[j][k]\,\mathbf{e}_k}_{\text{Hessian}\times\mathbf{I}} \;-\; \sum_d F_d\,\tilde{\mathbf{n}}_d\,r_j
$$

其中 $F_d = \kappa_d\,\xi_d + \lambda_d$ 在线性化点（惯性预测）处评估。

> **注意**：RHS 中 $\sum_k \texttt{sumD}[j][k]\cdot\mathbf{e}_k$ 项来自 penalty Hessian 在 $\mathbf{A}_L = \mathbf{I}$ 目标处的贡献。省略此项会导致 $\mathbf{A}_L \approx \mathbf{J}\,(\mathbf{J}+\texttt{sumD})^{-1} \neq \mathbf{I}$，造成严重的数值不稳定。

### 5.2 Schur Complement 消去 $\mathbf{c}_L$

将 $\mathbf{c}_L$ 从第一个方程消去：

$$
\mathbf{c}_L = \mathbf{K}^{-1}\,(\mathbf{b}_c - \mathbf{B}\,\mathbf{a})
$$

代入第二个方程，得到 **Schur 补** 系统：

$$
\mathbf{S}\,\mathbf{a} = \mathbf{b}_A' \quad\text{其中}\quad
\begin{cases}
\mathbf{S}_{j_1 j_2} = \mathbf{D}_{j_1 j_2} - \mathbf{B}_{j_1}^\top\,\mathbf{K}^{-1}\,\mathbf{B}_{j_2} \\
\mathbf{b}_A'[j] = \mathbf{b}_A[j] - \mathbf{B}_j^\top\,\mathbf{K}^{-1}\,\mathbf{b}_c
\end{cases}
$$

$\mathbf{K}$ 是 3×3 SPD 矩阵，一次 `inverse3x3` 即可。

### 5.3 9×9 Block Cholesky

$\mathbf{S}$ 是 9×9 对称正定矩阵，由 3×3 分块组成。使用 **分块 Cholesky 分解** $\mathbf{S} = \mathbf{L}\,\mathbf{L}^\top$：

$$
\mathbf{L} = \begin{bmatrix}
\mathbf{L}_{00} & & \\
\mathbf{L}_{10} & \mathbf{L}_{11} & \\
\mathbf{L}_{20} & \mathbf{L}_{21} & \mathbf{L}_{22}
\end{bmatrix}
$$

分解步骤：

| 步骤 | 公式 |
|------|------|
| 1 | $\mathbf{L}_{00} = \text{chol}(\mathbf{S}_{00})$ |
| 2 | $\mathbf{L}_{10} = \mathbf{S}_{10}\,\mathbf{L}_{00}^{-\top}$ |
| 3 | $\mathbf{L}_{20} = \mathbf{S}_{20}\,\mathbf{L}_{00}^{-\top}$ |
| 4 | $\mathbf{L}_{11} = \text{chol}(\mathbf{S}_{11} - \mathbf{L}_{10}\,\mathbf{L}_{10}^\top)$ |
| 5 | $\mathbf{L}_{21} = (\mathbf{S}_{21} - \mathbf{L}_{20}\,\mathbf{L}_{10}^\top)\,\mathbf{L}_{11}^{-\top}$ |
| 6 | $\mathbf{L}_{22} = \text{chol}(\mathbf{S}_{22} - \mathbf{L}_{20}\,\mathbf{L}_{20}^\top - \mathbf{L}_{21}\,\mathbf{L}_{21}^\top)$ |

每个 `chol` 和 `inverse` 操作只涉及 3×3 矩阵。

### 5.4 求解

1. **前代** $\mathbf{L}\,\mathbf{y} = \mathbf{b}_A'$：
$$
\mathbf{y}[0] = \mathbf{L}_{00}^{-1}\,\mathbf{b}'[0], \quad
\mathbf{y}[1] = \mathbf{L}_{11}^{-1}\,(\mathbf{b}'[1] - \mathbf{L}_{10}\,\mathbf{y}[0]), \quad \ldots
$$

2. **回代** $\mathbf{L}^\top\,\mathbf{a} = \mathbf{y}$：
$$
\mathbf{a}[2] = \mathbf{L}_{22}^{-\top}\,\mathbf{y}[2], \quad
\mathbf{a}[1] = \mathbf{L}_{11}^{-\top}\,(\mathbf{y}[1] - \mathbf{L}_{21}^\top\,\mathbf{a}[2]), \quad \ldots
$$

3. **回代 $\mathbf{c}_L$**：
$$
\mathbf{c}_L = \mathbf{K}^{-1}\,\Bigl(\mathbf{b}_c - \sum_j \mathbf{B}_j\,\mathbf{a}[j]\Bigr)
$$

---

## 6. Coulomb 摩擦锥投影

约束力 $\mathbf{F} = (F_n,\,F_{t_1},\,F_{t_2})$ 需要满足：

1. **法向单向性**：$F_n \leq 0$（压缩力）
2. **Coulomb 摩擦锥**：$\|\mathbf{F}_t\| \leq \mu\,|F_n|$

投影算法：

$$
F_n \leftarrow \min(F_n,\;0)
$$

$$
\text{if } \|\mathbf{F}_t\| > \mu\,|F_n| : \quad \mathbf{F}_t \leftarrow \mu\,|F_n|\,\frac{\mathbf{F}_t}{\|\mathbf{F}_t\|}
$$

---

## 7. 增广拉格朗日对偶更新

每次 primal 求解后，更新拉格朗日乘子和 penalty 参数：

$$
\lambda_d \leftarrow F_d = \kappa_d\,C_d + \lambda_d
$$

$$
\kappa_d \leftarrow \min\!\bigl(\kappa_d + \beta\,|C_d|,\;\kappa_{\max}\bigr)
$$

其中 $\beta$ 为 penalty 增长速率，$\kappa_{\max}$ 防止数值溢出。乘子更新后同样需要经过 Coulomb 投影。

---

## 8. 整体算法流程

```
每帧 (dt):
├── Phase 1: 惯性预测 (Symplectic Euler)
│   ├── v ← v + g·dt
│   ├── c̃ ← x + v·dt        (inertialLin)
│   └── Ã ← (I + [ω]×·dt)·A   (inertialAff, 再 polar projection)
│
├── Phase 2: 碰撞检测 + Warm Start
│   ├── 检测 ground contacts (gap < AVBD_GC_TOL)
│   ├── 检测 box-box contacts (SAT / manifold)
│   └── 从上一帧匹配继承 λ, κ (warm start)
│
├── Phase 3: 增广拉格朗日迭代 (重复 iterations 次)
│   ├── 迭代内重检测 (contacts 可能因位置更新而变化)
│   │
│   ├── Primal Update (逐 body):
│   │   ├── 1. 累积 Hessian:
│   │   │      对每个 contact, 每个方向 d:
│   │   │        ñ_d = Ã^T · n_d^world
│   │   │        schur.accumulate_dir(κ_d, ñ_d, r)
│   │   │
│   │   ├── 2. 分解 Hessian:
│   │   │        schur.factor(mass, inertia)
│   │   │
│   │   ├── 3. 构造 RHS:
│   │   │      srcC = Σ_j sumB[j]·e_j                   [Hessian×target]
│   │   │      srcA = J + Σ_{j,k} sumD[j][k]·e_k        [Hessian×target]
│   │   │      对每个 contact:
│   │   │        e_d = ñ_d^T·r + ξ_d                     [约束值]
│   │   │        F_d = κ_d·e_d + λ_d                     [Coulomb 投影]
│   │   │        srcC -= F_d·ñ_d
│   │   │        srcA[j] -= F_d·ñ_d·r_j
│   │   │
│   │   ├── 4. 求解:
│   │   │        schur.solve_update(srcC, srcA, cL, AL)
│   │   │
│   │   └── 5. 更新位置:
│   │          positionLin = c̃ + Ã·cL
│   │          affine = polar_rotation(Ã·AL)
│   │
│   └── Dual Update:
│       ├── 计算约束违反 C_d
│       ├── F_d = κ_d·C_d + λ_d  →  Coulomb 投影
│       ├── λ_d ← F_d
│       └── κ_d ← min(κ_d + β·|C_d|, κ_max)
│
└── Phase 4: 速度恢复
    ├── v ← (x_new - x_old) / dt
    └── ω ← mat_to_angular(A_new, A_old) / dt
```

---

## 9. 核心代码实现

### 9.1 Hessian 累积 (`accumulate_dir`)

```cpp
void accumulate_dir(float kappa, float3 n, float3 r) {
    float3x3 M = outer(n, n) * kappa;   // κ·(ñ⊗ñ), 3×3
    sumK += M;
    for (int j = 0; j < 3; j++) {
        float3x3 Mr = M * r[j];
        sumB[j] += Mr;
        for (int k = j; k < 3; k++) {
            float3x3 Mrr = Mr * r[k];
            sumD[j][k] += Mrr;
            if (k != j) sumD[k][j] += Mrr;   // 对称性
        }
    }
}
```

一次调用计算 $\mathbf{M} = \kappa\,(\tilde{\mathbf{n}} \otimes \tilde{\mathbf{n}})$，然后将 $\mathbf{M}$、$\mathbf{M}\,r_j$ 和 $\mathbf{M}\,r_{j_1}\,r_{j_2}$ 分别累加到 `sumK`、`sumB[j]`、`sumD[j1][j2]`。利用 `sumD` 的对称性只计算上三角。

### 9.2 分解 (`factor`)

```cpp
void factor(float mass, float3x3 inertia) {
    // K = m·I + sumK
    float3x3 K = identity3x3() * mass + sumK;
    Kinv = inverse3x3(K);

    // B blocks
    for (int j = 0; j < 3; j++)
        Bblk[j] = sumB[j];

    // Schur complement: S[j1][j2] = D[j1][j2] + δ·J - B^T[j1]·Kinv·B[j2]
    float3x3 S[3][3];
    for (int j1 = 0; j1 < 3; j1++) {
        for (int j2 = j1; j2 < 3; j2++) {
            float3x3 Dblk = sumD[j1][j2];
            if (j1 == j2) Dblk += inertia;
            S[j1][j2] = Dblk - transpose(Bblk[j1]) * Kinv * Bblk[j2];
            if (j2 != j1) S[j2][j1] = transpose(S[j1][j2]);
        }
    }

    block_cholesky(S);
}
```

先求 $\mathbf{K}^{-1}$（3×3 矩阵求逆），构建 9×9 Schur 补 $\mathbf{S}$，最后做分块 Cholesky 分解。

### 9.3 求解 (`solve_update`)

```cpp
void solve_update(float3 srcC, float3x3 srcA, float3& dxLin, float3x3& dxAff) {
    // Schur 修正后的 A_L RHS
    float3 Kinv_bc = Kinv * srcC;
    float3 rhs_A[3];
    for (int j = 0; j < 3; j++)
        rhs_A[j] = srcA[j] - transpose(Bblk[j]) * Kinv_bc;

    // Block Cholesky 前代 + 回代
    float3 y[3], A_cols[3];
    block_forward(rhs_A, y);
    block_backward(y, A_cols);

    // 回代 c_L
    float3 Bx = {0,0,0};
    for (int j = 0; j < 3; j++)
        Bx = Bx + Bblk[j] * A_cols[j];
    dxLin = Kinv * (srcC - Bx);

    for (int j = 0; j < 3; j++)
        dxAff[j] = A_cols[j];
}
```

### 9.4 RHS 构造（Primal Update 中）

```cpp
// 初始化：Hessian × q_target + inertia target
float3 srcC = {0, 0, 0};
float3x3 srcA = body->inertiaMatrix;    // J·e_j 项

// Hessian 在 A_L=I 目标处的贡献
for (int j = 0; j < 3; j++)
    for (int k = 0; k < 3; k++)
        srcA[j] = srcA[j] + schur.sumD[j][k].col(k);
for (int j = 0; j < 3; j++)
    srcC = srcC + schur.sumB[j].col(j);

// 约束力贡献（per contact, per direction）
for (auto& gc : groundContacts) {
    // ...
    float3 xRef = cTilde + ATilde * gc.rLocal;
    float3 F;
    for (int d = 0; d < 3; d++) {
        float e_d = dot(dirs[d], xRef) - dot(dirs[d], targets[d]);
        F[d] = gc.penalty[d] * e_d + gc.lambda[d];
    }
    F.x = std::min(F.x, 0.0f);       // 法向单向
    // Coulomb 摩擦投影 ...

    for (int d = 0; d < 3; d++) {
        float3 nd_local = ATildeT * dirs[d];
        srcC = srcC - nd_local * F[d];
        for (int j = 0; j < 3; j++)
            srcA[j] = srcA[j] - nd_local * (F[d] * gc.rLocal[j]);
    }
}
```

**RHS 的物理意义**：

- `srcA = J·e_j + Σ sumD[j][k]·e_k`：惯性和 penalty Hessian 将 $\mathbf{A}_L$ 拉向 $\mathbf{I}$
- `- Σ F_d·ñ_d·r_j`：约束力将 $\mathbf{A}_L$ 推离 $\mathbf{I}$ 以满足约束
- 两者平衡时 $\mathbf{A}_L \approx \mathbf{I} + \text{small correction}$

### 9.5 对偶更新

```cpp
// Ground contact dual update
float3 C;
C.x = worldPos.z - ground_z;              // 法向 gap
C.y = worldPos.x - initPos.x;             // 切向滑移 x
C.z = worldPos.y - initPos.y;             // 切向滑移 y

float3 F;
for (int d = 0; d < 3; d++)
    F[d] = gc.penalty[d] * C[d] + gc.lambda[d];
F.x = std::min(F.x, 0.0f);               // 法向单向

// Coulomb projection
float bounds = std::fabs(F.x) * gc.friction;
float tangMag = length(float2{F.y, F.z});
bool isStatic = tangMag <= bounds;
if (!isStatic && tangMag > 0) {
    F.y *= bounds / tangMag;
    F.z *= bounds / tangMag;
}
gc.lambda = F;                             // 更新乘子

// penalty 增长（仅约束活跃时）
if (F.x < 0)
    gc.penalty.x = std::min(gc.penalty.x + beta * std::fabs(C.x), PENALTY_MAX);
if (isStatic) {
    gc.penalty.y = std::min(gc.penalty.y + beta * std::fabs(C.y), PENALTY_MAX);
    gc.penalty.z = std::min(gc.penalty.z + beta * std::fabs(C.z), PENALTY_MAX);
}
```

---

## 10. 位置更新与旋转投影

求解得到 $(\mathbf{c}_L,\,\mathbf{A}_L)$ 后：

$$
\mathbf{x}^{\text{new}} = \tilde{\mathbf{c}} + \tilde{\mathbf{A}}\,\mathbf{c}_L
$$

$$
\mathbf{A}^{\text{new}} = \text{polar\_rotation}(\tilde{\mathbf{A}}\,\mathbf{A}_L)
$$

`polar_rotation` 提取仿射矩阵的旋转部分 $\mathbf{R} = \mathbf{A}\,(\mathbf{A}^\top\mathbf{A})^{-1/2}$，确保仿射矩阵保持正交。

---

## 11. 碰撞检测鲁棒性

地面碰撞检测使用小容差 `AVBD_GC_TOL = 1e-4`：

```cpp
if (worldPos.z - ground_z >= AVBD_GC_TOL) continue;
```

防止求解器的数值噪声导致接触点在相邻迭代间闪烁（创建/删除），避免因此引发的不稳定。

---

## 12. 数值稳定性设计

1. **分块求解**（非整体 12×12 Cholesky）：所有运算在 3×3 矩阵上进行，避免了 12×12 整体 Cholesky 中不同自由度之间的数值耦合，保持了各方向的对称性。

2. **3×3 Cholesky 正则化**：对角元素有 `1e-12f` 的下界，防止非正定时崩溃：
   ```cpp
   L[0][0] = sqrt(max(A[0][0], 1e-12f));
   ```

3. **RHS 的 Hessian×target 项**：确保无约束时 $\mathbf{A}_L = \mathbf{I}$，有约束时 $\mathbf{A}_L \approx \mathbf{I} + O(\text{gap}/\kappa)$，避免 $\mathbf{A}_L$ 被约束 Hessian "拉扁"。

4. **Warm start**：跨帧和跨迭代继承 $\lambda$、$\kappa$，加速增广拉格朗日收敛。
