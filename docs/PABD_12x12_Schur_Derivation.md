# PABD 12×12 Schur 补求解器推导

## 1. 广义坐标

刚体状态用 12 个自由度表示：

$$q = (c,\, A) \in \mathbb{R}^{12}$$

| 符号 | 维度 | 含义 |
|------|------|------|
| $c \in \mathbb{R}^3$ | 3 | 质心平移（全局） |
| $A \in \mathbb{R}^{3\times 3}$ | 9 | 仿射旋转矩阵 |

世界坐标中某 body-local 点 $r$ 的位置：

$$y(q) = c + A \cdot r$$

---

## 2. 优化问题

每个时间步求解隐式变分问题：

$$\min_{q}\; \underbrace{\frac{1}{2}(q - \tilde{q})^T M (q - \tilde{q})}_{\text{惯性项}} \;+\; \underbrace{\sum_{c} \Psi_c(q)}_{\text{约束能量}}$$

其中 $\tilde{q} = (\tilde{c},\, \tilde{A})$ 为惯性预测（symplectic Euler）：

$$\tilde{c} = c^n + \Delta t\,(v^n + \Delta t\, g)$$
$$\tilde{A} = (I + [\omega^n \Delta t]_\times) A^n$$

质量矩阵 $M$ 为对角块结构：

$$M = \begin{bmatrix} m I_3 & 0 \\ 0 & I_0 \otimes I_3 \end{bmatrix}$$

其中 $m$ 为质量标量，$I_0 = \mathrm{diag}(I_x, I_y, I_z)$ 为参考帧转动惯量。

---

## 3. 地面碰撞约束（二次罚函数）

### 3.1 约束定义

对每个穿透地面的顶点 $v$，设 body-local 坐标为 $r_v$，世界坐标 $y_v = c + A r_v$。地面投影点为 $x_v^{\text{target}} = (y_{v,x},\, y_{v,y},\, z_g)$。

使用 3D point-to-point 二次罚函数：

$$\Psi_v(q) = \frac{1}{2} k \|y_v(q) - x_v^{\text{target}}\|^2$$

其中 $k$ 为罚刚度（`contactStiffness`）。

### 3.2 Jacobian

$$J_v = \frac{\partial y_v}{\partial q} = \begin{bmatrix} \frac{\partial y}{\partial c} \\ \frac{\partial y}{\partial \mathrm{vec}(A)} \end{bmatrix} = \begin{bmatrix} I_3 \\ r_v \otimes I_3 \end{bmatrix} \quad (12 \times 3)$$

### 3.3 梯度与近似 Hessian

梯度（Gauss-Newton 型）：

$$\nabla_q \Psi_v = k \, J_v^T (y_v - x_v^{\text{target}})$$

近似 Hessian（忽略 $y - x^{\text{target}}$ 对 $q$ 的二阶导）：

$$H_v \approx k \, J_v^T J_v = k \begin{bmatrix} I_3 & r_v^T \otimes I_3 \\ r_v \otimes I_3 & (r_v r_v^T) \otimes I_3 \end{bmatrix}$$

---

## 4. 局部帧变换

为数值稳定性，在惯性预测 $\tilde{q}$ 的参考帧中求解。定义局部坐标：

$$c = \tilde{c} + \tilde{A}\, c_L, \quad A = \tilde{A}\, A_L$$

惯性目标在局部帧中为原点：$q_0^L = (0,\, I_3)$。

约束目标在局部帧中为：

$$x_v^L = \tilde{A}^T (x_v^{\text{target}} - \tilde{c})$$

局部帧中 contact 点的 Jacobian 结构不变：$J_v^L = [I_3;\; r_v \otimes I_3]$。

---

## 5. KKT 最优性条件

一阶最优条件（Newton 方程）：

$$(M + \sum_v H_v)\, q_L = M\, q_0^L + \sum_v k\, J_v^T\, x_v^L$$

展开块结构：

$$\underbrace{\begin{bmatrix} m + \sum k & \sum k\, r_v^T \\ \sum k\, r_v & I_0 + \sum k\, r_v r_v^T \end{bmatrix}}_{M_{\text{total}}} \begin{bmatrix} c_L \\ A_L \end{bmatrix} = \begin{bmatrix} \text{src}_c \\ \text{src}_A \end{bmatrix}$$

其中 RHS 各分量为：

$$\text{src}_c = m \cdot 0 + \sum_v k \cdot x_v^L = \sum_v k \cdot x_v^L$$

$$\text{src}_A = I_0 \cdot I_3 + \sum_v k \cdot (x_v^L \otimes r_v)$$

**无碰撞时**：$\text{src}_c = 0$，$\text{src}_A = I_0$，解为 $q_L = (0, I)$，即 $q = \tilde{q}$（保持惯性预测）。

---

## 6. Schur 补分解

### 6.1 定义辅助累积量

$$s_0 = \sum_v k, \quad s_B = s_C = \sum_v k \cdot r_v, \quad s_D = \sum_v k \cdot r_v r_v^T$$

### 6.2 归一化

$$\kappa = m + s_0, \quad B = s_B / \kappa, \quad C = s_C / \kappa$$

### 6.3 Schur 补矩阵

$$S = \frac{I_0 + s_D}{\kappa} - C\, B^T \quad \in \mathbb{R}^{3\times 3}$$

### 6.4 逆矩阵分解

$$M_{\text{total}}^{-1} = \begin{bmatrix} a & b^T \\ c & D \end{bmatrix}$$

其中：

| 符号 | 公式 | 维度 |
|------|------|------|
| $D$ | $S^{-1} / \kappa$ | $3 \times 3$ |
| $c$ | $-S^{-1} C / \kappa$ | $3 \times 1$ |
| $b$ | $-S^{-T} B / \kappa$ | $3 \times 1$ |
| $a$ | $(1 + B^T S^{-1} C) / \kappa$ | 标量 |

### 6.5 求解

$$c_L = a \cdot \text{src}_c + \text{src}_A \cdot b$$

$$A_L = \text{src}_c \otimes c + \text{src}_A \cdot D^T$$

### 6.6 局部→全局

$$c^{n+1} = \tilde{c} + \tilde{A}\, c_L$$

$$A^{n+1} = \text{polar}(\tilde{A}\, A_L)$$

极分解 `polar(·)` 将仿射矩阵投影回 $SO(3)$。

---

## 7. 增广拉格朗日迭代

上述 Schur 补求解嵌入在一个多次迭代的求解循环中：

```
for it = 0 to N_iter:
    // 1. 碰撞检测：重新检测穿透顶点
    contacts = detect_penetrating_vertices(body, ground_z)

    for each body:
        // 2. 累积 Hessian
        schur.reset()
        for each contact c of this body:
            schur.accumulate(k, r_c^local)

        // 3. 分解 Schur 补
        schur.factor(mass, I_0)

        // 4. 构建 RHS
        src_c = Σ k · x_j^local
        src_A = I_0 + Σ k · (x_j^local ⊗ r_c)

        // 5. 求解
        (c_L, A_L) = schur.solve_update(src_c, src_A)

        // 6. 更新全局坐标
        c = c̃ + Ã · c_L
        A = polar(Ã · A_L)

    // 7. 更新对偶变量（joints/springs 等其他约束）
    for each force: force.updateDual(α)
```

每次迭代中：

- **碰撞集合更新**：位置改变后重新检测穿透
- **Hessian 重建**：约束集变化 → 重建 Schur 补
- **位置收敛**：迭代使穿透深度 $\to 0$（由 $k/(m+k)$ 收缩率控制）

### 7.1 收敛性分析

对单个碰撞点，一次迭代后的穿透衰减率为：

$$\frac{d^{(i+1)}}{d^{(i)}} \approx \frac{m}{m + k}$$

当 $k \gg m$ 时收敛很快。例如 $m=1$, $k=10000$ 时，衰减率 $\approx 10^{-4}$，一次迭代即可消除 99.99% 的穿透。

### 7.2 与纯罚函数的区别

| 方面 | 纯罚函数（无迭代） | 迭代求解（当前实现） |
|------|-----|------|
| 穿透残余 | $d \cdot m/(m+k)$ | 接近 0（指数收敛） |
| 刚度 $k$ 选择 | 需要很大，导致刚性问题 | 中等即可，迭代补偿 |
| 每步计算量 | 1 次 Schur solve | $N$ 次 Schur solve |
| 碰撞集 | 固定 | 随迭代更新 |

---

## 8. 速度恢复

约束求解结束后，从位置差分恢复速度：

$$v^{n+1} = \frac{c^{n+1} - c^n}{\Delta t}$$

$$\omega^{n+1} = \frac{\text{vee}(A^{n+1} (A^n)^T - I)}{\Delta t}$$

其中 $\text{vee}(\cdot)$ 从反对称矩阵提取 3D 角速度向量。

---

## 9. 代码对应

| 公式 | 代码 (`avbd_solver.cpp` / `avbd_maths.h`) |
|------|------------------------------------------|
| $s_0, s_B, s_D$ 累积 | `AffineSchurSolver::accumulate(k, rLocal)` |
| Schur 补分解 $M^{-1}$ | `AffineSchurSolver::factor(mass, inertiaMatrix)` |
| $q_L = M^{-1} \text{src}$ | `AffineSchurSolver::solve_update(srcC, srcA, cL, AL)` |
| 局部→全局 | `c = c̃ + Ã·cL`, `A = polar(Ã·AL)` |
| 极分解投影 | `polar_rotation(A)` |
| 角速度提取 | `mat_to_angular(A_new, A_old)` |
| 惯性预测 | `stepCpuAffine()` Phase 1 |
| 碰撞检测 | `stepCpuAffine()` Phase 2 |
| 迭代求解 | `stepCpuAffine()` Phase 3 |
| 速度恢复 | `stepCpuAffine()` Phase 4 |
