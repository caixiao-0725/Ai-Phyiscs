# 论文 PABD 的 D 矩阵为什么是块主对角的？与我们的实现差异

## 核心结论

论文 PABD 中的 $D$ 矩阵确实是**块主对角**的（3 个相同的 3×3 块），总共只需要 13 个标量。这不是近似，而是其 Jacobian 结构的**精确结果**。我们的实现中 D 变成了一般 9×9（非块主对角），原因是 **Jacobian 的定义方式不同**。

---

## 1. 论文中的 Jacobian 结构

论文公式 (28)，Jacobian 是 3×12 矩阵：

$$
J = \begin{bmatrix}
1 & 0 & 0 & x & y & z & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & x & y & z & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & x & y & z
\end{bmatrix}
$$

关键观察：**每一行的 $(x,y,z)$ 块只出现在不同的 3 列组中**。第 1 行占第 4-6 列，第 2 行占第 7-9 列，第 3 行占第 10-12 列。

### 计算 $J^\top J$ 的 A-A 部分（9×9 右下角）

$J$ 的 A 部分按行分组：

$$
J_A = \begin{bmatrix}
\mathbf{x}^\top & \mathbf{0}^\top & \mathbf{0}^\top \\
\mathbf{0}^\top & \mathbf{x}^\top & \mathbf{0}^\top \\
\mathbf{0}^\top & \mathbf{0}^\top & \mathbf{x}^\top
\end{bmatrix}
= I_3 \otimes \mathbf{x}^\top
$$

$$
J_A^\top J_A = (I_3 \otimes \mathbf{x}^\top)^\top (I_3 \otimes \mathbf{x}^\top) = I_3 \otimes (\mathbf{x}\,\mathbf{x}^\top)
= \begin{bmatrix}
\mathbf{x}\mathbf{x}^\top & 0 & 0 \\
0 & \mathbf{x}\mathbf{x}^\top & 0 \\
0 & 0 & \mathbf{x}\mathbf{x}^\top
\end{bmatrix}
$$

**天然块主对角**，3 个块完全相同。

### 求和后

$$
D = \Pi + h^2 \sum_\xi \frac{\psi^+_\xi}{d}\,\mathbf{x}_\xi\,\mathbf{x}_\xi^\top
$$

仍然只是一个 3×3 矩阵，重复 3 次出现在对角线上。论文公式 (29) 正是这个结构。

### 为什么 13 个标量就够？

$$
H_i \text{ 由 } (a,\;\mathbf{b},\;D) \text{ 完全描述} = 1 + 3 + 9 = 13 \text{ 个标量}
$$

---

## 2. 论文中 bond 力的本质

论文公式 (21) 的 Hessian：

$$
H_i = M_i + h^2 \sum_\xi \frac{\psi^+_\xi(d)}{d}\,J_\xi^\top J_\xi
$$

这里 $\psi^+_\xi(d)/d$ 是一个**标量**——bond 的力沿着 $\mathbf{n}_\xi$ 方向，但其刚度系数是**各向同性的**（标量乘以 $J^\top J$）。换句话说，bond 在所有 3 个维度上贡献相同的刚度 $\psi^+/d$。

**这就是为什么 D 是块主对角的**：$J^\top J$ 的 Kronecker 结构 $I_3 \otimes (\mathbf{x}\mathbf{x}^\top)$ 在乘以标量后保持不变。

---

## 3. 我们的实现为什么不同

我们的约束不是论文中的"bond"（两个 body 之间的距离约束），而是**方向性接触约束**（法向 + 切向）。每个方向 $d$ 的约束：

$$
C_d = \tilde{\mathbf{n}}_d^\top (\mathbf{c}_L + \mathbf{A}_L\,\mathbf{r}) + \xi_d
$$

其 Jacobian 是 1×12 行向量（不是 3×12）：

$$
\tilde{J}_d = [\tilde{\mathbf{n}}_d^\top;\; \tilde{\mathbf{n}}_d^\top\,r_0;\; \tilde{\mathbf{n}}_d^\top\,r_1;\; \tilde{\mathbf{n}}_d^\top\,r_2]
$$

Hessian 贡献 $\kappa_d\,\tilde{J}_d^\top\,\tilde{J}_d$ 的 A-A 部分（9×9）：

$$
(\tilde{J}_d^\top\,\tilde{J}_d)_{AA}[j_1][j_2] = \kappa_d\,(\tilde{\mathbf{n}}_d \otimes \tilde{\mathbf{n}}_d)\,r_{j_1}\,r_{j_2}
$$

这里每个 3×3 子块是 $\kappa_d\,(\tilde{\mathbf{n}}_d \otimes \tilde{\mathbf{n}}_d)\,r_{j_1}\,r_{j_2}$。当 $j_1 \neq j_2$ 时（即非块主对角位置），**只有 $r_{j_1}\,r_{j_2} = 0$ 才为零**。

一般情况下 $r_{j_1}\,r_{j_2} \neq 0$（例如顶点 $\mathbf{r} = (1, 1, -1)$ 的 $r_0 r_1 = 1 \neq 0$），所以 **D 有非零的非对角块**。

### 和论文方法的对比

| | 论文 PABD | 我们的实现 |
|---|---|---|
| 约束类型 | 距离 bond（3D，各向同性） | 方向接触（1D per direction） |
| Jacobian | 3×12, $J = [I_3,\; I_3\otimes\mathbf{x}]$ | 1×12 per direction, $\tilde{J}_d = [\tilde{\mathbf{n}}_d;\; \tilde{\mathbf{n}}_d \cdot r]$ |
| $J^\top J$ 结构 | $I_3 \otimes (\mathbf{x}\mathbf{x}^\top)$（块主对角） | $(\tilde{\mathbf{n}}\otimes\tilde{\mathbf{n}}) \otimes (\mathbf{r}\mathbf{r}^\top)$（一般 9×9） |
| D 是块主对角？ | **是**（精确） | **否** |
| 存储 | 13 float | 117 float |

---

## 4. 为什么我们不能直接用论文的方法？

论文的 bond 是 **3D 各向同性** 的：一个 bond 在所有 3 个方向（x, y, z）贡献相同的刚度 $\psi^+/d$。这对应于我们代码中的 `accumulate(kappa, r)`，即 $\kappa \cdot I_3$ 的 Hessian 贡献。

**但接触约束天然是各向异性的**：法向和切向的 penalty $\kappa_d$ 可能完全不同。用 3D 各向同性的 bond 来模拟接触，有两种选择：

### 选项 A：各向同性近似（旧代码）

取 $\kappa_{\max} = \max(\kappa_n, \kappa_{t_1}, \kappa_{t_2})$，用一个标量乘以 $I_3$：

```
H 贡献 = κ_max · J^T J    (和论文完全一致)
```

优点：D 是块主对角，13 float，论文的求逆公式直接可用。

缺点：当 $\kappa_n \gg \kappa_t$ 时（摩擦场景），切向方向被人为加了巨大的虚假刚度。

### 选项 B：各向异性精确（新代码）

每个方向用自己的 $\kappa_d$，分别贡献 $\kappa_d \cdot (\tilde{\mathbf{n}}_d \otimes \tilde{\mathbf{n}}_d)$：

```
H 贡献 = Σ_d κ_d · J_d^T J_d
```

优点：精确，摩擦不会引入虚假刚度。

缺点：D 不再块主对角，需要完整 9×9 block Cholesky。

---

## 5. 如果只有法向约束（无摩擦），能恢复块主对角吗？

**不能**，除非在特殊情况下。

只有法向约束时，$\mathbf{M} = \kappa_n \cdot (\tilde{\mathbf{n}} \otimes \tilde{\mathbf{n}})$。D 的 $(j_1, j_2)$ 块 = $\sum \kappa_n \cdot (\tilde{\mathbf{n}} \otimes \tilde{\mathbf{n}}) \cdot r_{j_1} \cdot r_{j_2}$。当 $j_1 \neq j_2$ 且 $r_{j_1} \cdot r_{j_2} \neq 0$ 时，该块非零。

但如果所有接触点的 $\tilde{\mathbf{n}}$ 完全相同（例如所有底面顶点法向都是 $(0,0,1)$），且 $r_{j_1} r_{j_2}$ 关于接触点求和后恰好为零（例如 4 个对称顶点 $(±1, ±1, -1)$），则非对角块**恰好对消**为零。

这正是为什么旧代码在对称场景下"凑巧能用"的原因。

---

## 6. 总结

```
论文 PABD:
  bond = 3D 各向同性距离约束
  J = [I₃, I₃⊗x]   →   J^T J = I₃ ⊗ (xx^T)   →   D 块主对角
  H 用 (a, b, D) = 13 float 描述
  H^{-1} 有封闭公式（论文 Eq. 30）

我们的实现:
  约束 = 1D 方向约束 × 3 方向（法向 + 2切向）
  J_d = [ñ_d, ñ_d·r₀, ñ_d·r₁, ñ_d·r₂]
  J_d^T J_d 的 AA 部分 = (ñ_d⊗ñ_d) ⊗ (rr^T)   →   D 一般 9×9
  H 需要 (sumK, sumB[3], sumD[3][3]) = 117 float
  求解用 Schur complement + block Cholesky
```

论文的块主对角性质是正确的，不是近似。两者的区别在于**约束建模方式不同**：论文用各向同性 bond，我们用各向异性方向约束。

---

## 7. 精确版 Jacobian 与 Hessian 的逐步推导

### 7.1 世界坐标与局部坐标

刚体上一个材料点 $\mathbf{r}$（局部坐标）在世界坐标系中的位置为：

$$
\mathbf{y} = \tilde{\mathbf{c}} + \tilde{\mathbf{A}}\,(\mathbf{c}_L + \mathbf{A}_L\,\mathbf{r})
$$

其中 $(\tilde{\mathbf{c}}, \tilde{\mathbf{A}})$ 是惯性预测的参考状态（常量），$(\mathbf{c}_L, \mathbf{A}_L)$ 是待求的局部偏移量。定义局部位置：

$$
\bar{\mathbf{y}} = \mathbf{c}_L + \mathbf{A}_L\,\mathbf{r}
$$

使得 $\mathbf{y} = \tilde{\mathbf{c}} + \tilde{\mathbf{A}}\,\bar{\mathbf{y}}$。

### 7.2 方向约束的定义

一个接触点在世界坐标方向 $\mathbf{n}_d^w$ 上的约束：

$$
C_d = (\mathbf{n}_d^w)^\top\,\mathbf{y} - t_d
$$

代入 $\mathbf{y}$ 的表达式：

$$
C_d = (\mathbf{n}_d^w)^\top\,\tilde{\mathbf{A}}\,(\mathbf{c}_L + \mathbf{A}_L\,\mathbf{r}) + (\mathbf{n}_d^w)^\top\,\tilde{\mathbf{c}} - t_d
$$

定义**局部方向** $\tilde{\mathbf{n}}_d = \tilde{\mathbf{A}}^\top\,\mathbf{n}_d^w$ 和**常数偏移** $\xi_d = (\mathbf{n}_d^w)^\top\,\tilde{\mathbf{c}} - t_d$：

$$
\boxed{C_d = \tilde{\mathbf{n}}_d^\top\,(\mathbf{c}_L + \mathbf{A}_L\,\mathbf{r}) + \xi_d}
$$

### 7.3 Jacobian（1×12 行向量）

自由度排列为 $\mathbf{q} = [\mathbf{c}_L;\; \text{vec}(\mathbf{A}_L)] \in \mathbb{R}^{12}$，其中 $\text{vec}(\mathbf{A}_L) = [\mathbf{A}_L[0];\; \mathbf{A}_L[1];\; \mathbf{A}_L[2]]$（按行展平）。

$$
\frac{\partial C_d}{\partial \mathbf{c}_L} = \tilde{\mathbf{n}}_d^\top \quad (1 \times 3)
$$

对 $\mathbf{A}_L$ 的第 $j$ 行 $\mathbf{A}_L[j]$ 求导。由于 $\mathbf{A}_L\,\mathbf{r} = \sum_j \mathbf{A}_L[j]\,r_j$：

$$
\frac{\partial C_d}{\partial \mathbf{A}_L[j]} = \tilde{\mathbf{n}}_d^\top \cdot r_j \quad (1 \times 3)
$$

组合成完整的 1×12 Jacobian 行向量：

$$
\boxed{\mathbf{J}_d = \bigl[\tilde{\mathbf{n}}_d^\top,\;\; \tilde{\mathbf{n}}_d^\top r_0,\;\; \tilde{\mathbf{n}}_d^\top r_1,\;\; \tilde{\mathbf{n}}_d^\top r_2\bigr]}
$$

展开为 12 个元素：

$$
\mathbf{J}_d = [\tilde{n}_0,\; \tilde{n}_1,\; \tilde{n}_2,\;\;
\tilde{n}_0 r_0,\; \tilde{n}_1 r_0,\; \tilde{n}_2 r_0,\;\;
\tilde{n}_0 r_1,\; \tilde{n}_1 r_1,\; \tilde{n}_2 r_1,\;\;
\tilde{n}_0 r_2,\; \tilde{n}_1 r_2,\; \tilde{n}_2 r_2]
$$

**与论文 PABD 的 3×12 Jacobian 对比**：论文的 $J = [I_3,\; I_3 \otimes \mathbf{x}]$ 是 3 行（对应世界空间 3D 力）；我们的 $\mathbf{J}_d$ 只有 1 行（对应一个标量方向约束）。PABD 的一次 bond 贡献等效于我们沿 $\mathbf{e}_x, \mathbf{e}_y, \mathbf{e}_z$ 三个方向各调用一次 `accumulate_dir`，且三个方向的 $\kappa$ 相同。

### 7.4 Hessian（12×12 对称矩阵）

单个方向约束的 Gauss-Newton Hessian 贡献为：

$$
\mathbf{H}_d = \kappa_d\,\mathbf{J}_d^\top\,\mathbf{J}_d \quad (12 \times 12)
$$

将 $\mathbf{J}_d^\top \mathbf{J}_d$ 按 $(3 + 3 + 3 + 3)$ 的块结构展开：

$$
\mathbf{J}_d^\top \mathbf{J}_d = \begin{bmatrix}
\tilde{\mathbf{n}}_d \\
\tilde{\mathbf{n}}_d\,r_0 \\
\tilde{\mathbf{n}}_d\,r_1 \\
\tilde{\mathbf{n}}_d\,r_2
\end{bmatrix}
\begin{bmatrix}
\tilde{\mathbf{n}}_d^\top &
\tilde{\mathbf{n}}_d^\top r_0 &
\tilde{\mathbf{n}}_d^\top r_1 &
\tilde{\mathbf{n}}_d^\top r_2
\end{bmatrix}
$$

$$
= \begin{bmatrix}
\mathbf{N} & \mathbf{N}\,r_0 & \mathbf{N}\,r_1 & \mathbf{N}\,r_2 \\
\mathbf{N}\,r_0 & \mathbf{N}\,r_0^2 & \mathbf{N}\,r_0 r_1 & \mathbf{N}\,r_0 r_2 \\
\mathbf{N}\,r_1 & \mathbf{N}\,r_1 r_0 & \mathbf{N}\,r_1^2 & \mathbf{N}\,r_1 r_2 \\
\mathbf{N}\,r_2 & \mathbf{N}\,r_2 r_0 & \mathbf{N}\,r_2 r_1 & \mathbf{N}\,r_2^2
\end{bmatrix}
$$

其中 $\mathbf{N} = \tilde{\mathbf{n}}_d \otimes \tilde{\mathbf{n}}_d$（3×3 对称秩1矩阵）。

**关键观察**：所有 4×4 = 16 个 3×3 子块都是**同一个矩阵 $\mathbf{N}$** 乘以 $r_{j_1} \cdot r_{j_2}$ 的标量。因此整个 12×12 矩阵可以分解为：

$$
\mathbf{J}_d^\top \mathbf{J}_d = \underbrace{(\tilde{\mathbf{n}}_d \otimes \tilde{\mathbf{n}}_d)}_{\mathbf{N},\; 3\times3} \otimes \underbrace{\begin{bmatrix} 1 \\ r_0 \\ r_1 \\ r_2 \end{bmatrix} \begin{bmatrix} 1 & r_0 & r_1 & r_2 \end{bmatrix}}_{\mathbf{w}\mathbf{w}^\top,\;4\times4}
$$

其中 $\mathbf{w} = [1, r_0, r_1, r_2]^\top$。

> **与 PABD 的关键区别**：PABD 中 $J^\top J = I_3 \otimes (\mathbf{x}\mathbf{x}^\top)$，这里 $I_3$ 是固定的单位矩阵，所以行之间不耦合。我们的 $\mathbf{N} = \tilde{\mathbf{n}} \otimes \tilde{\mathbf{n}}$ 是一般的秩1矩阵（不是 $I_3$），所以行之间产生耦合。

### 7.5 累加后的 Hessian 分块

对所有接触点的所有方向求和后，总 Hessian $\mathbf{H} = \mathbf{H}_{\text{inertia}} + \sum_{\text{contacts}} \sum_d \mathbf{H}_d$，其 12×12 分块为：

$$
\mathbf{H} = \begin{bmatrix}
\mathbf{K} & \mathbf{B}_0 & \mathbf{B}_1 & \mathbf{B}_2 \\
\mathbf{B}_0^\top & \mathbf{D}_{00} & \mathbf{D}_{01} & \mathbf{D}_{02} \\
\mathbf{B}_1^\top & \mathbf{D}_{10} & \mathbf{D}_{11} & \mathbf{D}_{12} \\
\mathbf{B}_2^\top & \mathbf{D}_{20} & \mathbf{D}_{21} & \mathbf{D}_{22}
\end{bmatrix}
$$

各块公式（$\mathbf{M}_i = \sum_d \kappa_d\,(\tilde{\mathbf{n}}_d \otimes \tilde{\mathbf{n}}_d)$ 表示第 $i$ 个接触点的方向刚度矩阵）：

$$
\mathbf{K} = m\,\mathbf{I}_3 + \sum_i \mathbf{M}_i, \qquad
\mathbf{B}_j = \sum_i \mathbf{M}_i \cdot r_j^{(i)}
$$

$$
\mathbf{D}_{j_1 j_2} = \delta_{j_1 j_2}\,\mathbf{J}_{\text{inertia}} + \sum_i \mathbf{M}_i \cdot r_{j_1}^{(i)} \cdot r_{j_2}^{(i)}
$$

**对称性分析**：

| 矩阵 | 内部对称性 | 外部对称性 |
|-------|-----------|-----------|
| $\mathbf{K}$ | $\mathbf{K} = \mathbf{K}^\top$（3×3 对称） | — |
| $\mathbf{B}_j$ | $\mathbf{B}_j = \mathbf{B}_j^\top$（因为 $\mathbf{M}_i$ 对称） | — |
| $\mathbf{D}_{j_1 j_2}$ | $\mathbf{D}_{j_1 j_2} = \mathbf{D}_{j_1 j_2}^\top$（3×3 对称） | $\mathbf{D}_{j_1 j_2} = \mathbf{D}_{j_2 j_1}$（块对称） |

---

## 8. 最小寄存器/存储量分析

### 8.1 累加阶段

需要存储 `sumK`、`sumB[3]`、`sumD[3][3]` 的独立元素数量：

| 量 | 矩阵形状 | 内部对称 | 外部对称 | 独立元素 |
|----|----------|---------|---------|---------|
| `sumK` | 1 个 3×3 对称 | 6 | — | **6** |
| `sumB[3]` | 3 个 3×3 对称 | 每个 6 | 无（$j=0,1,2$ 互不相关） | **18** |
| `sumD[3][3]` | 3×3 个 3×3 对称 | 每个 6 | $\mathbf{D}_{j_1 j_2} = \mathbf{D}_{j_2 j_1}$ | **36** |

> `sumD` 详细计算：9 个块中，对角 3 个 + 上三角 3 个 = 6 个独立块 × 6 float/块 = **36**。

**累加阶段总计：6 + 18 + 36 = 60 float**（当前代码用 117，可以砍掉几乎一半）

### 8.2 分解阶段

分解需要存储的中间/最终结果：

| 量 | 用途 | 形状 | 对称性 | 独立元素 |
|----|------|------|--------|---------|
| $\mathbf{K}^{-1}$ | Schur 回代 | 3×3 对称 | 6 | **6** |
| $\mathbf{B}[3]$ | Schur 回代 | 3 个 3×3 对称 | 每个 6 | **18** |
| $\mathbf{L}[3][3]$ | block Cholesky 因子 | 下三角（6 块），每块 3×3 | $\mathbf{L}_{ii}$ 下三角(6)，$\mathbf{L}_{ij}$ 一般(9) | **6+6+9+6+9+9 = 45** |

> $\mathbf{L}$ 详细计算：
> - $\mathbf{L}_{00}$：3×3 下三角 → 6 float
> - $\mathbf{L}_{11}$：3×3 下三角 → 6 float
> - $\mathbf{L}_{22}$：3×3 下三角 → 6 float
> - $\mathbf{L}_{10}$：一般 3×3 → 9 float
> - $\mathbf{L}_{20}$：一般 3×3 → 9 float
> - $\mathbf{L}_{21}$：一般 3×3 → 9 float
> - 合计：18 + 27 = **45**

**分解阶段总计：6 + 18 + 45 = 69 float**

### 8.3 累加 + 分解可共享存储

累加结束后，`sumK`、`sumB`、`sumD` 不再需要（信息已转移到 $\mathbf{K}^{-1}$、$\mathbf{B}$、$\mathbf{L}$）。如果允许 in-place 覆写：

- `sumK`（6）→ 被 $\mathbf{K}^{-1}$（6）覆写 ✓
- `sumB[3]`（18）→ 被 $\mathbf{B}[3]$（18）覆写（相同数据，直接复用）✓
- `sumD[3][3]`（36）→ 被 $\mathbf{L}[3][3]$（45）覆写 ✗（L 比 sumD 大 9 个）

需要额外 45 - 36 = 9 float 给 L 的溢出部分。

**共享后峰值：6 + 18 + 45 + 临时 = 69 float + 少量临时变量**

### 8.4 求解阶段

求解只需要读取 $\mathbf{K}^{-1}$、$\mathbf{B}$、$\mathbf{L}$（已存储），加上：

- `srcC`（3 float）、`srcA`（9 float）：RHS
- `rhs_A[3]`（9 float）：Schur 修正后的 RHS
- `y[3]`、`A_cols[3]`（各 9 float）：前代/回代中间变量
- `cL`（3 float）、`AL`（9 float）：输出

这些都是临时变量，不计入 persistent 存储。

### 8.5 总结对比

| | 论文 PABD | 当前代码 | 利用对称性后（最优） |
|---|---|---|---|
| 累加器 | 13 float | 117 float | **60 float** |
| 分解存储 | 13 float | 117 float | **69 float** |
| 峰值（共享） | 13 float | 234 float | **~78 float** |
| GPU 寄存器安全？ | ✓（远低于 255） | ✗（可能溢出） | ✓（安全） |

利用对称性可以将存储量从 117 降到 60（累加）/ 69（分解），减少约 **47%~41%**。峰值约 78 float，远低于 GPU 的 255 寄存器限制。
