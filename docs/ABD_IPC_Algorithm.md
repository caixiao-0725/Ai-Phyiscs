# Affine Body Dynamics with IPC Contact (ABD-IPC)

ChysX 实现的 Affine Body Dynamics 算法文档。该实现参照 [libuipc](https://github.com/BeTa-tools/libuipc) 的 `hello_affine_body` 示例和 ABD 论文 (Lei et al., 2022)。

---

## 1. 状态表示

每个刚体用 **12 自由度 (DOF)** 的广义坐标 $\mathbf{q} \in \mathbb{R}^{12}$ 描述：

$$
\mathbf{q} = \begin{bmatrix} \mathbf{t} \\ \mathbf{a}_1 \\ \mathbf{a}_2 \\ \mathbf{a}_3 \end{bmatrix}_{12 \times 1}
$$

其中：
- $\mathbf{t} \in \mathbb{R}^3$：平移向量
- $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3 \in \mathbb{R}^3$：仿射矩阵 $A$ 的三个行向量

世界坐标中的顶点位置由 Jacobi 矩阵映射：

$$
\mathbf{x} = \mathbf{J}(\bar{\mathbf{x}}) \, \mathbf{q}
= \mathbf{t} + \begin{bmatrix} \bar{\mathbf{x}}^\top \mathbf{a}_1 \\ \bar{\mathbf{x}}^\top \mathbf{a}_2 \\ \bar{\mathbf{x}}^\top \mathbf{a}_3 \end{bmatrix}
$$

其中 $\bar{\mathbf{x}}$ 是顶点在参考构型中的坐标。Jacobi 矩阵 $\mathbf{J}$ 为 $3 \times 12$：

$$
\mathbf{J} = \begin{bmatrix}
1 & 0 & 0 & \bar{x}_1 & \bar{x}_2 & \bar{x}_3 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & \bar{x}_1 & \bar{x}_2 & \bar{x}_3 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & \bar{x}_1 & \bar{x}_2 & \bar{x}_3
\end{bmatrix}
$$

---

## 2. 质量矩阵

质量矩阵 $\mathbf{M} \in \mathbb{R}^{12 \times 12}$ 通过对四面体网格体积积分得到：

$$
\mathbf{M} = \sum_i \mathbf{J}_i^\top m_i \mathbf{J}_i
= \int_\Omega \rho \, \mathbf{J}^\top \mathbf{J} \, dV
$$

采用 **Dyadic Mass** 紧凑存储，只需三个量：
- 总质量：$m = \int_\Omega \rho \, dV$
- 质量加权质心矩：$m\bar{\mathbf{x}} = \int_\Omega \rho \, \bar{\mathbf{x}} \, dV$
- 质量加权二阶矩：$m\bar{\mathbf{x}}\bar{\mathbf{x}}^\top = \int_\Omega \rho \, \bar{\mathbf{x}}\bar{\mathbf{x}}^\top \, dV$

对单个四面体（顶点 $p_0, p_1, p_2, p_3$，行列式 $D = \mathbf{e}_1 \cdot (\mathbf{e}_2 \times \mathbf{e}_3)$），积分公式：

$$
\int dV = \frac{D}{6}, \quad
\int \bar{x}_i \, dV = \frac{D}{24} \sum_{k=0}^{3} p_k(i), \quad
\int \bar{x}_i \bar{x}_j \, dV = \frac{D}{120} \left( 2\sum_k p_k(i) p_k(j) + \sum_{k \ne l} p_k(i) p_l(j) \right)
$$

12×12 质量矩阵的块结构：

$$
\mathbf{M} = \begin{bmatrix}
m \mathbf{I}_3 & m\bar{\mathbf{x}}^\top & & \\
m\bar{\mathbf{x}} & m\bar{\mathbf{x}}\bar{\mathbf{x}}^\top & & \\
& & m\bar{\mathbf{x}}\bar{\mathbf{x}}^\top & \\
& & & m\bar{\mathbf{x}}\bar{\mathbf{x}}^\top
\end{bmatrix}
$$

交叉耦合项出现在 $(k, 3+3k)$ 位置，三个 $3 \times 3$ 对角块相同。

---

## 3. 时间积分 (BDF1)

采用一阶后向差分 (BDF1) 的隐式 Euler 格式。

### 3.1 预测步

$$
\tilde{\mathbf{q}} = \mathbf{q}^{n} + \Delta t \, \dot{\mathbf{q}} + \Delta t^2 \, \mathbf{g}_{\text{abd}}
$$

其中广义重力加速度 $\mathbf{g}_{\text{abd}} = \mathbf{M}^{-1} \mathbf{F}_{\text{body}}$，广义体力为：

$$
\mathbf{F}_{\text{body}} = \int_\Omega \mathbf{J}^\top \, (\rho \, \mathbf{g}) \, dV
= \begin{bmatrix}
\mathbf{f} \cdot V \\
f_x \cdot \mathbf{Q}_s \\
f_y \cdot \mathbf{Q}_s \\
f_z \cdot \mathbf{Q}_s
\end{bmatrix}, \quad
\mathbf{Q}_s = \int_\Omega \bar{\mathbf{x}} \, dV
$$

$\mathbf{f} = \rho \mathbf{g}$ 是体力密度，$V$ 是四面体体积，$\mathbf{Q}_s$ 是 rest mesh 的坐标矩。

### 3.2 增量势能最小化

每步求解以下能量最小化问题：

$$
\mathbf{q}^{n+1} = \arg\min_{\mathbf{q}} \; E_{\text{total}}(\mathbf{q})
$$

$$
E_{\text{total}} = E_{\text{kinetic}} + E_{\text{shape}} + E_{\text{contact}}
$$

### 3.3 速度更新

$$
\dot{\mathbf{q}}^{n+1} = \frac{\mathbf{q}^{n+1} - \mathbf{q}^{n}}{\Delta t}
$$

---

## 4. 能量项

### 4.1 动能（惯性势能）

$$
E_{\text{kinetic}} = \frac{1}{2} (\mathbf{q} - \tilde{\mathbf{q}})^\top \mathbf{M} (\mathbf{q} - \tilde{\mathbf{q}})
$$

梯度和 Hessian：

$$
\nabla_{\mathbf{q}} E_{\text{kin}} = \mathbf{M}(\mathbf{q} - \tilde{\mathbf{q}}), \quad
\nabla^2_{\mathbf{q}} E_{\text{kin}} = \mathbf{M}
$$

### 4.2 形状保持能量 (OrthoPotential)

惩罚仿射矩阵 $A$ 偏离旋转群 SO(3)：

$$
\Psi(A) = \| A A^\top - I \|_F^2
= \sum_{i} (|\mathbf{a}_i|^2 - 1)^2 + 2 \sum_{i < j} (\mathbf{a}_i \cdot \mathbf{a}_j)^2
$$

带时间步缩放的完整形状能量：

$$
E_{\text{shape}} = \kappa \cdot V \cdot \Delta t^2 \cdot \Psi(A)
$$

**关键**：乘以 $\Delta t^2$ 是因为在增量势能框架中，所有能量项需要与动能在同一量纲下比较。动能 $E_{\text{kin}}$ 已经隐含了 $\Delta t^2$ 的缩放（通过 $\tilde{\mathbf{q}}$ 的构造），因此弹性能量也需要匹配。

梯度是 9 维向量（仅对 $\mathbf{q}[3{:}12]$ 有贡献），嵌入到 12 维后前 3 个分量为零：

$$
\nabla_{\mathbf{q}} E_{\text{shape}} = \kappa V \Delta t^2 \cdot \begin{bmatrix} \mathbf{0}_3 \\ \nabla_{\mathbf{a}} \Psi \end{bmatrix}
$$

Hessian 为 $12 \times 12$，左上 $3 \times 3$ 块为零，右下 $9 \times 9$ 为 $\nabla^2_{\mathbf{a}} \Psi$ 经 SPD 投影后的结果。

实现中使用 libuipc 的 SymEigen 符号计算自动生成的解析公式。

### 4.3 接触能量 (IPC Log-Barrier)

IPC (Incremental Potential Contact) 使用对数障碍函数处理碰撞：

$$
E_{\text{contact}} = \kappa_c \sum_{k} B(D_k)
$$

其中 $D = d^2$ 是几何元素间的平方距离，$\hat{D} = \hat{d}^2$ 是激活阈值。障碍函数：

$$
B(D) = \begin{cases}
-(D - \hat{D})^2 \ln\!\left(\dfrac{D}{\hat{D}}\right) & 0 < D < \hat{D} \\[6pt]
0 & D \geq \hat{D}
\end{cases}
$$

其一阶和二阶导数：

$$
\frac{dB}{dD} = -(D - \hat{D}) \left( 2 \ln\frac{D}{\hat{D}} + \frac{D - \hat{D}}{D} \right)
$$

$$
\frac{d^2 B}{dD^2} = -\left( 2 \ln\frac{D}{\hat{D}} + 3 - 2\frac{\hat{D}}{D} - \frac{\hat{D}^2}{D^2} \right)
$$

性质：
- $B(D) \to +\infty$ 当 $D \to 0^+$（无穿透保证）
- $B(\hat{D}) = 0$，$B'(\hat{D}) = 0$（C1 连续衔接）
- $B'(D) < 0$ 对所有 $0 < D < \hat{D}$（排斥力）

---

## 5. 距离基元

支持四种几何元素间的平方距离计算：

| 类型 | 符号 | 维度 | 说明 |
|------|------|------|------|
| Point–Point | PP | 6 | $D = \lVert\mathbf{p}_0 - \mathbf{p}_1\rVert^2$ |
| Point–Edge | PE | 9 | 点到线段最近点的平方距离 |
| Point–Triangle | PT | 12 | 点到三角形最近点的平方距离 |
| Edge–Edge | EE | 12 | 两线段间最近点对的平方距离 |

每种类型都实现了：
- 距离值 `dist2_*()`
- 梯度 `dist2_*_grad()`（对每个顶点坐标的偏导）
- 接触类型分类 `classify_*()`（退化情况处理：PT 退化为 PE/PP，EE 退化为 PE/PP）

---

## 6. 从世界坐标到广义坐标的映射

接触力在 **世界空间** 中作用于顶点，需要通过 Jacobi 矩阵提升到 12-DOF $\mathbf{q}$ 空间。

### 6.1 梯度提升

对体 $k$ 上的顶点 $i$，其接触能量梯度：

$$
\frac{\partial E_c}{\partial \mathbf{q}_k}
= \kappa_c \frac{dB}{dD} \cdot \mathbf{J}_i^\top \frac{\partial D}{\partial \mathbf{x}_i}
$$

$\mathbf{J}^\top$ 将 3D 力映射为 12D 广义力：

$$
\mathbf{J}^\top \mathbf{g} = \begin{bmatrix}
\mathbf{g} \\
g_x \bar{\mathbf{x}} \\
g_y \bar{\mathbf{x}} \\
g_z \bar{\mathbf{x}}
\end{bmatrix}_{12 \times 1}
$$

### 6.2 Hessian 提升

完整的接触 Hessian 包含两项：

$$
\frac{\partial^2 E_c}{\partial \mathbf{q}_i \partial \mathbf{q}_j}
= \kappa_c \left[
\underbrace{\frac{d^2 B}{dD^2} \left(\mathbf{J}_i^\top \frac{\partial D}{\partial \mathbf{x}_i}\right) \left(\mathbf{J}_j^\top \frac{\partial D}{\partial \mathbf{x}_j}\right)^\top}_{\text{rank-1 barrier curvature}}
+
\underbrace{\frac{dB}{dD} \cdot \mathbf{J}_i^\top \frac{\partial^2 D}{\partial \mathbf{x}_i \partial \mathbf{x}_j} \mathbf{J}_j}_{\text{distance Hessian}}
\right]
$$

$\mathbf{J}_i^\top \mathbf{H}_{3 \times 3} \mathbf{J}_j$ 的 12×12 块结构由 Kronecker 乘积给出：

$$
\mathbf{J}_i^\top \mathbf{H} \mathbf{J}_j = \begin{bmatrix}
\mathbf{H} & H_{\cdot 1} \bar{\mathbf{y}}^\top & H_{\cdot 2} \bar{\mathbf{y}}^\top & H_{\cdot 3} \bar{\mathbf{y}}^\top \\
\bar{\mathbf{x}} H_{1\cdot} & H_{11} \bar{\mathbf{x}} \bar{\mathbf{y}}^\top & H_{12} \bar{\mathbf{x}} \bar{\mathbf{y}}^\top & H_{13} \bar{\mathbf{x}} \bar{\mathbf{y}}^\top \\
\bar{\mathbf{x}} H_{2\cdot} & H_{21} \bar{\mathbf{x}} \bar{\mathbf{y}}^\top & H_{22} \bar{\mathbf{x}} \bar{\mathbf{y}}^\top & H_{23} \bar{\mathbf{x}} \bar{\mathbf{y}}^\top \\
\bar{\mathbf{x}} H_{3\cdot} & H_{31} \bar{\mathbf{x}} \bar{\mathbf{y}}^\top & H_{32} \bar{\mathbf{x}} \bar{\mathbf{y}}^\top & H_{33} \bar{\mathbf{x}} \bar{\mathbf{y}}^\top
\end{bmatrix}
$$

其中 $\bar{\mathbf{x}}, \bar{\mathbf{y}}$ 分别是两个顶点的 rest 坐标，$H_{\cdot k}$ 表示 $\mathbf{H}$ 的第 $k$ 列，$H_{k\cdot}$ 表示第 $k$ 行。

---

## 7. Newton 求解器

### 7.1 整体流程

```
for each time step:
    1. q_prev ← q
    2. 预测: q̃ = q + Δt·q̇ + Δt²·g_abd
    3. CCD 截断预测步（防止穿透）
    4. Newton 迭代:
       a. 更新表面顶点位置
       b. 检测碰撞对 (D < d̂)
       c. 组装线性系统 H·Δq = -g
       d. PCG 求解 Δq
       e. 收敛检查
       f. CCD 步长限制
       g. Armijo 线搜索
       h. 更新 q ← q + α·Δq
    5. 速度更新: q̇ = (q - q_prev) / Δt
```

### 7.2 线性系统

全局线性系统 $\mathbf{H} \Delta\mathbf{q} = -\mathbf{g}$ 具有块稀疏结构：

- **对角块** $\mathbf{H}_{ii} = \mathbf{M}_i + \nabla^2 E_{\text{shape},i} + \sum_{\text{contacts}} \mathbf{H}_{\text{contact},ii}$
- **非对角块** $\mathbf{H}_{ij} = \sum_{\text{contacts}} \mathbf{H}_{\text{contact},ij}$（来自跨体碰撞耦合）

使用 **预处理共轭梯度法 (PCG)** 求解，预处理矩阵为对角块的逆：

$$
\mathbf{P}_i = \mathbf{H}_{ii}^{-1}
$$

### 7.3 收敛判据

当广义速度增量的最大分量小于阈值时认为收敛：

$$
\max_i \| \Delta \mathbf{q}_i \|_\infty < v_{\text{tol}} \cdot \Delta t
$$

---

## 8. 碰撞处理

### 8.1 碰撞检测 (DCD)

在每次 Newton 迭代开始时检测所有 **跨体** 的 PT 和 EE 对，筛选条件为 $D < \hat{D}$。

当前实现使用暴力枚举（$O(n^2)$），适用于少量刚体。后续可接入 ChysX 的 `QuantBvh` broadphase 加速。

### 8.2 连续碰撞检测 (CCD)

CCD 用于两个阶段：

1. **预测步截断**：防止 BDF1 predictor 穿透现有几何体
2. **线搜索步长限制**：在 Newton 迭代的线搜索中限制最大步长

算法：**轨迹采样 + 二分精化**

```
function ccd_pt(p, dp, t0..t2, dt0..dt2, d_min):
    对 k = 1..N_samples:
        t = k / N_samples
        计算插值位置: p(t), t0(t), t1(t), t2(t)
        D = dist2_pt(p(t), t0(t), t1(t), t2(t))
        if D < d_min²:
            lo = (k-1)/N_samples, hi = t
            二分 10 次精化:
                mid = (lo + hi) / 2
                if D(mid) < d_min² then hi = mid
                else lo = mid
            return lo  // 最后一个安全 α
    return 1.0  // 无碰撞
```

`d_min` 的选择策略：
- **预测步 CCD**：`d_min = 0.5 · d_hat`（允许物体进入 barrier 激活范围）
- **线搜索 CCD**：`d_min = 0.1 · d_hat`（更宽松，让 barrier 力来处理排斥）

### 8.3 线搜索 (Armijo Backtracking)

在 CCD 限制的步长范围内，使用 Armijo 回溯法确保能量下降：

$$
E(\mathbf{q} + \alpha \Delta\mathbf{q}) \leq E(\mathbf{q}) + c \cdot \alpha \cdot \mathbf{g}^\top \Delta\mathbf{q}
$$

其中 $c = 10^{-4}$。初始 $\alpha$ 为 CCD 返回的安全步长，逐次减半直到满足 Armijo 条件或达到最大迭代次数。

**重要**：线搜索中的能量评估会 **重新检测碰撞对**，而不是使用 Newton 迭代开始时的旧列表。这防止了因 $\mathbf{q}$ 改变后某些碰撞对距离跳出 $\hat{d}$ 范围而导致错误接受穿透步。

---

## 9. 模块结构

```
src/rigid/abd_ipc/
├── abd_ipc_types.cuh      # 基础类型: q向量, ABDJacobi, DyadicMass, ABDBody, ContactPair, ABDConfig
├── abd_ipc_mesh.h          # 网格工具: dyadic mass 积分, 广义体力, body 初始化
├── abd_ipc_energy.cuh      # OrthoPotential: Ψ, ∇Ψ, ∇²Ψ (符号计算公式)
├── abd_ipc_barrier.cuh     # IPC log-barrier: B, dB/dD, d²B/dD²
├── abd_ipc_distance.cuh    # 平方距离基元: PP, PE, PT, EE 及其梯度
├── abd_ipc_assembly.cuh    # 线性系统组装: body gradient/hessian, contact lifting
├── abd_ipc_contact.h       # 碰撞检测: brute-force PT + EE
├── abd_ipc_ccd.cuh         # 连续碰撞检测 + 线搜索
├── abd_ipc_pcg.cuh         # 预处理共轭梯度求解器
└── abd_ipc_solver.h        # 主求解器: Newton loop, BDF1 integration
```

---

## 10. 参考

1. **ABD 论文**: Lei, M., et al. "Affine Body Dynamics: Fast, Stable & Intersection-free Simulation of Stiff Materials." (2022)
2. **IPC**: Li, M., et al. "Incremental Potential Contact: Intersection-and Inversion-Free Large-Deformation Dynamics." (2020)
3. **libuipc**: https://github.com/BeTa-tools/libuipc — 参考实现
