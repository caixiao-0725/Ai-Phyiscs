# ChysX 布料 VBD 算法文档

## 1. 算法概述

VBD（Vertex Block Descent）是一种基于**逐顶点块坐标下降**的隐式时间积分求解器。
与传统的全局 PCG 求解器不同，VBD 将全局非线性优化问题分解为每个顶点上的独立 3×3 局部线性系统，通过多轮 Gauss-Seidel 迭代求解收敛。

为了保证并行 GPU 执行中不出现数据竞争，VBD 使用**图着色**（Graph Coloring）将顶点分组：同一颜色组内的顶点之间不共享任何弹性元素（三角形或弯曲边），因此可以安全地并行更新。

### 优化目标

VBD 最小化的增量势能（Incremental Potential）：

$$
E(\mathbf{x}) = \sum_i \frac{m_i}{2\Delta t^2} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2 + \sum_t \Psi_{\text{membrane}}^{(t)}(\mathbf{x}) + \sum_e \Psi_{\text{bending}}^{(e)}(\mathbf{x})
$$

其中 $\hat{\mathbf{x}}_i = \mathbf{x}_i^n + \Delta t \, \mathbf{v}_i^n + \Delta t^2 \, \mathbf{g}$ 是惯性预测位置。

---

## 2. 求解流程（单 substep）

每个 substep 由 3 个阶段组成：

```
Phase 1: vbd_forward_step_kernel         // 惯性预测
Phase 2: for iter = 0..iterations-1:      // VBD 迭代
           for color = 0..num_colors-1:
             vbd_cloth_solve_kernel        // 按颜色组并行求解
Phase 3: vbd_update_velocity_kernel       // 速度更新
```

---

## 3. 核函数详解

### 3.1 `vbd_forward_step_kernel`

**作用**：惯性预测 — 保存当前位置快照，计算惯性目标位置。

| 输入 | 说明 |
|------|------|
| `pos[i]` | 当前位置 $\mathbf{x}_i^n$ |
| `vel[i]` | 当前速度 $\mathbf{v}_i^n$ |
| `inv_mass[i]` | 逆质量 $w_i$（=0 表示 pin） |
| `gravity` | 重力加速度 |
| `dt` | 时间步长 |

| 输出 | 说明 |
|------|------|
| `q_prev[i]` | 位置快照 $\mathbf{x}_i^n$（用于之后恢复和速度计算） |
| `inertia[i]` | 惯性目标 $\hat{\mathbf{x}}_i = \mathbf{x}_i^n + (\mathbf{v}_i^n + \mathbf{g} \Delta t) \Delta t$ |
| `mass[i]` | 质量 $m_i = 1/w_i$ |
| `displacements[i]` | 初始位移 $\mathbf{d}_i = (\mathbf{v}_i^n + \mathbf{g} \Delta t) \Delta t$ |
| `pos[i]` | 更新为惯性位置 $\hat{\mathbf{x}}_i$（作为 VBD 迭代的起点） |

**固定顶点处理**：若 $w_i = 0$，则 `inertia[i] = pos[i]`，`displacements[i] = 0`，位置不变。

**线程映射**：每线程一个顶点，共 `n_particles` 个线程。

---

### 3.2 `vbd_cloth_solve_kernel`

**作用**：VBD 核心 — 对当前颜色组中的每个顶点执行局部 3×3 求解。

**线程映射**：每线程一个顶点（仅当前颜色组），共 `n_color` 个线程。

**算法步骤**：

```
对于颜色组中的每个顶点 pi：
  1. 初始化：f = m/dt² × (inertia[pi] - pos[pi]),  H = m/dt² × I₃
  2. 遍历 pi 相邻的所有三角形 → 累加膜能量 force 和 hessian
  3. 遍历 pi 相邻的所有弯曲边 → 累加弯曲能量 force 和 hessian
  4. 局部求解：displacement[pi] += H⁻¹ × f
  5. 更新位置：pos[pi] = q_prev[pi] + displacement[pi]
```

核函数内部调用两个 device 函数完成能量计算：

#### 3.2.1 `cloth_vbd_membrane_force_hessian`

**来源**：Newton `evaluate_neo_hookean_membrane_force_hessian`

**物理模型**：Stable Neo-Hookean（SNH）2D 膜能量（Smith et al. 2018 扩展到薄壳）

**能量密度**：
$$
\Psi = \frac{\mu}{2}(I_c - 2) + \frac{\lambda}{2}(J_s - \alpha)^2
$$

其中：
- $I_c = \|\mathbf{F}\|_F^2$（Cauchy-Green 不变量的迹）
- $J_s = \sqrt{\det(\mathbf{F}^T\mathbf{F})}$（面积比）
- $\alpha = 1 + \mu / \lambda$

**计算过程**：

1. **变形梯度**：$\mathbf{f}_0 = \Delta\mathbf{x}_{01} D_{00} + \Delta\mathbf{x}_{02} D_{10}$，$\mathbf{f}_1 = \Delta\mathbf{x}_{01} D_{01} + \Delta\mathbf{x}_{02} D_{11}$，其中 $D$ 是 2×2 的 $\mathbf{D}_m^{-1}$
2. **面积比**：$J_s = \sqrt{|\mathbf{f}_0|^2 |\mathbf{f}_1|^2 - (\mathbf{f}_0 \cdot \mathbf{f}_1)^2}$
3. **PK1 应力**：$\mathbf{P}_k = \mu \, \mathbf{f}_k + \lambda(J_s - \alpha) \, \mathbf{g}_k$，其中 $\mathbf{g}_k = \partial J_s / \partial \mathbf{f}_k$
4. **力**：$\mathbf{f}_{\text{vert}} = -A \, (\mathbf{P}_0 \frac{\partial f_0}{\partial x} + \mathbf{P}_1 \frac{\partial f_1}{\partial x})$
5. **Hessian（PSD 投影后）**：$\mathbf{H} = A \, (\mu \cdot I_{\text{coeff}} \cdot \mathbf{I} + c_1 \, \mathbf{d}J \otimes \mathbf{d}J - r \, \mathbf{w} \otimes \mathbf{w})$

**阻尼项**（当 `damping > 0`）：基于 Green 应变率的 Rayleigh 阻尼，分 μ 和 λ 两个分量。

| 参数 | 含义 |
|------|------|
| `mu` (tri_ke) | Lamé 第一参数 μ，控制剪切刚度 |
| `lambda` (tri_ka) | Lamé 第二参数 λ，控制面积保持 |
| `damping` (tri_kd) | 阻尼系数 |
| `area` | 三角形静止面积 |
| `DmInv` | 2×2 参考构型逆矩阵 |

---

#### 3.2.2 `cloth_vbd_bending_force_hessian`

**来源**：Newton `evaluate_dihedral_angle_based_bending_force_hessian`

**物理模型**：Grinspun 离散壳弯曲能（Discrete Shells, 2003）

**能量**：
$$
\Psi_{\text{bend}} = \frac{1}{2} k \, l_e \, (\theta - \bar{\theta})^2
$$

其中 $k$ 是弯曲刚度，$l_e$ 是静止边长，$\theta$ 是二面角，$\bar{\theta}$ 是静止二面角。

**顶点编号**：`[o0, o1, v2, v3]` — `o0` 和 `o1` 是两个三角形的对向顶点，`v2` 和 `v3` 是公共边的两个端点。

```
    o0
   / \
  v2---v3   (公共边 e = v3 - v2)
   \ /
    o1
```

**计算过程**：

1. **法向量**：$\mathbf{n}_1 = (\mathbf{x}_{v2} - \mathbf{x}_{o0}) \times (\mathbf{x}_{v3} - \mathbf{x}_{o0})$，$\mathbf{n}_2 = (\mathbf{x}_{v3} - \mathbf{x}_{o1}) \times (\mathbf{x}_{v2} - \mathbf{x}_{o1})$
2. **二面角**：$\theta = \text{atan2}((\hat{\mathbf{n}}_1 \times \hat{\mathbf{n}}_2) \cdot \hat{\mathbf{e}}, \; \hat{\mathbf{n}}_1 \cdot \hat{\mathbf{n}}_2)$
3. **角度对位置的导数**：通过法向量对位置的导数链式推导，$\frac{\partial \hat{\mathbf{n}}}{\partial \mathbf{x}} = \frac{1}{\|\mathbf{n}\|}(\mathbf{I} - \hat{\mathbf{n}} \hat{\mathbf{n}}^T) \frac{\partial \mathbf{n}}{\partial \mathbf{x}}$
4. **力**：$\mathbf{f} = -k \, l_e \, (\theta - \bar{\theta}) \, \frac{\partial \theta}{\partial \mathbf{x}_v}$
5. **Hessian（Gauss-Newton 近似）**：$\mathbf{H} = k \, l_e \, \frac{\partial \theta}{\partial \mathbf{x}_v} \otimes \frac{\partial \theta}{\partial \mathbf{x}_v}$

| 参数 | 含义 |
|------|------|
| `stiffness` (edge_ke) | 弯曲刚度 $k$ |
| `damping` (edge_kd) | 弯曲阻尼 |
| `rest_angle` | 静止二面角 $\bar{\theta}$ |
| `rest_length` | 静止边长 $l_e$（能量按边长加权） |

---

### 3.3 `vbd_update_velocity_kernel`

**作用**：从位置差分更新速度。

$$
\mathbf{v}_i^{n+1} = \frac{\mathbf{x}_i^{n+1} - \mathbf{x}_i^n}{\Delta t}
$$

**线程映射**：每线程一个顶点，共 `n_particles` 个线程。

---

## 4. 图着色

### 着色算法

`build_cloth_coloring` 在 CPU 上执行贪心图着色：

1. **构建冲突图**：
   - 每个三角形产生 $\binom{3}{2} = 3$ 条冲突边
   - 每个弯曲边元素 `[o0, o1, v2, v3]` 产生 $\binom{4}{2} = 6$ 条冲突边
   - 去重后形成无向图

2. **贪心着色**（Largest-Degree-First）：
   - 按度数从大到小排序
   - 每个顶点选择不与邻居冲突的最小颜色

### 颜色数量

| 网格分辨率 | 顶点数 | 三角形数 | 弯曲边数 | 颜色数 |
|-----------|-------|---------|---------|-------|
| 64×32     | 2,145 | 4,096   | 6,240   | **8** |
| 100×100   | 10,201 | 20,000 | 30,200  | **7** |

对于规则的矩形布料网格，颜色数通常在 **7–8** 之间。
这是因为每个内部顶点最多与 12 个邻居产生冲突（6 个直接三角形邻居 + 弯曲元素带来的额外 2-hop 邻居），而贪心着色在规则网格上的色数近似为最大度数 + 1。

### 颜色分组的数据结构

```
color_groups_[c].indices  →  GPU 数组，存储该颜色的所有顶点 ID
color_groups_[c].count    →  该颜色的顶点数量
```

每轮 VBD 迭代中，依次对每个颜色组启动一次 `vbd_cloth_solve_kernel`。
同色组内的顶点可安全并行（无数据竞争），不同颜色组间串行执行。

---

## 5. 邻接表（CSR 格式）

每个顶点需要知道它参与了哪些三角形和弯曲边，以及它在该元素中的局部编号（`v_order`）。

### 三角形邻接 `tri_adj_`

对于顶点 `i`，它参与的所有三角形存储在：
```
tri_adj_.data[tri_adj_.offsets[i] .. tri_adj_.offsets[i+1])
```
每两个 int 为一组：`(tri_id, v_order)`，其中 `v_order ∈ {0, 1, 2}` 表示该顶点在三角形中的局部索引。

### 弯曲边邻接 `edge_adj_`

同理，每两个 int 一组：`(edge_id, v_order)`，其中 `v_order ∈ {0, 1, 2, 3}` 对应 `[o0, o1, v2, v3]`。

---

## 6. 仿真参数（对齐 Newton `example_cloth_hanging.py` VBD 路径）

| 参数 | 值 | 说明 |
|------|-----|------|
| `fps` | 60 | 帧率 |
| `sim_substeps` | 10 | 每帧子步数 |
| `sim_dt` | 1/600 s | 子步时间步长 |
| `vbd_iterations` | 10 | 每 substep 的 VBD 迭代次数 |
| `cell_x/y` | 0.1 m | 网格单元尺寸 |
| `mass` | 0.1 kg/particle | 质量参数（经面积加权重分配） |
| `tri_ke` (μ) | 1000 | Lamé 第一参数 |
| `tri_ka` (λ) | 1000 | Lamé 第二参数 |
| `tri_kd` | 0.1 | 膜阻尼 |
| `edge_ke` | 10 | 弯曲刚度 |
| `edge_kd` | 0.0 | 弯曲阻尼 |
| `gravity` | (0, 0, -9.81) | 重力 |

### 质量分配（复现 Newton `add_cloth_grid`）

```
total_mass = mass × (dim_x + 1)²
total_area = cell_x × cell_y × dim_x × dim_y
density = total_mass / total_area
particle_mass[v] += density × tri_area / 3    (对每个关联三角形)
```

---

## 7. 核函数调用时序图

```
substep_cloth()
│
├─ vbd_forward_step_kernel ◄─── 1 次, N 线程
│     保存 q_prev, 计算 inertia, 初始化 displacement
│
├─ for iter = 0..9:
│   ├─ vbd_cloth_solve_kernel [color 0] ◄─── ~N/8 线程
│   ├─ vbd_cloth_solve_kernel [color 1] ◄─── ~N/8 线程
│   ├─ ...
│   └─ vbd_cloth_solve_kernel [color 7] ◄─── ~N/8 线程
│
│   (10 iterations × 8 colors = 80 次 kernel launch)
│
└─ vbd_update_velocity_kernel ◄─── 1 次, N 线程
      v[i] = (pos[i] - q_prev[i]) / dt
```

**总 kernel 启动次数（每 substep）**：$1 + \text{iterations} \times \text{num\_colors} + 1 = 82$

**每帧总启动次数**：$82 \times 10 = 820$（10 substeps/frame）

---

## 8. 文件索引

| 文件 | 内容 |
|------|------|
| `ChysX/src/solver/vbd_solver.h` | VBDSolver 类声明、ElementAdjacency / ColorGroup 结构体 |
| `ChysX/src/solver/vbd_solver.cu` | 所有核函数实现 + host 方法（着色、邻接表构建、step_cloth） |
| `ChysX/src/solver/cloth_vbd_kernels.cuh` | 膜能量和弯曲能量的 device 函数 |
| `newton/examples/chysx/cloth/example_chysx_cloth_hanging_vbd.py` | 布料悬挂场景示例 |
