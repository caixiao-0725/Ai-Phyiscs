# 正交势能 `||A^T A - I||_F^2` 的正定性：完整推导与实验

> 本文是 `ABD_Orthogonality_Potential.md` 的补充。
> 原文档给出了梯度、精确 Hessian block、以及部分一维正定性分析（`A=0` 负定、正交点半正定、`σ<1/√3` 出现负曲率）。
> 本文把 9×9 Hessian **在 SVD 坐标系下完全对角化**，给出全部 9 个特征值的闭式表达式，
> 据此精确刻画 PSD 区域，并给出一套可复跑的数值实验。
>
> 配套脚本：
>
> - `docs/ortho_hessian_experiment.py`（5 个验证实验）
> - `docs/ortho_hessian_plot.py`（可视化，输出 `docs/ortho_hessian_curvature.png`）

---

## 0. 约定与一个重要等价

设 `A` 为 3×3 仿射矩阵，stiffness 为 `k`，定义

```math
S = A^T A - I,\qquad E(A) = k\,\|A^T A - I\|_F^2 = k\,\operatorname{tr}(S^2).
```

**`A^T A` 与 `A A^T` 等价**：二者奇异值相同，能量值完全相同，Hessian 的特征谱也完全相同
（只差一个 `A ↔ A^T` 的坐标重排）。因此本文用 `A^T A`（与约束 `||ATA-I||` 一致），
而原文档用 `A A^T`，所有正定性结论通用。

---

## 1. 推导思路

### 1.1 第一步：把能量归约到奇异值

整个分析的出发点是：**能量只依赖奇异值**。

对 `A` 做 SVD `A = U Σ V^T`。因为 Frobenius 范数在正交变换下不变：

```math
\|A^T A - I\|_F^2
= \|V(\Sigma^2 - I)V^T\|_F^2
= \|\Sigma^2 - I\|_F^2,
```

所以

```math
\boxed{\,E(A) = k\sum_{i=1}^{3}(\sigma_i^2 - 1)^2\,}
```

这一步是后面一切的基础：它告诉我们 `E` 在「旋转 × 旋转」作用 `A → Q A R`（`Q,R` 正交）下不变，
真正的自由度只有三个奇异值。

### 1.2 第二步：写出梯度

```math
\boxed{\nabla_A E = 4k\,A\,(A^T A - I) = 4k\,A S}
```

**详细推导**（矩阵微分法）：

$$
\begin{aligned}
E &= k\,\|A^T A - I\|_F^2 = k\,\operatorname{tr}(S^2), \quad S = A^T A - I. \\[4pt]
\end{aligned}

$$

对 $E$ 取微分，利用 $\operatorname{tr}((dS)S) = \operatorname{tr}(S\,dS)$：

$$
dE = k\,\operatorname{tr}(d(S^2)) = k\,\operatorname{tr}\big((dS)S + S(dS)\big)
     = 2k\,\operatorname{tr}(S\,dS).

$$

代入 $dS = dA^T A + A^T dA$：

$$
dE = 2k\,\operatorname{tr}\big(S(dA^T A + A^T dA)\big)
    = 2k\Big[\operatorname{tr}(S\,dA^T A) + \operatorname{tr}(S\,A^T dA)\Big].

$$

第一项用循环恒等式 $\operatorname{tr}(B\,dA^T) = \operatorname{tr}(B^T dA)$ 转为 $dA$ 的形式：

$$
\operatorname{tr}(S\,dA^T A) = \operatorname{tr}(A S\,dA^T)
= \operatorname{tr}\big((A S)^T dA\big)
= \operatorname{tr}(S A^T dA),

$$

其中用到 $S$ 对称（$(AS)^T = S A^T$）。代入得两项相等：

$$
dE = 2k\Big[\operatorname{tr}(S A^T dA) + \operatorname{tr}(S A^T dA)\Big]
    = 4k\,\operatorname{tr}(S A^T dA).

$$

由梯度定义 $dE = \operatorname{tr}\big((\nabla_A E)^T dA\big)$ 读出：

$$
\nabla_A E = 4k\,A S = 4k\,A\,(A^T A - I).

$$

（与原文档 `4k(AA^T-I)A` 互为转置约定。）驻点条件 `AS = 0`：要么 `A` 行/列退化，要么 `S=0`（正交）。

### 1.3 第三步：精确 Hessian 的二次型

> **记号说明**：Hessian **矩阵** $H$ 是一个 9×9 矩阵，只由 $A$ 决定，不依赖方向 $X$。
> 但写出 $H$ 的每个元素需要将 $A$ 向量化为 9 维向量再求 9×9 的二阶偏导，非常繁琐。
> 因此我们改而计算 **沿方向 $X$ 的二阶方向导数**（即曲率）：
>
> $$
> E''(X) = \left.\frac{d^2}{d\varepsilon^2}E(A+\varepsilon X)\right|_{\varepsilon=0}
>       = \operatorname{vec}(X)^T \, H \, \operatorname{vec}(X),
>
> $$
>
> 它是一个**标量**（依赖于 $A$ **和**方向 $X$），等价于将 $H$ 夹在 $\operatorname{vec}(X)$ 两侧得到的二次型。
> 下面推导它的闭式表达式。

沿方向 `X` 取 `A(ε) = A + εX`，对 `E` 做二阶展开。

**方法一（直接展开，推荐）**：写出 $S(\varepsilon) = A(\varepsilon)^T A(\varepsilon) - I$ 到 $\varepsilon^2$ 阶：

$$
\begin{aligned}
S(\varepsilon) &= (A+\varepsilon X)^T (A+\varepsilon X) - I \\
&= \underbrace{A^T A - I}_{S}
   + \varepsilon\underbrace{(X^T A + A^T X)}_{dS}
   + \varepsilon^2 \underbrace{X^T X}_{\frac12 d^2S}.
\end{aligned}

$$

代入 $E(\varepsilon) = k\operatorname{tr}(S(\varepsilon)^2)$，展开到 $\varepsilon^2$：

$$
\begin{aligned}
E(\varepsilon) &= k\operatorname{tr}\!\big[(S + \varepsilon\,dS + \varepsilon^2 X^T X)^2\big] \\
&= k\operatorname{tr}\!\big[S^2
   + \varepsilon(S\,dS + dS\,S) \\
&\qquad\quad + \varepsilon^2(dS^2 + S X^T X + X^T X S)\big] + O(\varepsilon^3).
\end{aligned}

$$

一阶项（梯度）验证：

$$
\tfrac12 E'(0) = \tfrac12\cdot k\operatorname{tr}(S\,dS + dS\,S) = k\operatorname{tr}(S\,dS)
= k\operatorname{tr}\!\big(S(X^T A + A^T X)\big) = 2k\operatorname{tr}(S A^T X),

$$

与 $\langle \nabla E, X\rangle = \langle 4k A S, X\rangle = 4k\operatorname{tr}(S A^T X)$ 一致 ✓。

**二阶项（Hessian 二次型）**：收集 $\varepsilon^2$ 系数：

$$
\tfrac12 E''(0) = k\operatorname{tr}(dS^2) + k\operatorname{tr}(S X^T X + X^T X S).

$$

由迹的循环性 $\operatorname{tr}(S X^T X) = \operatorname{tr}(X^T X S)$，第二项合并为 $2k\operatorname{tr}(S X^T X) = 2k\langle S, X^T X\rangle$。

而 $dS = X^T A + A^T X$，所以 $\operatorname{tr}(dS^2) = \|X^T A + A^T X\|_F^2$。

综上得到（脚本 `hess_quadform_analytic` 已数值验证）：

```math
\boxed{
\tfrac12 E''(X)
= \underbrace{k\,\|X^T A + A^T X\|_F^2}_{\text{Gauss–Newton, } \ge 0}
+ \underbrace{2k\,\langle A^T A - I,\; X^T X\rangle}_{\text{曲率项, 可负}}
}
```

> 重申：上式是 **二次型** $\tfrac12 \operatorname{vec}(X)^T H \operatorname{vec}(X)$，不是 $H$ 本身。
> $H$ 是一个只由 $A$ 决定的 9×9 矩阵——把上式看成 $X$ 的二次函数，它的系数矩阵就是 $H$。
> 我们不需要显式写出 $H$ 的每个元素（实验 2 用数值差分组装），因为 1.4 节会直接在 SVD 坐标系下对角化这个二次型。

**关键洞察**：负曲率只可能来自第二项。它为负当且仅当 `A^T A - I` 在某些方向上为负，
即存在方向 `v` 被压缩到 `||A v|| < 1`。这就是"压扁导致不正定"的本质。

### 1.4 第四步（核心）：在 SVD 坐标系下对角化

把扰动写到 SVD 主轴坐标系 `X = U Y V^T`。代入上式后：

```math
X^T A + A^T X = V(Y^T\Sigma + \Sigma Y)V^T,\qquad
\langle A^T A - I, X^T X\rangle = \langle \Sigma^2 - I, Y^T Y\rangle,
```

问题完全退化到 `A = Σ = diag(σ_1,σ_2,σ_3)` 的对角情形。此时 9 个自由度 `Y_{ij}` **解耦成三族**：

- **对角 `Y_{ii}`（stretch）**：彼此独立，3 个标量模态；
- **非对角对 `(Y_{ij}, Y_{ji})`（i<j）**：每对是一个 2×2 块，其对称组合是 sym-shear，反对称组合是 rotation。

逐项展开 2×2 块（`p=σ_i^2+σ_j^2-1`, `q=σ_iσ_j`，矩阵 `[[p,q],[q,p]]` 特征值 `p±q`），得到**全部 9 个特征值的闭式**。

---

## 2. 主结果：Hessian 的 9 个特征值（闭式）

在 SVD 坐标系下，9×9 Hessian 块对角化为：


| 模态          | 个数 | 特征值（= 该方向曲率）                 | 几何含义               |
| --------------- | ------ | ---------------------------------------- | ------------------------ |
| **stretch**   | 3    | `4k(3σ_i² − 1)`                     | 改变单个奇异值         |
| **sym-shear** | 3    | `4k(σ_i² + σ_j² + σ_iσ_j − 1)`  | `(i,j)` 平面对称剪切   |
| **rotation**  | 3    | `4k(σ_i² + σ_j² − σ_iσ_j − 1)` | `(i,j)` 平面无穷小旋转 |

> 这是对原文档的补充与修正：原文档只分析了 stretch 方向（一维 `e''(σ)=4k(3σ²−1)`），
> 漏掉了 sym-shear 与 rotation 两族。三族里 **rotation 因带 `−σ_iσ_j` 项而最易转负**。

### 2.1 PSD 充要条件

Hessian 半正定 `⇔` 下面 9 个量同时 `≥ 0`：

```math
\text{(a) stretch:}\quad 3\sigma_i^2 - 1 \ge 0 \quad (\forall i)
```

```math
\text{(b) sym-shear:}\quad \sigma_i^2 + \sigma_j^2 + \sigma_i\sigma_j - 1 \ge 0 \quad (\forall i<j)
```

```math
\text{(c) rotation:}\quad \sigma_i^2 + \sigma_j^2 - \sigma_i\sigma_j - 1 \ge 0 \quad (\forall i<j)\ \ \text{← 最先被破坏}
```

### 2.2 三个标志性位置


| 位置          | 奇异值 | Hessian                                                 | 说明              |
| --------------- | -------- | --------------------------------------------------------- | ------------------- |
| `A = 0`       | `σ=0` | 所有曲率`=−4k` → **负定**                             | 驻点但是极大/鞍点 |
| `A = sI`      | `σ=s` | stretch/sym-shear`=4k(3s²−1)`；rotation `=4k(s²−1)` | 见下              |
| 正交`A∈O(3)` | `σ=1` | rotation 三模态`=0`，其余 `=8k` → **半正定且奇异**     | 零空间 =`so(3)`   |

**关于 `A = sI` 的关键修正**：

```math
\text{rotation } = 4k(s^2-1) < 0 \iff s < 1,\qquad
\text{stretch } = 4k(3s^2-1) < 0 \iff s < \tfrac{1}{\sqrt3}\approx 0.577.
```

也就是说：**只要均匀缩小 `s < 1`，rotation 模态就已经出现负曲率**。
原文档给的 `1/√3` 只是 stretch 方向的阈值，并不是 Hessian 失去 PSD 的真正阈值。
真正"最先破坏 PSD"的阈值是 `s = 1`（rotation 模态）。

---

## 2.3 设计更好的能量

上述分析表明，原能量 $E = k\|A^TA - I\|_F^2$ 的 Hessian 在 $\sigma_i < 1$ 时必然出现负特征值
（rotation 模态首先被破坏）。有没有办法设计一个能量，使其 Hessian 在更大的区域保持半正定？

### 设计目标

良好的正交保持能量应满足：

1. **$A \in O(3)$ 时 $E=0$**（正交矩阵是能量极小点）
2. **$E(A)$ 随偏离正交单调增加**
3. **Hessian 的 PSD 区域尽可能大**
4. **旋转不变性**：$E(QA) = E(A)$ 对任意 $Q\in SO(3)$（不惩罚刚体旋转）

### 候选设计

下面比较几种候选能量在 $A = sI$ 路径上的 Hessian 最小特征值（$k=1$）：

| 能量形式 | 名称 | 旋转不变 | PSD 阈值 $s^*$ | $s=0.7$ 时 $\lambda_{\min}$ | 说明 |
|---|---|---|---|---|---|
| $\|A^TA-I\|^2$ | 原能量 (StVK) | ✅ | $s=1$ | $-2.04$ | 基准线 |
| $\frac12\sum(\sigma_i-1)^2$ | **corotational** | ✅ | $s=1$ | $-0.43$ | 负曲率缩小 5× |
| $+\beta\|A\|_F^2,\ \beta=2k$ | Frobenius 正则化 | ✅ | $s=0$ | $+1.96$ | PSD 全区域，但极小点偏移 |
| $+\lambda\|A-I\|^2,\ \lambda=1.1k$ | 身份正则化 | ❌ | $s\approx0.65$ | $+0.16$ | 效果好但破坏旋转不变 |
| $-\mu\log\det(A^TA)$ | **对数障碍** | ✅ | — | $-4.08$ | ❌ 让 rotation 更负！ |

下面详细介绍每个设计。

---

#### 设计 A：Frobenius 正则化（推荐折中）

$$E_A(A) = k\|A^TA - I\|_F^2 + \beta\|A\|_F^2$$

**Hessian**：第二项贡献 $2\beta I_{9\times 9}$，均匀抬高所有特征值。

以 $A = sI$ 为例，rotation 模态特征值变为：

$$\lambda_{\text{rot}} = 4k(s^2 - 1) + 2\beta$$

半正定条件 $\lambda_{\text{rot}} \ge 0$ 要求 $\beta \ge 2k(1-s^2)$。取 $\beta = 2k$ 可保证 **对所有 $s\ge0$ 半正定**。

**代价**：极小点从 $\sigma_i=1$ 偏移到 $\sigma_i = \sqrt{1 - \beta/(2k)}$。

| $\beta/k$ | 极小点 $\sigma$ | PSD 范围 | 适用场景 |
|---|---|---|---|
| $0$ | $1$ | $s \ge 1$ | 纯正交约束 |
| $0.5$ | $0.865$ | $s \gtrsim 0.7$ | 允许轻微压缩 |
| $1.0$ | $0.705$ | $s \gtrsim 0.4$ | 允许显著压缩 |
| $2.0$ | $0$ | 全区域 | 但极小点为 $A=0$，不推荐 |

**推荐取值** $\beta/k \in [0.5, 1.0]$，在 PSD 范围与极小点偏移之间取得平衡。

> **优点**：旋转不变、计算简单（只需额外 $\beta\|A\|^2$ 项）、可保证 PSD。
> **缺点**：极小点偏移，需要接受一定压缩。

---

#### 设计 B：Corotational 能量（理论最优）

$$E_B(A) = \frac{k}{2}\|A - R\|_F^2 = \frac{k}{2}\sum_{i=1}^3 (\sigma_i - 1)^2$$

其中 $R = \text{polar}(A) = UV^T$ 来自 SVD $A = U\Sigma V^T$。

以 $A = sI$ 为例，Hessian 特征值为：

- stretch/sym-shear：$k$（**恒正，与 $s$ 无关**）
- rotation：$k\,\dfrac{s-1}{s}$

与原能量对比（$s=0.8$ 时）：

| 模态 | 原能量 | Corotational |
|---|---|---|
| rotation | $4k(s^2-1) = -1.44k$ | $k(s-1)/s = -0.25k$ |
| stretch | $4k(3s^2-1) = 3.68k$ | $k$ |

**Corotational 的负曲率幅度仅为原能量的 $1/8$ 左右**（在 $s\approx 1$ 附近），显著改善但无法完全消除。

> **优点**：旋转不变、负曲率大幅减轻、极小点精确在正交群。
> **缺点**：每步需要 SVD 计算 $R$；仍然存在负曲率（但更小幅）。

---

#### 设计 C：混合策略 — 求解器级保护

不改能量，只改牛顿步的 Hessian：

$$H_{\text{used}} = H_{\text{exact}} + \tau I,\qquad \tau = \max\big(0,\; 4k(1-\sigma_{\min}^2)\big)$$

其中 $\sigma_{\min}$ 是 $A$ 的最小奇异值。利用第 5 节的实用判据：

- 当 $\sigma_{\min} \ge 1$：$\tau = 0$，使用精确 Hessian（此时它已 PSD）
- 当 $\sigma_{\min} < 1$：$\tau = 4k(1-\sigma_{\min}^2)$，正好抵消 rotation 模态的最负特征值

这是求解器层面最轻量的方案，无需改变目标函数。

---

#### 设计 D：身份正则化（效果好但破坏旋转不变）

$$E_D(A) = k\|A^TA - I\|_F^2 + \lambda\|A - I\|_F^2$$

Hessian 增加 $2\lambda I_{9\times 9}$。取 $\lambda = 1.1k$ 时，
$s=0.7$ 处 $\lambda_{\min} = +0.16$，接近 PSD。

**代价**：$E(R) > 0$ 对非恒等的旋转 $R$，**刚体旋转也会产生能量**。
仅适用于有固定参考姿态（如 $A=I$）的场景。

---

### 小结与推荐

| 场景 | 推荐方案 | 理由 |
|---|---|---|
| 需要精确旋转不变性 | **Corotational**（设计 B） | 极小点在正交群，负曲率缩小 5-8× |
| 允许轻微压缩偏差 | **Frobenius 正则化** $\beta=0.5k$（设计 A） | 计算简单，PSD 范围扩大到 $s \gtrsim 0.7$ |
| 不改能量、只改求解器 | **阻尼牛顿** $\tau = 4k(1-\sigma_{\min}^2)$（设计 C） | 零侵入，按需开启 |
| 有固定参考姿态 $A=I$ | **身份正则化** $\lambda=1.1k$（设计 D） | PSD 范围最大 |

**数值验证**：对应脚本见 `docs/ortho_hessian_experiment.py`。

---

## 2.4 显式 9×9 Hessian 矩阵与阻尼牛顿实现

设计 C 需要在每步牛顿迭代中组装 $H + \tau I$。下面给出可直接编程实现的显式公式。

### 2.4.1 梯度与 Hessian 的逐元素公式

沿用第 0 节 $S = A^T A - I$。梯度：

$$(\nabla_A E)_{ij} = 4k\sum_m A_{im} S_{mj}.$$

Hessian 是 **4 阶张量** $H_{ij,pq} = \partial^2 E / \partial A_{ij} \partial A_{pq}$，下标约定为 $\partial/\partial A_{pq}$ 作用在 $(\nabla E)_{ij}$ 上：

$$
\boxed{\;
H_{ij,pq} = 4k\Big[
\delta_{ip} (A^T A)_{qj}
+ A_{iq} A_{pj}
+ (A A^T)_{ip} \delta_{jq}
- \delta_{ip} \delta_{jq}
\Big]
\;}
$$

其中 $\delta_{ab}$ 是 Kronecker delta。

### 2.4.2 组装 9×9 矩阵

采用**列优先**（Fortran order）向量化 $\operatorname{vec}(A)$，坐标映射：

- 行索引：$r = i + 3(j-1)$，$i,j=1,2,3$
- 列索引：$c = p + 3(q-1)$，$p,q=1,2,3$

则 9×9 Hessian 矩阵 $H$ 的每个元素为：

$$H_{r,c} = H_{ij,pq}$$

**算法（伪代码）**：

```
# 输入: A (3×3), k (标量)
# 输出: H (9×9)

S = A'*A - eye(3)
ATA = A'*A
AAT = A*A'

H = zeros(9,9)
for i,j,p,q in 1..3:
    r = i + 3*(j-1)
    c = p + 3*(q-1)
    H(r,c) = 4*k * (
        kronecker(i,p) * ATA(q,j)
        + A(i,q) * A(p,j)
        + AAT(i,p) * kronecker(j,q)
        - kronecker(i,p) * kronecker(j,q)
    )
```

> **说明**：对于 3×3 矩阵，$H$ 只有 $9\times 9 = 81$ 个元素，直接逐元素计算即可，无需任何稀疏技巧。

### 2.4.3 阻尼牛顿步

每步牛顿迭代：

1. 计算梯度 $G = 4k A S$（3×3 矩阵）
2. 计算 $\sigma_{\min}$（$A$ 的最小奇异值）
3. 计算阻尼系数 $\tau = \max\big(0,\; 4k(1 - \sigma_{\min}^2)\big)$
4. 组装 $H$（9×9），修正为 $H_{\text{used}} = H + \tau I_{9\times 9}$
5. 解线性系统 $H_{\text{used}} \, \operatorname{vec}(\Delta A) = -\operatorname{vec}(G)$
6. 更新 $A \leftarrow A + \Delta A$

**等价替代**（不显式组装 $H$）：直接用 Hessian-向量积公式

$$\operatorname{vec}\big(H[X]\big) = \operatorname{vec}\!\Big(4k\big[X S + A (X^T A + A^T X)\big]\Big),$$

配合共轭梯度法（CG）求解 $(H + \tau I)\Delta A = -G$，仅需 $H[X]$ 运算，避免组装 81 个元素。对 3×3 系统两者差别不大，选择更清晰的实现即可。

--- 

## 3. 实验思路

目标：用数值手段独立验证上面每一条解析结论，确保推导没有代数错误，并量化"何时不正定"。

实现见 `docs/ortho_hessian_experiment.py`，向量化约定为 `vec(A)` 列优先（Fortran order）。

### 实验 1 — 验证梯度与 Hessian 二次型

- **思路**：解析公式 vs 有限差分，是一切后续结论的地基。
- 解析梯度 `4kAS` 对比能量的中心差分梯度。
- 解析二次型 `½E''(X)` 对比能量的二阶差分 `(E(A+hX)−2E(A)+E(A−hX))/h²`。
- **判据**：随机 200 个 `A`、每个 5 个随机方向 `X`，最大误差应 `≲1e−5`（梯度）/`1e−2`（二阶差分）。

### 实验 2 — 验证 9 个特征值闭式公式

- **思路**：用解析梯度做中心差分组装出完整 9×9 数值 Hessian，对其 `eigvalsh`，
  与第 2 节闭式公式排序后逐一对比。
- **判据**：随机 300 个 `A`，排序后特征值最大误差应 `≲1e−3`。
- 附带打印 `A=diag(1.2,0.8,0.3)` 的逐模态特征值，直观看到哪些模态负、对应哪个奇异值。

### 实验 3 — `A = sI` 扫描，定位符号翻转阈值

- **思路**：沿最干净的一维路径 `A=sI` 扫 `s`，打印 `λ_min(H)`、是否 PSD、最先转负的模态名。
- **预期**：`s<1` 全程非 PSD，且最先转负的永远是 `rotation`；`s≥1` 才 PSD。
- 验证 `1/√3` 与 `1` 两个阈值的不同角色。

### 实验 4 — 随机矩阵统计，`σ_min` 与 PSD 的关系

- **思路**：生成 2 万个随机 `A`（控制最大奇异值在 `[0.1,1.6]`），按 `σ_min` 分桶统计 PSD 比例。
- **预期**：`σ_min` 越接近/超过 1，PSD 比例越高；`σ_min<1/√3` 几乎必含负曲率。
  这给出工程上"用 `σ_min` 当 PSD 代理指标"的依据。

### 实验 5 — 正交点零空间维数

- **思路**：取随机旋转 `R∈SO(3)`，数值组装 Hessian，统计 `|λ|<1e−6` 的特征值个数。
- **预期**：恰好 3 个零特征值，对应 `dim so(3)=3`（无穷小旋转），证明势能不惩罚刚体旋转。

### 可视化（`ortho_hessian_plot.py`）

- **左图**：`A=sI` 时 stretch/sym-shear 与 rotation 曲率随 `s` 的曲线，标出 `1/√3` 与 `1`。
- **右图**：单个 `(σ_i,σ_j)` 平面内 rotation/sym-shear 的零等高线与负曲率区域，正交点 `(1,1)` 落在 rotation 边界上。

---

## 4. 实验结果（实测）


| 实验 | 指标                   | 结果                           |
| ------ | ------------------------ | -------------------------------- |
| 1    | 梯度最大误差           | `2.8e−7` ✓                   |
| 1    | Hessian 二次型最大误差 | `5.0e−5` ✓                   |
| 2    | 9 特征值排序最大误差   | `4.1e−8` ✓                   |
| 3    | `s<1` 是否始终非 PSD   | 是，且首先转负总是 rotation ✓ |
| 5    | 正交点零特征值个数     | `3`（`= dim so(3)`）✓         |

实验 4 的分桶统计（节选）：

```text
sigma_min 区间      PSD 比例
[1.00,1.10)         1.000
[0.90,1.00)         0.800
[0.80,0.90)         0.450
[0.50,0.60)         0.023
[0.00,0.50)         0.000
```

结论一致：`σ_min` 落到 1 以下，PSD 概率迅速崩塌；`σ_min<0.577` 时几乎必有负曲率。

---

## 5. 给求解器的实用判据

不必每步做完整特征分解，用奇异值即可分层判断（危险程度从强到弱）：

```text
if sigma_min(A) < 1:           # rotation 模态可能负 —— 最常见的失稳源
    Hessian 可能非 PSD
if sigma_min(A) < 1/sqrt(3):   # 连 stretch 方向也负 —— 强负曲率
    Hessian 几乎必非 PSD
```

不做 SVD 时可用 `λ_min(A^T A) = σ_min² < 1` 近似。
检测到危险区时配合下列任一保护：

- line search；
- modified Cholesky；
- 对角正则（diagonal regularization）；
- 远离正交时回退到恒半正定的 Gauss–Newton Hessian `H_GN = 2k J^T J`；
- 限制 affine step size。

---

## 6. 小结

- `E(A)=k‖A^TA−I‖_F²` 只依赖奇异值：`E=k∑(σ_i²−1)²`。
- 精确 Hessian 二次型 = Gauss–Newton 项（恒 `≥0`）+ 曲率项 `2k⟨A^TA−I, X^TX⟩`（可负）。
- 在 SVD 坐标系下 Hessian 完全对角化为 **stretch / sym-shear / rotation** 三族，9 个特征值有闭式。
- PSD 充要条件由这 9 个量给出；**rotation 模态最先被破坏，阈值是 `σ` 进入 1 以下**，
  而非原文档所说的 `1/√3`（后者只是 stretch 方向阈值）。
- 正交点处半正定且奇异，零空间正是 `so(3)`，符合"不惩罚刚体旋转"的物理期望。
- 工程上以 `σ_min < 1`（或 `λ_min(A^TA)<1`）作为非 PSD 预警，配合 line search / 正定化 / GN 回退。
