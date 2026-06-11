# ABD 正交势能 `||AA^T - I||_F^2` 推导与正定性分析

## 1. 背景

```math
\| A A^T - I \|_F^2 \to 0
```

## 2. 行向量记号

论文按行存储 affine 矩阵：

```math
A =
\begin{bmatrix}
a_0^T \\
a_1^T \\
a_2^T
\end{bmatrix}
```

其中每个 `a_i` 是一个 3D 行向量对应的列表示。

定义：

```math
S = A A^T - I
```

则：

```math
S_{ij} = a_i \cdot a_j - \delta_{ij}
```

所以势能可写成：

```math
E(A) = k \sum_{i=0}^{2}\sum_{j=0}^{2}
(a_i \cdot a_j - \delta_{ij})^2
```

也就是：

```math
E(A) =
k\left[
\sum_i (a_i \cdot a_i - 1)^2
+
\sum_{i\ne j}(a_i \cdot a_j)^2
\right]
```

这正是论文 3.2 节给出的多项式形式。

## 3. 矩阵形式梯度

从矩阵形式出发：

```math
E(A) = k \operatorname{tr}\left[(A A^T - I)^2\right]
```

令：

```math
S = A A^T - I
```

微分：

```math
dS = dA A^T + A dA^T
```

于是：

```math
dE = 2k \operatorname{tr}(S\,dS)
```

代入：

```math
dE =
2k \operatorname{tr}\left[S(dA A^T + A dA^T)\right]
```

利用 `S` 对称：

```math
dE = 4k \operatorname{tr}(A^T S\,dA)
```

因此梯度为：

```math
\boxed{
\nabla_A E = 4k (A A^T - I) A
}
```

## 4. 行向量形式梯度

对每一行 `a_i`，有：

```math
g_i = \frac{\partial E}{\partial a_i}
```

由矩阵梯度可得：

```math
\boxed{
g_i = 4k \sum_{j=0}^{2} S_{ij} a_j
}
```

展开就是：

```math
g_i =
4k\left[
(a_i\cdot a_i - 1)a_i
+
\sum_{j\ne i}(a_i\cdot a_j)a_j
\right]
```

这个形式很直观：

- 如果 `a_i` 长度不是 1，则第一项修正长度。
- 如果 `a_i` 与其他行不正交，则第二项修正夹角。

## 5. 精确 Hessian

我们需要对梯度

```math
g_i = 4k \sum_j S_{ij} a_j
```

继续对 `a_p` 求导。

先写出：

```math
S_{ij} = a_i\cdot a_j - \delta_{ij}
```

对 `a_p` 的扰动 `da_p`：

```math
dS_{ij}
=
\delta_{ip}(da_p \cdot a_j)
+
\delta_{jp}(a_i \cdot da_p)
```

因此 Hessian block `H_{ip}` 是从 `da_p` 到 `dg_i` 的线性映射：

```math
dg_i = \sum_p H_{ip} da_p
```

可得：

```math
\boxed{
H_{ip}
=
4k\left[
\delta_{ip}\sum_j a_j a_j^T
+
a_p a_i^T
+
S_{ip} I
\right]
}
```

其中：

```math
S_{ip} = a_i \cdot a_p - \delta_{ip}
```

这是精确 Hessian，不是 Gauss-Newton 近似。

### 5.1 对角块

当 `i = p`：

```math
H_{ii}
=
4k\left[
\sum_j a_j a_j^T
+
a_i a_i^T
+
(\|a_i\|^2 - 1)I
\right]
```

### 5.2 非对角块

当 `i != p`：

```math
H_{ip}
=
4k\left[
a_p a_i^T
+
(a_i\cdot a_p)I
\right]
```

### 5.3 对称性

注意：

```math
H_{pi}^T
=
4k\left[
a_p a_i^T
+
S_{pi}I
\right]
= H_{ip}
```

因为 `S_{ip}=S_{pi}`。所以完整 9x9 Hessian 是对称的。

## 6. 与论文公式的关系

论文给出了类似形式：

```math
\frac{\partial V_\perp}{\partial a_i}
=
2\kappa\nu
\left(
2(a_i\cdot a_i - 1)a_i
+
2\sum_j (a_j\otimes a_j)a_i
\right)
```

用我们的记号整理后等价于：

```math
4k \sum_j S_{ij}a_j
```

其中 `k = kappa * volume`。

论文 Hessian 中对角块包含：

```math
4a_i a_i^T
+
2(\|a_i\|^2 - 1)I
+
2\sum_j a_j a_j^T
```

不同写法的系数差异通常来自：

- 是否把 `i != j` 的交叉项双重计入；
- 是否把势能前面写成 `k ||.||^2` 或 `1/2 k ||.||^2`；
- 行向量/列向量记号差异。

工程实现时最重要的是保持：

```math
gradient = \nabla E
Hessian = \nabla^2 E
```

二者系数一致。整体常数可吸收到 stiffness 中。

## 7. Gauss-Newton Hessian

也可以把每个：

```math
C_{ij}(A) = a_i \cdot a_j - \delta_{ij}
```

看作残差，则：

```math
E = k \sum_{ij} C_{ij}^2
```

Gauss-Newton Hessian 为：

```math
H_{GN} = 2k J^T J
```

它忽略了残差本身的二阶项：

```math
2k \sum_{ij} C_{ij} \nabla^2 C_{ij}
```

所以：

```math
H_{exact}
=
H_{GN}
+
2k \sum_{ij} C_{ij}\nabla^2 C_{ij}
```

### 7.1 Gauss-Newton 的优点

`H_GN` 一定是半正定：

```math
H_{GN} \succeq 0
```

这对数值求解很友好。

### 7.2 Gauss-Newton 的缺点

当 `A` 离正交矩阵较远时，忽略二阶项会降低 Newton 收敛速度，也可能需要更多迭代或更强 line search。

## 8. 精确 Hessian 的正定性分析

精确 Hessian **不是全局正定**。

### 8.1 在 `A = 0` 处

当：

```math
A = 0
```

则：

```math
S = -I
```

梯度为 0：

```math
\nabla E = 4kSA = 0
```

但 Hessian 对角块：

```math
H_{ii} = 4k[-I]
```

所以 Hessian 是负定方向。

这说明 `A=0` 是一个驻点，但不是极小值，而是局部极大/鞍点。

因此：

```math
\boxed{
\nabla^2 E(A) \text{ is not globally positive semidefinite}
}
```

### 8.2 在正交矩阵处

当：

```math
A A^T = I
```

则：

```math
S = 0
```

势能达到全局最小：

```math
E = 0
```

在 `A = I` 附近令：

```math
A = I + X
```

一阶展开：

```math
A A^T - I
=
X + X^T + O(\|X\|^2)
```

所以二阶能量为：

```math
E(I+X)
=
k\|X+X^T\|_F^2
+
O(\|X\|^3)
```

如果 `X` 是反对称矩阵：

```math
X^T = -X
```

则：

```math
X + X^T = 0
```

所以二阶能量为 0。

这意味着 Hessian 在正交矩阵处是：

```math
\boxed{
\text{positive semidefinite but singular}
}
```

零空间对应 infinitesimal rotations。

这正是我们想要的：正交势能不应该惩罚真正的刚体旋转，只应该惩罚 stretch/shear。

### 8.3 在远离正交矩阵处

由于精确 Hessian 包含：

```math
S_{ip} I
```

当某些 `S_{ip}` 为负且幅度较大时，Hessian 会出现负方向。

更具体地，可以从奇异值方向看。因为

```math
E(A) = k\sum_i(\sigma_i^2 - 1)^2
```

其中 `sigma_i` 是 `A` 的奇异值。

若只看某一个奇异值方向，定义一维函数：

```math
e(\sigma)=k(\sigma^2-1)^2
```

一阶导：

```math
e'(\sigma)=4k\sigma(\sigma^2-1)
```

二阶导：

```math
\boxed{
e''(\sigma)=4k(3\sigma^2-1)
}
```

因此：

```math
e''(\sigma)<0
\quad\Longleftrightarrow\quad
\sigma^2 < \frac{1}{3}
```

也就是：

```math
\boxed{
\sigma < \frac{1}{\sqrt{3}} \approx 0.577
}
```

当某个奇异值小于约 `0.577` 时，沿着增大该奇异值的方向，精确 Hessian 会出现负曲率。

直观理解：

```math
(\sigma^2-1)^2
```

在 `sigma=0` 附近是向下弯的。因为从完全压扁的状态开始，稍微增大 `sigma` 会让能量下降，但这个下降一开始具有负二阶曲率。

### 8.4 标量缩放例子：`A = sR`

更直观的例子是：

```math
A = sR
```

其中 `R` 是任意正交矩阵。

则：

```math
AA^T = s^2 I
```

势能为：

```math
E(s)=3k(s^2-1)^2
```

二阶导：

```math
E''(s)=12k(3s^2-1)
```

所以：

```math
E''(s)<0
\quad\Longleftrightarrow\quad
s < \frac{1}{\sqrt{3}}
```

当 `s=0` 时：

```math
E''(0)=-12k
```

这对应前面说的 `A=0` 处存在强负曲率。

### 8.5 一般扰动下的二次型

令扰动为：

```math
A(\epsilon)=A+\epsilon X
```

二阶展开中，Hessian 二次型可写成：

```math
\frac{1}{2}\frac{d^2}{d\epsilon^2}E(A+\epsilon X)\bigg|_{\epsilon=0}
=
k\|XA^T + AX^T\|_F^2
+
2k\langle AA^T-I,\; XX^T\rangle
```

第一项：

```math
k\|XA^T + AX^T\|_F^2
```

总是非负，对应 Gauss-Newton 部分。

第二项：

```math
2k\langle AA^T-I,\; XX^T\rangle
```

可能为负。

因此，负曲率来自：

```math
AA^T-I
```

在某些方向上为负，也就是某些方向被压缩：

```math
v^T(AA^T-I)v < 0
```

等价于：

```math
\|A^T v\|^2 < 1
```

如果扰动 `X` 主要沿这个被压缩方向增加尺度，就可能产生负曲率。

### 8.6 什么时候是负定，什么时候只是有负方向

严格来说，完整 Hessian 很少“整体负定”。更常见、更重要的是：

```math
\exists X \ne 0,\quad X^T H X < 0
```

也就是 Hessian **不正定**，存在负方向。

在 `A=0` 处，情况更强：对任意非零扰动 `X`，

```math
E(\epsilon X)
=
k\| \epsilon^2 XX^T - I\|_F^2
=
3k - 2k\epsilon^2\|X\|_F^2 + O(\epsilon^4)
```

所以二阶项对所有非零 `X` 都是负的。这时 Hessian 是负定的。

但在一般远离正交的 `A` 上，通常只是“部分方向负曲率”，不是整个 Hessian 负定。

### 8.7 对求解器的实际判据

可以用奇异值作为实践判断：

```text
if min_singular_value(A) < 1/sqrt(3):
    exact Hessian may contain negative curvature
```

在不做 SVD 的情况下，可以用：

```math
AA^T
```

的最小特征值近似判断。因为：

```math
\lambda_{\min}(AA^T)=\sigma_{\min}^2
```

所以危险区域是：

```math
\lambda_{\min}(AA^T) < \frac{1}{3}
```

实际工程上，如果发现 `A` 被压扁得很厉害，精确 Hessian 应该配合：

- line search；
- modified Cholesky；
- diagonal regularization；
- 或切换到 Gauss-Newton Hessian。

所以精确 Newton Hessian 需要谨慎使用：

- 可配合 line search；
- 可做 Hessian regularization；
- 可在远离正交时使用 Gauss-Newton；
- 可使用 modified Cholesky；
- 可限制 affine step size。

## 9. 对 ABD 求解的影响

### 9.1 为什么它比直接 polar projection 更一致

直接做：

```text
A <- polar(A)
```

会在 contact solve 之后改变接触点位置。

而 `V_perp` 是能量项，参与同一个局部优化目标：

```math
E_{local}
=
E_{inertia}
+
E_{contact}
+
E_{\perp}
```

因此它不会作为后处理破坏刚刚求出的 contact residual。

### 9.2 为什么它比 OrthoAL 更稳定

OrthoAL 会引入 dual variable：

```text
lambda += rho * C
```

正交约束不是接触约束，没有 unilateral projection，也没有天然阻尼。lambda 很容易在非接触模式中积累出巨大力。

`V_perp` 没有 lambda，只是一个光滑势能，因此更温和。

### 9.3 为什么它仍可能不稳定

因为：

```math
\nabla^2 V_\perp
```

不是全局正定。

如果 affine 矩阵偏离正交太远，精确 Hessian 可能给出非下降方向。此时需要：

- line search；
- damping；
- Hessian regularization；
- 或切换到 Gauss-Newton Hessian。

## 10. 实现建议

### 10.1 局部变量

在当前 ABD 中，局部 affine 变量是：

```math
A_L
```

世界 affine 是：

```math
A = \tilde A A_L
```

为了保持和局部 Schur 系统一致，`V_perp` 可以作用在 `A_L` 上：

```math
E_\perp(A_L) = k \| A_L A_L^T - I \|_F^2
```

也可以作用在世界矩阵 `A` 上。若 `\tilde A` 是正交矩阵，两者等价；若 `\tilde A` 已经有 affine drift，则二者不同。

当前代码更自然地作用在 `A_L`。

### 10.2 加入局部线性系统

在当前 `A_0` 处计算：

```math
g = \nabla E(A_0)
H = \nabla^2 E(A_0)
```

Newton 二次模型：

```math
E(A) \approx
E(A_0)
+
g^T(A-A_0)
+
\frac{1}{2}(A-A_0)^T H (A-A_0)
```

展开得到线性系统 RHS：

```math
H A = H A_0 - g
```

所以实现中应：

```text
system_H += H
system_rhs += H * A0 - grad
```

### 10.3 正定性处理

由于精确 Hessian 不全局正定，建议保留至少一种保护：

```text
1. line search
2. diagonal regularization
3. fallback to Gauss-Newton
4. cap affine step
```

在已经验证的 `Rotating Gate` 中，`V_perp` 的精确 Hessian 表现很好；但在堆叠中仍可能失稳，因此它还不是全场景默认解。

## 11. 小结

`||AA^T-I||_F^2` 是 ABD 论文中的核心刚体化势能。它的关键性质是：

- 它是光滑势能，不是 AL 等式约束。
- 梯度为：

```math
\nabla_A E = 4k(AA^T-I)A
```

- 精确 Hessian block 为：

```math
H_{ip}
=
4k\left[
\delta_{ip}\sum_j a_j a_j^T
+
a_p a_i^T
+
S_{ip}I
\right]
```

- Hessian 在正交矩阵处半正定且有旋转零空间。
- Hessian 不全局正定，例如在 `A=0` 处有负方向。
- 因此精确 Hessian 需要配合 line search 或正定化策略。

在 ABD 研究中，它比直接 `polar_rotation` 后处理更一致，也比 naive OrthoAL 更稳定，是目前更贴近论文的主线方向。
