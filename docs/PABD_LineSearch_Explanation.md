# PABD Line Search 逻辑说明

本文解释 PeriDyno `PABDConstraintSolver` 和 ChysX CPU 版 `stepCpuPABD()` 中的 line search。对应代码位置：

- ChysX: `src/rigid/avbd_cpu/pabd_solver.cpp` 的 `// LineSearch`
- PeriDyno: `PABDConstraintSolver.cu` 的 `PABDS2_LineSearch`

## 1. 结论先说

当前 ChysX CPU PABD 已经做了 line search。

它不是传统优化里的 Armijo backtracking，也不会反复试探多个步长。它是一个一次性计算的阻尼步长：

$$
\alpha = \min\left(1,\; \frac{\Delta t^2 E}{a + b + \epsilon}\right)\cdot \text{relax}
$$

然后把 `CalcQ` 算出的局部解从当前状态往目标状态走一部分：

$$
c_L \leftarrow c_{\text{ref}} + \alpha(c_L^{\text{new}} - c_{\text{ref}})
$$

$$
R_L \leftarrow \mathrm{polar}\left(R_{\text{ref}} + \alpha(R_L^{\text{new}} - R_{\text{ref}})\right)
$$

直觉上，它是在说：

> `CalcQ` 给了一个“如果完全相信局部二次近似，应该跳到哪里”的候选解；line search 决定这一步跳多远，避免一下子走太猛。

## 2. PABD 每轮迭代在干什么

PABD 每一轮大致是：

1. `Transform2Global`

   把局部变量转成世界坐标，用来评估当前接触约束。

2. `CalcSumsForMat1`

   根据当前激活约束，累计近似 Hessian 的块：

   - `sumA`
   - `sumB`
   - `sumC`
   - `sumD`

3. `CalcMat1Inv`

   解析求解每个刚体局部 12 维系统的逆块。

4. `CalcSumsForMat2`

   累计 RHS/source 项：

   - `sum3`
   - `sum4`
   - `sum5`
   - `sum6`
   - `energy`

5. `CalcMat2`

   得到 `srcA` 和 `srcBm`。

6. `CalcQ`

   根据 `Mat1Inv * Mat2` 得到一个候选局部状态：

   - `cenLocal`
   - `rotLocal`

7. `LineSearch`

   不直接接受 `CalcQ` 的候选解，而是算一个步长 `alpha`，从当前局部状态向候选解移动。

## 3. 局部坐标中的状态

PABD 在预测状态 `q*` 附近求解。设：

$$
c = c_* + R_* c_L
$$

$$
R = R_* R_L
$$

其中：

- `qStarA` / `centerRef`: 预测中心 $c_*$
- `qStarB` / `rotRef`: 预测旋转 $R_*$
- `cenLocal`: 局部平移 $c_L$
- `rotLocal`: 局部旋转 $R_L$
- `cenGlobal`: 当前世界中心 $c$
- `rotGlobal`: 当前世界旋转 $R$

在 line search 开始时，代码先把当前世界状态重新转回局部参考系：

```cpp
float3x3 RT = transpose(qStarB[i]);
float3 cRef = RT * (cenGlobal[i] - qStarA[i]);
float3x3 rRef = RT * rotGlobal[i];
```

这里 `cRef` / `rRef` 是本轮 line search 的起点，`cenLocal` / `rotLocal` 是 `CalcQ` 算出的候选终点。

## 4. `a` 和 `b` 是什么

代码：

```cpp
float3 partA = cRef + rRef * sumB[i];
float a = abs(dot(partA - srcA[i], cenLocal[i] - cRef));
```

`a` 衡量平移方向上，这一步可能带来的线性化误差或残差规模。

再看旋转部分：

```cpp
sumDI = sumD[i] + inertiaMatrix;
partB = dyadic(cRef, sumC[i]) + rRef * transpose(sumDI);
b += abs((partB - srcBm[i]) * (rotLocal[i] - rRef));
```

`b` 衡量旋转矩阵方向上的对应规模。

所以：

$$
a+b
$$

可以理解为“如果从当前状态走到候选状态，这个步子在局部二次模型里有多激进”。

## 5. `c = energy * dt^2` 是什么

代码：

```cpp
float c = energy[i] * dt * dt;
float al = c / (a + bv + 1e-10f);
```

`energy[i]` 是当前 body 相关约束的 barrier energy 累加。对于 contact：

$$
E(d) = k\left((1-\gamma) + \frac{1}{2}(1-\gamma)^2 + \frac{1}{3}(1-\gamma)^3\right),
\quad \gamma = \frac{d}{\hat d}
$$

当接触越深，尤其 signed distance $d$ 为负时，`energy` 会变大。于是：

- 穿透越严重，`energy` 越大，允许步长越大。
- 当前候选步越激进，`a+b` 越大，允许步长越小。

这就是 line search 的核心平衡。

## 6. 为什么要乘 `dt^2`

PABD 的位置级优化里，约束能量和惯性项都被时间步缩放。前面 `CalcSumsForMat1/2` 中也用了：

```cpp
alpha = dt * dt * positivePart(d);
beta  = dt * dt * negativePart(d);
```

因此 line search 里用：

$$
\Delta t^2 E
$$

是为了让步长估计和前面的离散系统尺度一致。

## 7. 为什么最后要 clamp 到 1

代码：

```cpp
if (al > 1.0f) al = 1.0f;
al *= PABD_RELAX;
```

`CalcQ` 的候选解是这轮局部二次近似的完整步。如果 `alpha > 1`，就代表当前能量预算足够接受完整步，但算法不做 over-relaxation，所以最多走到候选解。

`PABD_RELAX` 是额外松弛系数。当前 BoxStack 里是 `1.0`，所以不额外缩小。

## 8. 为什么旋转后要做 polar

代码：

```cpp
matR = rRef + al * (rotLocal[i] - rRef);
rotLocal[i] = polar_rotation(matR);
```

矩阵线性插值后一般不再是合法旋转矩阵。比如列向量不再正交，行列式可能偏离 1。

所以需要 polar decomposition 把它投影回最近的旋转矩阵：

$$
R_L \leftarrow \mathrm{polar}(R_L)
$$

这和 PeriDyno 的：

```cpp
polarDecomposition(mat, R, 0.00001);
rotLocal[pId] = R;
```

是同一个逻辑。

## 9. 它和传统 line search 的区别

传统 line search 常见流程是：

1. 先尝试 $\alpha=1$
2. 如果能量没下降，就 $\alpha *= 0.5$
3. 反复评估直到满足条件

PABD 这里不是这样。

它没有重新计算真实总能量，也没有循环试探。它使用当前迭代中已经算好的 `energy`、`srcA/srcBm`、`sumA/sumB/sumC/sumD`，直接估计一个步长。因此更像：

> analytic damping / safeguarded update

而不是标准 backtracking line search。

## 10. 和这次稳定性 bug 的关系

这次 PABD 堆叠不稳的关键不是 line search 缺失，而是 signed contact distance 被破坏。

原来 CPU 版里有类似逻辑：

```cpp
if (d < 1e-10f) d = 1e-10f;
```

这会把深穿透的负 `d` 变成一个很小的正数。问题是：

- `negativePart(d)` 需要负 `d` 来产生更强的修正项。
- `energy(d)` 需要负 `d` 来反映深穿透的严重程度。
- line search 的分子 `energy * dt^2` 会被错误压小。

结果就是：约束看起来一直像浅接触，line search 给不出足够的修正，堆叠会慢慢穿地。

去掉 clamp 后，深穿透会得到更大的 `energy` 和 `negativePart`，line search 才能配合 `CalcQ` 把 body 拉回稳定接触区。

## 11. 读代码时的变量对照

| 数学含义 | ChysX 变量 | PeriDyno 变量 |
|---|---|---|
| 预测中心 $c_*$ | `qStarA` | `q_starPartA` / `centerRef` |
| 预测旋转 $R_*$ | `qStarB` | `q_starPartB` / `rotRef` |
| 当前局部中心 | `cRef` | `center` |
| 当前局部旋转 | `rRef` | `rot` |
| 候选局部中心 | `cenLocal` | `centerLocal` |
| 候选局部旋转 | `rotLocal` | `rotLocal` |
| Hessian 平移块 | `sumA` | `sumA` |
| Hessian cross 块 | `sumB`, `sumC` | `sumB`, `sumC` |
| Hessian 旋转块 | `sumD` | `sumD` |
| RHS 平移项 | `srcA` | `srcPartA` |
| RHS 旋转项 | `srcBm` | `srcPartB` |
| barrier 能量 | `energy` | `energy` |
| line search 步长 | `al` | `alpha` |

## 12. 一句话理解

PABD 的 line search 是一个基于当前 barrier energy 的单步阻尼器：

> 约束越严重，允许修正越大；候选步越激进，步长越小；旋转插值后再投影回合法旋转。

它的目标不是精确最小化能量，而是让 `CalcQ` 的局部解在接触、堆叠、多体耦合时不要一步走炸。
