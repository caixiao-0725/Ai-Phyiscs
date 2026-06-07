# ABD 仿射矩阵框架下的 Coulomb 摩擦

本文档记录 `stepCpuAffine()` 中实现的 Coulomb 摩擦算法。

## 1. 约束表示

每个接触点存储一个 3D 约束基底 `basis = {n, t₁, t₂}`（法线 + 两个正交切线）：

```
C = [C_n, C_t1, C_t2]^T
```

- **法线** `C_n`：穿透深度，与之前标量 gap 相同。
- **切线** `C_t1, C_t2`：接触点在切线方向的相对滑动量。

对应 3D Lagrange 乘子 `λ = [λ_n, λ_t1, λ_t2]` 和 3D 自适应 penalty `κ = [κ_n, κ_t1, κ_t2]`。

### Ground Contact 基底

固定基底，无需存储：

```
n  = (0, 0, 1)    — 法线（向上）
t₁ = (1, 0, 0)    — 切线 x
t₂ = (0, 1, 0)    — 切线 y
```

### Box-Box Contact 基底

由 SAT 碰撞检测返回的 `float3x3 basis`，`basis[0]` 为法线。

## 2. Hessian 累积（各向同性近似）

`AffineSchurSolver` 只支持各向同性累积 `accumulate(κ, r)`。保持不变，用最大 penalty 作为近似刚度：

```
κ_eff = max(κ_n, κ_t1, κ_t2)
```

用 `max` 而不是 `mean` 确保 Hessian 不被未激活的切线 penalty（初始值 1.0）拉低。

**理论依据**：增广拉格朗日中 Hessian 只影响 Newton 步长方向和大小（近似不影响收敛极限），`λ` 的 dual update 保证渐近收敛。代价是可能需要更多迭代。

## 3. RHS 构建（分离式 Penalty + Lambda）

RHS 由两部分组成，保持与 Hessian 的一致性：

### 3.1 Penalty 部分 — 推向目标位置

**Ground Contact**: 目标位置是**帧初始时的锚点** `(initPos.x, initPos.y, ground_z)`，而非当前帧的 `(worldPos.x, worldPos.y, ground_z)`。这确保 penalty 在切线方向也产生恢复力来阻止滑动：

```cpp
float3 initPos = body->initialLin + body->initialAff * gc.rLocal;
float3 xTarget = {initPos.x, initPos.y, ground_z};
srcC += ATildeT * (xTarget - cTilde) * κ_eff;
srcA += outer(ATildeT * (xTarget - cTilde), rLocal) * κ_eff;
```

**Box-Box Contact**: 推向法线方向的无穿透位置（与之前一致）：

```cpp
float3 xTarget = xSelf + normal * (-gap * sign);
```

### 3.2 Lambda 部分 — 3D Coulomb 投影乘子力

Lambda 存储了上一次 dual update 后的约束力，经 Coulomb 投影后加入 RHS：

```
F = project_coulomb(λ)
```

投影规则：
1. **法线单侧**：`F_n = min(λ_n, 0)`（只压不拉）
2. **Coulomb 锥**：`||F_t|| ≤ μ|F_n|`，超出时按比例缩放

然后沿 3 个基底方向加入 RHS（注意符号为 `-F`，因为 Schur solver 的 RHS 是目标值而非力）：

```cpp
for d in {n, t1, t2}:
    n_d_local = ATildeT * basis[d] * sign
    srcC += n_d_local * (-F[d])
    srcA += outer(n_d_local, rLocal) * (-F[d])
```

## 4. Dual Update

### 4.1 约束值计算

**Ground Contact**: 法线用当前 gap，切线用相对于帧开始位置的位移：

```
C_n = worldPos.z - ground_z
C_t1 = worldPos.x - initPos.x    (帧内 x 滑动)
C_t2 = worldPos.y - initPos.y    (帧内 y 滑动)
```

**Box-Box Contact**: 直接在当前基底中投影：

```
C_d = dot(xA - xB, basis[d]) + margin_d
```

### 4.2 力计算与 Coulomb 投影

```
F[d] = κ[d] * C[d] + λ[d],  d ∈ {n, t1, t2}
F_n = min(F_n, 0)            — 法线单侧

bounds = |F_n| * μ
tangMag = ||(F_t1, F_t2)||
if tangMag > bounds:          — 动摩擦（Coulomb 锥外）
    F_t *= bounds / tangMag
```

### 4.3 更新规则

```
λ ← F    (投影后的力直接作为新乘子)
```

**法线 penalty 增长**：穿透时增长

```
if F_n < 0:  κ_n += β * |C_n|
```

**切线 penalty 增长**：仅在**静摩擦**模式下增长

```
if tangMag ≤ bounds (静摩擦):
    κ_t1 += β * |C_t1|
    κ_t2 += β * |C_t2|
```

动摩擦时不增长切线 penalty，避免过硬导致的振荡。

## 5. 与 AVBD 轴角模式的对应关系

| AVBD (`avbd_manifold.cpp`) | ABD (`stepCpuAffine()`) |
|---|---|
| `J * dq` 增量约束 | 直接世界坐标约束值 |
| `J^T K J` 精确各向异性 Hessian | `κ_eff · I` 各向同性近似 |
| `rhsLin += J^T * F` | `srcC += ATildeT * basis[d] * sign * (-F[d])` |
| 6x6 LDL solver | 12x12 Schur solver |
| Ground: 无独立处理 | Ground: 帧初始锚点 + 3 方向力 |

核心等价关系：

```
n_d_L = Ã^T · basis[d] · sign
srcC += -F_d · n_d_L     ↔    rhsLin += J^T[:,d] · F_d
srcA += -F_d · (n_d_L ⊗ r)  ↔    rhsAng 仿射行贡献
```

## 6. Warm Start

`featureKey` 精确匹配 + `rA/rB` 体局部空间 fallback 不变。仅 lambda/penalty 从标量升级为 `float3`。

帧间衰减：`λ *= α·γ`，`κ *= γ` 后 clamp。

迭代间：直接继承（不衰减）。
