# PABD vs ChysX 堆栈稳定性差异分析

## 核心问题

ChysX 使用了更精确的 Hessian（12×12 block Schur complement with Cholesky）和 Augmented Lagrangian
对偶更新，理论上应该比 PABD 收敛更快。但实际测试中，50 层堆叠在 ChysX 中长期不稳定，而 PABD
的 20 层堆叠几乎无震荡。为什么"更精确"反而"更不稳定"？

---

## 1. 架构差异总览

| 特性 | PABD (PeriDyno) | ChysX (当前实现) |
|------|-----------------|-----------------|
| **Solver 阶段** | 速度求解 → 位置求解（两阶段） | 纯位置求解（单阶段 AL） |
| **碰撞模型** | IPC 三次 barrier 能量 | 增广拉格朗日 (AL)：κ·C + λ |
| **Hessian** | 4×4 block inverse (3 标量 + 3×3 矩阵) | 12×12 block Schur + Cholesky |
| **碰撞 penalty 更新** | **固定不变** (stiffness = 0.01) | 加性增长 κ += β·\|C\| |
| **碰撞 λ 更新** | **无**（仅 barrier 梯度） | 每次迭代更新 λ = κ·C + λ |
| **重力处理** | 速度阶段处理（impulse-based） | 编码进 inertial prediction |
| **行搜索** | 有（backtracking line search） | 无 |
| **极分解时机** | 行搜索内，每步投影 | 全部迭代结束后投影一次 |
| **迭代结构** | 20 全局 × (联合 + 碰撞) 分开求解 | N 次 AL (primal + dual 交替) |

---

## 2. 关键差异分析

### 2.1 碰撞力模型：Barrier vs Augmented Lagrangian

这是最核心的差异。

**PABD 的 IPC Barrier**：

```
E_contact(d) = k · [(1-γ) + 0.5·(1-γ)² + 0.333·(1-γ)³],  γ = d / dHat
```

- 当 `γ < 1`（即 `d < dHat`）时，能量非零且梯度指向分离方向
- **约束力不依赖 λ**，只依赖当前穿透深度 `d`
- **stiffness 固定不变**（0.01），无需 dual update
- 当 `d ≥ dHat` 时，能量和力均为 0，约束完全不活跃

**ChysX 的 Augmented Lagrangian**：

```
F = κ · C(x) + λ
```

- 力同时依赖穿透量 `C(x)` **和**对偶变量 `λ`
- `λ` 在 dual update 中逐步积累：`λ ← min(κ·C + λ, 0)`
- `κ` 按 `κ += β·|C|` 增长，**依赖穿透量大小**

**关键问题**：当相邻盒子精确接触（gap = 0）且 λ = 0 时：
- PABD：如果 `d < dHat`，barrier 产生非零力 → **可以立即建立接触力**
- ChysX：`F = κ·0 + 0 = 0` → **碰撞力完全为零**

### 2.2 Penalty 增长策略的致命影响

**PABD**：碰撞 stiffness 完全固定，不增长。这避免了 AL 中经典的"penalty 太小力不够 vs penalty
太大系统过刚"的两难困境。固定 barrier 的曲率提供一致的有效刚度。

**ChysX**：penalty 按 `κ += β·|C|` 增长：
- 当 solver 将 gap 消得很干净（|C| ≈ 0）时，penalty 几乎不涨
- 当 solver 没消干净（|C| 较大）时，penalty 暴涨，导致下一帧 Hessian 过刚

这是 AL 方法的经典困境。实验数据证实：

```
[DUAL f1 BC] maxPen=1.0  maxLam=0.0000   ← 第1帧 penalty 和 lambda 都为零
[DUAL f2 BC] maxPen=1.0  maxLam=0.0000   ← 第2帧仍然为零
[DUAL f3 BC] maxPen=60.5 maxLam=0.0069   ← 第3帧才开始增长
```

50 层堆栈需要数十帧才能建立足够的 penalty/lambda，期间积累了大量动能。

### 2.3 速度阶段 vs 纯位置求解

这是稳定性差异的**最大来源**。

**PABD 的速度阶段** (10 子步 × 50 Jacobi 迭代 = 500 次速度校正)：
1. 施加重力冲量
2. 碰撞检测
3. 用 Baumgarte 误差校正的投影 Jacobi 求解速度约束
4. 积分得到预测位置

速度阶段做了什么？**在位置求解开始之前，速度阶段已经消解了大部分重力-碰撞矛盾**。进入位置
solver 时，body 的预测位置已经大致平衡——位置 barrier 只需微调。

速度阶段的 Jacobi 迭代是 **全局耦合的**：每个 body 的约束通过速度变量直接相互影响。这意味着
底层地面的约束力可以在 50 次迭代中传播到顶层。

**ChysX 的纯 AL**：
- 没有速度阶段
- 重力编码进 inertial prediction：所有 body 以相同 g·dt² 下移
- 相邻 body 间 gap 始终为 0（对称性），碰撞力为 0
- 只有地面约束打破对称性
- GS 传播每层衰减 ~10 倍

实验证据：

```
[GS f1 it0 b0]  cL.z = 0.001389   ← 底层被地面修正
[GS f1 it0 b1]  cL.z = 0.000000   ← 第1层零修正
[GS f1 it49 b1] cL.z = 0.000137   ← 50次迭代后第1层修正：衰减10倍
[GS f1 it49 b2] cL.z = 0.000013   ← 第2层：衰减100倍
[GS f1 it49 b3] cL.z = 0.000001   ← 第3层：衰减1000倍
```

**100 次 GS 迭代只能有效传播 3 层。** 这不是 Hessian 精度的问题，而是 **GS 在链式耦合系统中
的固有传播瓶颈**。

### 2.4 Hessian 精度反而有害？

直觉上 "更精确的 Hessian 应该更好"，但在此场景中恰恰相反：

**PABD 的 4×4 block**：
- 不区分 3 个方向的 penalty
- 标量 Hessian 权重 `α = h²·B''(d)/dHat²`
- 相当于各向同性弹簧
- 计算的位移较大，传播更快

**ChysX 的各向异性 12×12 block**：
- 3 个方向分别有不同的 penalty（法向、切向1、切向2）
- 法向 penalty 很大时，Hessian 在法向极其"刚硬"
- 求解出的位移几乎只沿法向——**锁死效应**
- 传播被高刚度"阻断"

实验证实：penalty_min 从 1 提高到 100000 后，传播反而更差：

```
penalty_min=1:      b1 cL.z = 0.000137
penalty_min=100000: b1 cL.z = 0.000003   ← 更差！
```

### 2.5 Line Search 的作用

**PABD 有 backtracking line search**：
```
α = E_old / (E_new + ε)
α = min(α, 1) × relaxCoeff
```

这保证了每一步都不会增加总能量，即使 Hessian 近似不精确，也能维持单调下降。

**ChysX 没有 line search**：primal update 直接使用牛顿步，如果 Hessian 条件数差（penalty
高时可达 10⁶+），步长可能过大导致能量增加。

---

## 3. 根因层级总结

按重要性排序：

### 第一层：重力-碰撞耦合方式（决定性差异）

PABD 的速度阶段（500 次全局 Jacobi）在位置求解前已经建立了正确的支撑力分布。ChysX 的纯位置
AL 在第一帧面对的是"50 个 body 同步自由落体，gap 全部为 0，碰撞力全部为 0"的死局。

**这不是 Hessian 精度能解决的问题。** 你可以有世界上最精确的 Hessian，但如果 RHS（约束违反量）
为零，求解出的修正量就是零。

### 第二层：碰撞力模型（AL vs Barrier）

Barrier 在 `d < dHat` 时无条件产生力（不依赖 λ），相当于一个内置的 "预接触力"。AL 需要 λ
积累后才能产生非零力，而 λ 的增长依赖 penalty × gap，在 gap ≈ 0 时增长极慢。

### 第三层：Solver 结构导致的传播瓶颈

ChysX 的 per-body GS 在链式系统中传播极慢（每层 ~10 倍衰减）。PABD 的速度阶段用 Jacobi
（所有约束并行处理）天然具有全局传播能力；位置阶段的 global solve（所有约束一起积累到 per-body
Hessian）也具有更好的传播性。

### 第四层：Line search 和数值稳定性

PABD 的 line search 保证能量单调下降。ChysX 缺少 line search，在 penalty 高时可能产生过大
的牛顿步。

---

## 4. "更精确的 Hessian" 为什么没有帮助

精确 Hessian 的优势是：**给定正确的 RHS，能算出最优的步长和方向**。

但在堆栈场景中：
1. RHS 为零（gap = 0, λ = 0）→ 精确 Hessian 算出的结果也是零
2. 当 penalty 增大后，精确 Hessian 的条件数极差 → 步长受限
3. 各向异性 penalty 导致 "法向锁死"，反而阻碍了力的传播

PABD 用的是更"粗糙"的 4×4 block，但由于：
- Barrier 能量自动提供非零梯度
- 速度阶段已经建立力分布
- Line search 限制了每步的冒险程度

反而在实践中更稳定。**这是 "准确性" 和 "鲁棒性" 的经典权衡**——在约束不充分或初始化不良的
条件下，一个保守但稳定的方法比一个精确但脆弱的方法更可靠。

---

## 5. 解决方案及效果

### 已实施：Gravity-aware Lambda Warm-start

在每帧 AL 迭代开始前，通过自顶向下的重力载荷传播为每个碰撞约束预设 λ 初值：

```
对于高度排序后的每个 body（从顶到底）：
  cumLoad[body] = body.mass × |g| × dt²   // 自身重力
  perContact = cumLoad[body] / 下方接触数
  对于每个下方碰撞约束:
    λ_init = -perContact
    cumLoad[下方body] += perContact         // 载荷传递
```

这样底层 body 的 λ 包含了上面所有 body 的总重量，第一次 primal update 就能产生正确量级的
约束力，打破了 "gap=0 → F=0" 的死局。

**效果**：50 层堆栈从 f1200 开始完全静止（vel ≈ 0），总下沉量 ~0.73（1.5%）。

### 潜在改进方向

1. **添加速度阶段预处理**：参照 PABD，在位置求解前用一个轻量级的速度 Jacobi 迭代建立力分布
2. **切换为 Barrier 碰撞模型**：使用 IPC-style barrier 替代 AL，消除 λ 初始化问题
3. **加入 line search**：回溯行搜索防止 penalty 高时的过大步长
4. **混合迭代策略**：前几次迭代用较低 penalty 建立初步力分布，后续再增大 penalty 精化
