# PABD 摩擦处理方法分析

> 基于 `PABDConstraintSolver.cu` 的源码分析

## 1. 总体架构：双层摩擦模型

PABD 对摩擦的处理分为两个独立的层次，分别在不同的求解阶段实施：

| 层次 | 阶段 | 作用 | 对应函数 |
|------|------|------|----------|
| **Position-level 静态摩擦** | Schur 位置求解 | 阻止切向滑动（位置修正） | `PABDS2_CalcSumsForMat1Friction` / `CalcSumsForMat2Friction` |
| **Velocity-level 动态摩擦** | Jacobi 速度求解 | 限制切向速度（脉冲修正） | `JacobiIteration` + Coulomb 锥 |

---

## 2. Position-level 静态摩擦（核心创新）

### 2.1 基本思想

将摩擦建模为一个**切向距离约束**，纳入 PABD 的 12×12 Schur 求解框架。这不是传统的 Coulomb 摩擦锥，而是一个**位置级弹性约束**：如果接触点在切线方向上发生了位移，就施加一个恢复力将它拉回。

### 2.2 切向距离的计算

```cuda
// PABDS2_CalcSumsForMat1Friction / CalcSumsForMat2Friction

// 当前两个接触点的世界坐标
Coord x0 = qPartAGlobal[id0] + qPartBGlobal[id0] * r0;
Coord x1 = (id1 >= 0) ? qPartAGlobal[id1] + qPartBGlobal[id1] * r1 : r1;

// 法线方向（从body0的局部法线旋转到世界）
Coord normal_global = qPartBGlobal[id0] * constraints[tId].normal0;

// 两接触点之间的位移
Coord x10 = x1 - x0;

// 投影掉法线分量，得到切向位移
Coord dP_t = x10 - (x10.dot(normal_global) * normal_global);

// 切向距离
Real d = dP_t.norm();

// 切向方向（归一化）
dP_t.normalize();
Coord tangent_global = dP_t;
```

**关键点**：PABD 不预定义固定的切线方向，而是在**每次迭代中根据当前接触点的相对位移**动态计算切线方向。这避免了选择切线基的歧义。

### 2.3 摩擦的能量函数

PABD 对切向距离 $d_t$ 使用与法线碰撞相同的能量框架，但用一个**不同的刚度 `staticFrictionStiffness`**。摩擦约束被建模为一个 `Distance` 类型的约束（目标距离为 0）：

```cuda
PABDConstraint constraint = PABDConstraint(id0, id1);
constraint.stiffness = staticFrictionStiffness;  // e.g. 10.0
constraint.setDistance(r0, r1, 0.0f);  // target distance = 0
```

能量梯度通过 `positivePart(d)` 和 `negativePart(d)` 分解：

- **positivePart** $\alpha = h^2 \cdot (\text{stiffness} + \lambda / d)$：正定部分，进入 Hessian
- **negativePart** $\beta = -\text{stiffness} \cdot \text{distance} = 0$：负定部分，进入 RHS

对于摩擦（`distance=0`），`negativePart = 0`，所以 **RHS 中没有 `sum3`/`sum5`（β 项）的贡献**。

### 2.4 Hessian 和 RHS 的累积

摩擦约束的 Hessian 和 RHS 以**完全相同的 Schur 结构**累积到全局求和变量中：

**Hessian（Mat1）**：
```
sum0 += α            (标量权重)
sumB += α · r        (lever arm × 权重)
sumC += α · r        (同上)
sumD += α · r ⊗ r    (outer product)
```

**RHS（Mat2）**：

```cuda
// CalcSumsForMat2Friction - 与普通碰撞约束结构相同
Coord tangent0_l = qStarBGlobal[id0].transpose() * tangent_global;

// sum3: β · tangent_local  (但 β=0，所以实际无贡献)
Coord sum3_0 = beta * tangent0_l;

// sum4: α · x_j_local (目标点在惯性帧下的坐标)
Coord sum4_0 = alpha * x1_local;

// sum5: β · tangent_local ⊗ r  (无贡献)
Matrix sum5_0 = beta * dyadic(tangent0_l, r0);

// sum6: α · x_j_local ⊗ r
Matrix sum6_0 = alpha * dyadic(x1_local, r0);
```

**关键理解**：摩擦约束的 sum4/sum6 贡献将接触点拉向对方 body 的接触点位置（沿切线方向），从而阻止切向滑动。

### 2.5 Line Search 与 rotFixed

在静态摩擦的 Line Search 步中，`rotFixed = true`：

```cpp
cuExecute(bodyNum, PABDS2_LineSearch, ..., false, true);
//                                       trans   rot
```

这意味着**摩擦只修正平移，不修正旋转**。这是合理的，因为摩擦力主要影响切向平移。

### 2.6 执行流程

在 `constrain()` 的主循环中：

```
for each iteration:
    1. 碰撞约束求解 (updateStatesFromConstraints(mContactConstraints))
    2. 距离/关节约束求解 (updateStatesFromConstraints(mDistanceConstraints))
    3. if staticFrictionEnabled:
       a. 保存当前状态为 q_Static
       b. 以 q_Static 为参考帧，构建摩擦的 Hessian/RHS
       c. 求解摩擦更新（仅修正平移）
       d. 变换回全局帧
```

注意：摩擦求解**在碰撞和关节求解之后**进行，作为一个额外的修正步。

---

## 3. Velocity-level 动态摩擦

### 3.1 框架

PABD 在位置求解之前（或之后，取决于 VelocitySolverEnabled），通过子步（SubStepping）运行一个 Jacobi 迭代速度求解器：

```
for i in SubStepping:
    applyGravity
    updateVelocity(external impulse)
    initializeJacobian
    for j in VelocityIterations:
        JacobiIteration (with Coulomb friction cone)
    updateVelocity(contact impulse)
    updateGesture
```

### 3.2 Coulomb 锥

`JacobiIteration` 中（在 `SharedFuncsForRigidBody.cu`），每个接触约束有法线和两个切线方向。速度求解器计算法线脉冲 $\lambda_n$，然后将切线脉冲限制在 Coulomb 锥内：

$$|\lambda_t| \leq \mu \cdot |\lambda_n|$$

其中 $\mu$ 来自 `DynamicFrictionCoefficient` × `FrictionCoefficients[body]`。

### 3.3 双摩擦模型的关系

| 属性 | 位置级静态摩擦 | 速度级动态摩擦 |
|------|----------------|----------------|
| 作用 | 阻止已接触点的滑动 | 限制接触面的相对速度 |
| 数学模型 | 二次能量约束（距离=0） | Coulomb 锥（脉冲限制） |
| 刚度控制 | `StaticFrictionStiffness` | `DynamicFrictionCoefficient` |
| 求解框架 | Schur 12×12 | Jacobi 迭代 |
| 计算代价 | 低（融入已有框架） | 高（需要额外迭代） |

---

## 4. 如何应用到我们的二次能量框架

### 4.1 设计方案

参考 PABD 的 Position-level 静态摩擦方案，在我们的 `stepCpuAffine` 中加入切向摩擦约束：

**核心思路**：对于每个接触点（地面或 box-box），计算接触点的**切向位移**（相对于上一次求解后的位置），然后添加一个 **"拉回" 目标**，将接触点在切线方向上拉回到无滑动的位置。

### 4.2 具体实现步骤

#### Step 1: 记录接触点的 anchor 位置

在每帧开始检测到接触时，记录接触点当前的世界坐标作为 "anchor"（参考点）。对于跨帧持久化的接触，anchor 保持不变。

#### Step 2: 在 Primal Update 中累积摩擦 Hessian/RHS

对于每个接触点：

```
// 计算当前接触点世界坐标
x_current = body.pos + body.affine * rLocal

// 计算切向位移
tangent_disp = (x_current - anchor) - dot(x_current - anchor, normal) * normal

// 如果 |tangent_disp| > 0:
//   目标：将接触点拉回 anchor 的切线位置
//   x_target = x_current - tangent_disp
//   转到局部帧：x_target_local = ATildeT * (x_target - cTilde)
//   累积到 Schur: schur.accumulate(frictionStiffness, rLocal)
//                 srcC += x_target_local * frictionStiffness
//                 srcA += outer(x_target_local, rLocal) * frictionStiffness
```

#### Step 3: 参数

- `frictionStiffness`：控制摩擦强度的刚度参数。PABD 使用 `staticFrictionStiffness = 10.0`
- 可选：基于 Coulomb 系数动态调整——如果法线力为 $F_n$，则限制切向力 $F_t \leq \mu F_n$

### 4.3 相比 PABD 的简化

PABD 用 `positivePart/negativePart` 分解能量梯度，并有 line search。我们可以简化：
- 直接用二次罚函数 $\Psi = \frac{1}{2} k_f |\Delta p_t|^2$
- 不需要 line search（penalty 方法自然稳定）
- `negativePart` 为 0（因为目标距离为 0）

### 4.4 与地面/box-box 的统一

地面摩擦和 box-box 摩擦可以用同一套框架：
- 地面：法线 = (0,0,1)，anchor = 接触点投影到地面的 xy 坐标
- Box-box：法线 = basis[0]，anchor = other body 的接触点坐标（初次检测时记录）
