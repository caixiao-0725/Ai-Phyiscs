# AVBD CPU Joint / Hinge Constraint 实现分析

## 1. 先说结论

当前 CPU AVBD 里有一个 `Joint` 类，但它严格来说不是传统意义上的“单轴 hinge”。

它现在实现的是：

```text
ball-socket position constraint + optional angular lock
```

也就是说：

- `stiffnessLin > 0`：约束两个局部 anchor 点重合或保持初始相对位移。
- `stiffnessAng > 0`：约束两个刚体保持初始相对姿态。
- 默认 `stiffnessAng = 0`，所以默认只是 ball-socket，不锁相对旋转。

如果想让两个刚体“视为一体”，当前推荐用法是：

```cpp
new Joint(s, bodyA, bodyB, rA, rB, INFINITY, INFINITY);
new IgnoreCollision(s, bodyA, bodyB);
```

其中 `IgnoreCollision` 很重要，否则两个被固定连接的部件会互相碰撞，导致 joint 和 collision 打架。

## 2. 现有 Joint 的数据结构

定义在：

```text
src/rigid/avbd_cpu/avbd_solver.h
```

核心成员：

```cpp
struct Joint : Force {
    float3 rA, rB;
    float3 C0Lin, C0Ang;
    float3 penaltyLin, penaltyAng;
    float3 lambdaLin, lambdaAng;
    float stiffnessLin, stiffnessAng, fracture;
    float torqueArm;
    bool broken;
    bool restInitialized;
};
```

含义：

- `rA`, `rB`: 两个刚体上的局部连接点。
- `C0Lin`: 初始连接点相对位移。
- `C0Ang`: 初始相对姿态误差。
- `penaltyLin`, `lambdaLin`: 线性约束的 AL penalty / multiplier。
- `penaltyAng`, `lambdaAng`: 角度约束的 AL penalty / multiplier。
- `stiffnessLin`: 线性约束上限。
- `stiffnessAng`: 角度约束上限。
- `restInitialized`: body-body joint 是否已经记录初始 rest pose。

## 3. 初始化：记录 rest pose

代码位置：

```text
src/rigid/avbd_cpu/avbd_joint.cpp
```

初始化时：

```cpp
C0Lin = bodyA->transformVec(rA, rmode)
      - bodyB->transformVec(rB, rmode);
```

也就是两个 anchor 点的初始世界空间相对位置。

角度部分：

```cpp
if (rmode == RotationMode::Affine) {
    float3x3 qA = bodyA ? bodyA->affine : identity3x3();
    float3x3 qB = bodyB->affine;
    C0Ang = mat_to_angular(qA, qB) * torqueArm;
} else {
    C0Ang =
        ((bodyA ? bodyA->positionAng : quat{0, 0, 0, 1})
         - bodyB->positionAng) * torqueArm;
}
```

这里 `C0Ang` 表示初始相对姿态。

一个重要修复是：

```cpp
if (!restInitialized || bodyA == nullptr) {
    ...
    if (bodyA != nullptr)
        restInitialized = true;
}
```

也就是说：

- body-body joint 只在第一次初始化时记录 rest pose；
- mouse drag 这类 world-anchor joint 仍然每帧刷新目标。

如果每帧刷新 body-body joint 的 `C0Ang`，fixed joint 就会失去“保持初始角度”的含义。

## 4. 线性约束：anchor 点不分离

线性残差：

```cpp
float3 C = bodyA->transformVec(rA, rmode)
         - bodyB->transformVec(rB, rmode);

if (std::isinf(stiffnessLin))
    C -= C0Lin * alpha;
```

如果两个 anchor 初始重合，那么：

```text
C0Lin = 0
```

约束就变成：

```text
bodyA anchor == bodyB anchor
```

对应的 Jacobian：

```cpp
jLin = +I or -I
jAng = skew(-rWorld) or skew(rWorld)
```

也就是标准 ball-socket 约束：线速度和角速度都会影响连接点位置。

## 5. 角度约束：保持相对姿态

角度残差：

```cpp
if (rmode == RotationMode::Affine) {
    C = mat_to_angular(qA, qB) * torqueArm;
} else {
    C = (qA - qB) * torqueArm;
}

if (std::isinf(stiffnessAng))
    C -= C0Ang * alpha;
```

如果：

```cpp
stiffnessAng = INFINITY
```

则目标是保持初始相对姿态。

如果：

```cpp
stiffnessAng = 0
```

则不约束相对姿态。这时两个刚体可以绕连接点任意相对旋转。

## 6. 当前 Joint 和真正 hinge 的区别

传统 hinge / revolute joint 的意思通常是：

```text
两个刚体共享一个连接点；
只允许绕指定 hinge axis 相对旋转；
禁止其他两个角方向的相对旋转。
```

因此 true hinge 有：

```text
3 个 anchor 位置约束
2 个角度约束
= 5 个约束
```

剩下 1 个自由度是绕 hinge axis 的旋转。

而当前 `Joint` 有两种用法：

### 6.1 Ball-socket

```cpp
new Joint(s, A, B, rA, rB, INFINITY, 0.0f);
```

约束数：

```text
3 个位置约束
```

允许：

```text
任意相对旋转
```

这不是 fixed，也不是严格 hinge。它更像球铰。

### 6.2 Fixed joint

```cpp
new Joint(s, A, B, rA, rB, INFINITY, INFINITY);
```

约束数：

```text
3 个位置约束 + 3 个角度约束 = 6 个约束
```

允许：

```text
不允许任何相对运动
```

这适合“两个刚体视为一体”。

但它不是单轴 hinge，因为它不保留相对转动自由度。

## 7. 为什么你会看到绕长轴旋转

你观察到：

> 两个刚体绕着自身比较长的轴旋转。

这可能有几种原因。

### 7.1 如果使用的是默认 Joint

默认构造函数是：

```cpp
Joint(..., stiffnessLin = INFINITY, stiffnessAng = 0.0f)
```

这意味着只约束连接点，不约束相对姿态。

此时两个长方体当然可以绕连接点旋转，也可能表现为绕长轴 twist。

### 7.2 如果想固定成一个整体，但内部碰撞没关

两个固定连接的刚体如果仍然互相碰撞，会出现：

```text
joint 想维持相对姿态；
collision 想把两段推开；
两者互相打架。
```

这会产生非物理扭矩，看起来像绕长轴转。

所以复合刚体内部必须加：

```cpp
new IgnoreCollision(s, A, B);
```

### 7.3 当前角度约束不是严格 Lie group 相对旋转约束

当前 axis-angle 模式下，角度误差是：

```cpp
C = (qA - qB) * torqueArm;
```

其中 `operator-(quat a, quat b)` 实际返回：

```cpp
(a * inverse(b)).vec() * 2.0f
```

这是一种小角度近似。对于大角度冲击、强接触、长杆高惯量情况，它不如严格的：

```text
log(R_A R_B^T R_rest^T)
```

稳定和准确。

所以 fixed joint 虽然可以工作，但在剧烈碰撞中仍可能出现一定相对姿态漂移。

### 7.4 刚体整体绕长轴旋转是允许的

如果两个刚体真的被 fixed joint 绑定为整体，那么它们仍然可以作为一个整体绕任意轴旋转。

这不是 joint 失效。

需要区分：

```text
整体旋转：允许
相对旋转：fixed joint 不应该允许
```

可以用诊断量区分：

```text
angleDeg   = 两个杆局部长轴之间的夹角
anchorErr  = 两个连接点之间的距离
relAngVel  = 两刚体相对角速度
```

如果：

```text
angleDeg 基本恒定
anchorErr 接近 0
relAngVel 接近 0
```

说明 fixed joint 是工作的。即使整体在转，那也是复合刚体整体转动。

## 8. 当前诊断结果

对 `AVBD: CPU Fixed Pair Drop` 加诊断后，修复 rest pose 和设置 `alpha=1` 前，观察到：

```text
angleDeg 从 60° 附近逐渐变到接近 0°
```

原因是 `C0Ang` 每帧被刷新，并且 `alpha=0.95` 使目标相对角度被松弛。

修复后，平放掉落场景诊断为：

```text
angleDeg  = 60.000
anchorErr = 0.000000
relAngVel ≈ 0
```

说明 fixed joint 在温和场景下可以保持相对姿态。

但把复合体立起来以后，落地冲击更强，曾观察到：

```text
angleDeg 从 60° 漂到 38° 左右
```

这说明当前角度约束在强冲击下仍可能不够强或误差形式不够准确。

## 9. 如果要实现真正 hinge，应该怎么做

如果目标不是 fixed joint，而是真正 hinge，推荐新增 `HingeJoint`。

### 9.1 数据

```cpp
struct HingeJoint : Force {
    float3 rA, rB;          // anchor points
    float3 axisA, axisB;    // local hinge axes
    float3 C0Lin;
    float stiffnessLin;
    float stiffnessAng;
    float3 lambdaLin;
    float2 lambdaAng;
    float3 penaltyLin;
    float2 penaltyAng;
};
```

### 9.2 位置约束

同 ball-socket：

```text
xA(rA) - xB(rB) = C0Lin
```

提供 3 个约束。

### 9.3 角度约束

设世界 hinge axis：

```text
u = R_A axisA
v = R_B axisB
```

hinge 要求两个轴对齐：

```text
u × v = 0
```

但 `u × v` 有 3 个分量，其中只有 2 个独立约束。

可以构造两个垂直于 `u` 的基向量 `t1, t2`，然后约束：

```text
C1 = t1 · v = 0
C2 = t2 · v = 0
```

这样会禁止除 hinge axis 以外的相对旋转，但允许绕 hinge axis 的 twist。

### 9.4 轴角/旋转向量下的小扰动公式

AVBD CPU 里的角度自由度本质上用的是“小角度旋转向量”更新。

如果刚体当前旋转为：

```math
R
```

给它施加一个小旋转增量：

```math
\delta\theta \in \mathbb{R}^3
```

则一阶近似为：

```math
R' \approx \exp([\delta\theta]_\times) R
      \approx (I + [\delta\theta]_\times)R
```

其中：

```math
[\delta\theta]_\times x = \delta\theta \times x
```

对一个局部方向 `a`，世界方向为：

```math
u = R a
```

扰动后：

```math
u' = R'a
   \approx (I + [\delta\theta]_\times)Ra
   = u + \delta\theta \times u
```

所以方向向量的一阶变化是：

```math
\boxed{
\delta u = \delta\theta \times u = -[u]_\times \delta\theta
}
```

这个公式是 hinge 角度 Jacobian 的核心。

## 9.5 Hinge 轴对齐约束的残差

给定两个刚体的局部 hinge 轴：

```math
a_A,\quad a_B
```

对应世界轴：

```math
u = R_A a_A
```

```math
v = R_B a_B
```

理想 hinge 要求两个轴平行：

```math
u \parallel v
```

等价于：

```math
u \times v = 0
```

但 `u × v` 有 3 个分量，秩只有 2。为了得到独立的两个角约束，构造两个与 `u` 垂直的单位切向量：

```math
t_1,\;t_2,\quad
t_1\cdot u = 0,\quad
t_2\cdot u = 0,\quad
t_1\cdot t_2 = 0
```

然后约束 `v` 在这两个方向上的分量为 0：

```math
\boxed{
C_1 = t_1^T v = 0
}
```

```math
\boxed{
C_2 = t_2^T v = 0
}
```

这两个约束只禁止 `v` 偏离 `u`，但不限制绕 `u` 的相对 twist。

因此它正好是 hinge 的 2 个角度约束。

## 9.6 Hinge 轴约束的 Jacobian

先忽略 `t_1,t_2` 随 `u` 的变化，把它们视作当前线性化点上的常量。这是常见的 Gauss-Newton/约束线性化做法。

对 body B 的角增量：

```math
\delta\theta_B
```

有：

```math
\delta v = \delta\theta_B \times v
```

残差：

```math
C_k = t_k^T v,\quad k=1,2
```

变化为：

```math
\delta C_k
=
t_k^T(\delta\theta_B \times v)
```

利用三重积恒等式：

```math
t^T(\delta\theta \times v)
=
(v \times t)^T \delta\theta
```

所以对 body B：

```math
\boxed{
\frac{\partial C_k}{\partial \theta_B}
=
(v \times t_k)^T
}
```

如果也考虑 body A 的旋转会改变 hinge 轴 `u`，更对称的写法可以用：

```math
C = u \times v
```

或者用 `t_k` 跟随 `u` 的切空间变化。工程上更简单稳定的做法是：每次迭代用当前 `u` 重新构造 `t_1,t_2`，但在本次线性化中把 `t_k` 固定。

此时 body A 的角 Jacobian 可以近似为 0；这会让约束主要推动 body B 轴对齐 body A 轴。为了对称地分配到两个刚体，更推荐使用下面的相对旋转误差形式。

## 9.7 用相对旋转构造 hinge 角误差

更严格的 hinge 写法是使用相对旋转：

```math
R_{rel} = R_A^T R_B
```

初始相对旋转为：

```math
R_{rel}^0 = (R_A^0)^T R_B^0
```

如果是 fixed joint，需要约束：

```math
\log(R_{rel}(R_{rel}^0)^T)=0
```

这是 3 个角约束。

如果是 hinge joint，需要允许绕 hinge 轴的相对旋转。假设 hinge 轴在 body A 局部坐标中为：

```math
a_A
```

则只约束相对旋转误差中垂直于 hinge 轴的两个分量：

```math
e = \log(R_{rel}(R_{rel}^0)^T)
```

取一组与 `a_A` 垂直的局部基：

```math
s_1,\;s_2
```

约束：

```math
\boxed{
C_1 = s_1^T e = 0
}
```

```math
\boxed{
C_2 = s_2^T e = 0
}
```

这样：

- `e` 沿 hinge axis 的分量允许存在；
- `e` 垂直 hinge axis 的分量被消除；
- 也就是只允许绕 hinge axis 相对转动。

### 9.7.1 小角度近似

如果相对旋转误差较小，可以近似：

```math
e \approx \theta_B - \theta_A - \theta_{rest}
```

于是：

```math
\frac{\partial C_k}{\partial \theta_A} \approx -s_k^T
```

```math
\frac{\partial C_k}{\partial \theta_B} \approx s_k^T
```

这和当前 `Joint` 里的 fixed angular Jacobian：

```cpp
jAng = +I or -I
```

是同一类小角度线性化，只是 hinge 只取 2 个方向，而 fixed joint 取 3 个方向。

### 9.7.2 为什么当前 fixed joint 会出现 twist 观察

如果使用默认 `Joint(..., INFINITY, 0)`，角度约束完全没开，所以绕长轴 twist 是允许的。

如果使用 `Joint(..., INFINITY, INFINITY)`，理论上 twist 不允许。但当前误差：

```cpp
(qA * inverse(qB)).vec() * 2
```

是小角度近似。强冲击、大旋转时，它可能不足以准确保持相对旋转。

真正 robust 的 fixed joint / hinge joint 都应该使用：

```math
\log(R_{rel}(R_{rel}^0)^T)
```

作为角误差。

### 9.8 fixed joint 与 hinge joint 的区别

Fixed joint：

```text
锁 3 个平移 + 3 个旋转
```

Hinge joint：

```text
锁 3 个平移 + 2 个旋转
保留 1 个绕 hinge axis 的旋转自由度
```

## 10. 如果当前目标是“视为一体”

如果你现在的目标仍然是：

> 两个长方体无相对运动，视为一体

那应该继续用 fixed joint，但建议加强它：

1. 保持：

```cpp
new Joint(s, a, b, rA, rB, INFINITY, INFINITY);
new IgnoreCollision(s, a, b);
```

2. 对 fixed pair 场景设置：

```cpp
s->alpha = 1.0f;
```

3. 可以增加迭代数：

```cpp
s->iterations = 40;
```

4. 如果强冲击下仍漂移，应改进角度误差形式：

```text
用 log(R_A R_B^T R_rest^T) 替代当前 quaternion vector 小角度误差
```

这是最关键的长期修复。

## 11. 总结

当前 CPU AVBD 的 `Joint`：

- 默认是 ball-socket；
- 传 `INFINITY, INFINITY` 时可作为 fixed joint；
- 不是严格 hinge；
- 强冲击下 fixed 角度漂移可能来自当前角度误差近似；
- 真正 hinge 应该单独实现 5 约束系统；
- 如果只是“两个刚体视为一体”，先用 fixed joint + ignore internal collision + alpha=1。
