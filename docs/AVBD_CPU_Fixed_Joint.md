# AVBD CPU 固定铰接 Joint 实现说明

## 1. 结论

当前 `avbd_cpu` 里已经有可以实现“两个刚体无相对运动，视为一体”的约束。

对应类型是：

```cpp
Joint
```

定义位置：

```text
src/rigid/avbd_cpu/avbd_solver.h
src/rigid/avbd_cpu/avbd_joint.cpp
```

它不是只做普通 hinge 的单自由度铰链，而是一个更通用的约束：

```text
ball-socket position constraint + angular relative-orientation constraint
```

如果只开线性约束，它是球铰/点铰：

```cpp
new Joint(s, bodyA, bodyB, rA, rB, INFINITY, 0.0f);
```

如果同时开线性和角度约束，它就是固定连接：

```cpp
new Joint(s, bodyA, bodyB, rA, rB, INFINITY, INFINITY);
```

对于你说的：

> 两个长方体成一个角度铰接在一起，所谓铰接，就是两刚体无相对运动，视为一体。

应该使用：

```cpp
new Joint(s, bodyA, bodyB, rA, rB, INFINITY, INFINITY);
```

## 2. Joint 的数据结构

`Joint` 在 `avbd_solver.h` 中定义：

```cpp
struct Joint : Force {
    float3 rA, rB;
    float3 C0Lin, C0Ang;
    float3 penaltyLin, penaltyAng;
    float3 lambdaLin, lambdaAng;
    float stiffnessLin, stiffnessAng, fracture;
    float torqueArm;
    bool broken;
};
```

各变量含义：

- `rA`: 连接点在 bodyA 局部坐标中的位置。
- `rB`: 连接点在 bodyB 局部坐标中的位置。
- `C0Lin`: 初始化时记录的连接点世界空间相对位移。
- `C0Ang`: 初始化时记录的两个刚体初始相对姿态。
- `penaltyLin`: 线性约束的 AL penalty。
- `penaltyAng`: 角度约束的 AL penalty。
- `lambdaLin`: 线性约束的拉格朗日乘子。
- `lambdaAng`: 角度约束的拉格朗日乘子。
- `stiffnessLin`: 线性约束上限。
- `stiffnessAng`: 角度约束上限。
- `fracture`: 断裂阈值。
- `torqueArm`: 角度约束缩放因子。

## 3. 初始化逻辑

`Joint::initialize()` 中会记录当前帧初始约束值。

线性部分：

```cpp
C0Lin = (bodyA ? bodyA->transformVec(rA, rmode) : rA)
      - bodyB->transformVec(rB, rmode);
```

意思是：

```text
C0Lin = bodyA 上的连接点世界位置 - bodyB 上的连接点世界位置
```

如果两个连接点一开始重合，那么：

```text
C0Lin = 0
```

角度部分：

```cpp
if (rmode == RotationMode::Affine) {
    float3x3 qA = bodyA ? bodyA->affine : identity3x3();
    float3x3 qB = bodyB->affine;
    C0Ang = mat_to_angular(qA, qB) * torqueArm;
} else {
    C0Ang = ((bodyA ? bodyA->positionAng : quat{0, 0, 0, 1}) - bodyB->positionAng) * torqueArm;
}
```

意思是：

```text
C0Ang = 初始相对姿态
```

所以固定 joint 并不是强制两个刚体姿态相同，而是强制它们保持初始化时的相对姿态。

这正适合“两个长方体以某个角度固定连接”的情况。

## 4. 线性约束：连接点不分离

`Joint::updatePrimal()` 中的线性约束：

```cpp
float3 C = (bodyA ? bodyA->transformVec(rA, rmode) : rA)
         - bodyB->transformVec(rB, rmode);

if (std::isinf(stiffnessLin))
    C -= C0Lin * alpha;

float3 F = K * C + lambdaLin;
```

如果：

```cpp
stiffnessLin = INFINITY
```

则使用增广拉格朗日形式保留初始连接点偏移：

```text
C - alpha * C0Lin
```

在 `alpha` 接近 1 时，它会把连接点拉回初始相对位置。

对于固定连接，通常希望：

```text
C0Lin = 0
```

也就是两个 body 的连接点初始重合。

## 5. 角度约束：相对姿态不改变

角度约束部分：

```cpp
if (rmode == RotationMode::Affine) {
    float3x3 qA = bodyA ? bodyA->affine : identity3x3();
    float3x3 qB = bodyB->affine;
    C = mat_to_angular(qA, qB) * torqueArm;
} else {
    C = ((bodyA ? bodyA->positionAng : quat{0, 0, 0, 1}) - bodyB->positionAng) * torqueArm;
}

if (std::isinf(stiffnessAng))
    C -= C0Ang * alpha;

float3 F = K * C + lambdaAng;
```

如果：

```cpp
stiffnessAng = INFINITY
```

则 joint 会保持两个刚体的初始相对姿态。

这就是“无相对转动”的关键。

如果 `stiffnessAng = 0`，两个刚体只共享连接点，但可以绕连接点相对旋转，这更像球铰，而不是固定连接。

## 6. Dual Update

线性部分：

```cpp
if (std::isinf(stiffnessLin)) {
    C -= C0Lin * alpha;
    float3 F = K * C + lambdaLin;
    lambdaLin = F;
}

penaltyLin = min(
    penaltyLin + abs(C) * solver->betaLin,
    min(stiffnessLin, AVBD_PENALTY_MAX)
);
```

角度部分：

```cpp
if (std::isinf(stiffnessAng)) {
    C -= C0Ang * alpha;
    float3 F = K * C + lambdaAng;
    lambdaAng = F;
}

penaltyAng = min(
    penaltyAng + abs(C) * solver->betaAng,
    min(stiffnessAng, AVBD_PENALTY_MAX)
);
```

这说明 `Joint` 和 contact 一样，也是用 adaptive penalty + lambda 的增广拉格朗日风格在迭代中逐步变硬。

## 7. 如何做两个成角度长方体固定连接

假设有两个长方体：

```cpp
Rigid* a = new Rigid(s, {4.0f, 0.4f, 0.4f}, 1.0f, 0.5f, {0, 0, 4});
Rigid* b = new Rigid(s, {4.0f, 0.4f, 0.4f}, 1.0f, 0.5f, {0, 0, 4});
```

我们希望它们在端点处连接，并形成一个固定角度，比如 60 度。

### 7.1 设置初始姿态

例如让 `a` 沿世界 x 方向，`b` 绕 z 轴旋转 60 度。

```cpp
a->positionAng = {0, 0, 0, 1};
a->syncFromQuat();

float angle = rad(60.0f);
b->positionAng = {0, 0, std::sin(angle * 0.5f), std::cos(angle * 0.5f)};
b->syncFromQuat();
```

### 7.2 设置连接点

如果 `a` 的右端连接 `b` 的左端：

```cpp
float3 rA = {+2.0f, 0.0f, 0.0f};  // a 的局部右端
float3 rB = {-2.0f, 0.0f, 0.0f};  // b 的局部左端
```

为了让两个端点初始重合，需要设置 `b` 的中心位置。

```cpp
float3 jointWorld = a->positionLin + rotate(a->positionAng, rA);
b->positionLin = jointWorld - rotate(b->positionAng, rB);
```

### 7.3 添加固定 Joint

关键代码：

```cpp
new Joint(
    s,
    a,
    b,
    rA,
    rB,
    INFINITY,   // stiffnessLin: 连接点固定
    INFINITY    // stiffnessAng: 相对姿态固定
);
```

这会让两个长方体：

- 连接点不分离；
- 初始相对角度保持不变；
- 整体行为接近一个复合刚体。

## 8. 推荐场景函数示例

```cpp
void setupAVBDCPUFixedAnglePair(Solver* s) {
    s->clear();
    s->force_cpu = true;
    s->rotation_mode = RotationMode::AxisAngle;
    s->set_ground_plane(0.0f, 0.5f);
    s->iterations = 20;

    Rigid* a = new Rigid(s, {4.0f, 0.4f, 0.4f}, 1.0f, 0.5f, {0.0f, 0.0f, 3.0f});
    Rigid* b = new Rigid(s, {4.0f, 0.4f, 0.4f}, 1.0f, 0.5f, {0.0f, 0.0f, 3.0f});

    a->positionAng = {0, 0, 0, 1};
    a->syncFromQuat();

    float angle = rad(60.0f);
    b->positionAng = {0, 0, std::sin(angle * 0.5f), std::cos(angle * 0.5f)};
    b->syncFromQuat();

    float3 rA = {+2.0f, 0.0f, 0.0f};
    float3 rB = {-2.0f, 0.0f, 0.0f};

    float3 jointWorld = a->positionLin + rotate(a->positionAng, rA);
    b->positionLin = jointWorld - rotate(b->positionAng, rB);

    new Joint(s, a, b, rA, rB, INFINITY, INFINITY);
}
```

## 9. 现有实现的注意事项

### 9.1 默认 Joint 不是固定连接

`Joint` 构造函数默认参数是：

```cpp
Joint(..., stiffnessLin = INFINITY, stiffnessAng = 0.0f, fracture = INFINITY)
```

所以默认只是：

```text
ball-socket / 点铰
```

不会锁定相对旋转。

想要“视为一体”，必须显式传：

```cpp
INFINITY, INFINITY
```

### 9.2 一个固定 Joint 通常足够

一个线性 anchor + 一个角度约束已经有 6 个约束自由度：

```text
3 个平移约束 + 3 个旋转约束
```

因此理论上足够固定两个刚体的相对运动。

### 9.3 如果只用线性 Joint，需要多个 anchor

如果不用 `stiffnessAng`，那一个 anchor 只能约束一个点。

这时两个刚体仍然可以绕该点相对旋转。

要用纯线性约束实现固定连接，至少需要多个不共线的连接点，比如：

```cpp
new Joint(s, a, b, rA0, rB0, INFINITY, 0.0f);
new Joint(s, a, b, rA1, rB1, INFINITY, 0.0f);
new Joint(s, a, b, rA2, rB2, INFINITY, 0.0f);
```

但当前代码已经支持角度约束，所以推荐一个 fixed joint 即可。

### 9.4 CPU AVBD 路径

对于你现在要做 CPU AVBD demo，应设置：

```cpp
s->force_cpu = true;
s->rotation_mode = RotationMode::AxisAngle;
```

这样会走：

```cpp
Solver::stepCpuAVBD()
```

而不是 GPU narrowphase/solver。

## 10. 总结

当前 CPU AVBD 已经有实现“两个刚体无相对运动”的机制：

```cpp
new Joint(s, bodyA, bodyB, rA, rB, INFINITY, INFINITY);
```

它的含义是：

- `rA` 和 `rB` 两个局部 anchor 保持初始相对位置；
- bodyA 和 bodyB 保持初始相对姿态；
- 两个刚体整体表现为一个固定连接的复合刚体。

所以不需要从零实现新的 fixed joint。后续只需要写一个场景函数，构造两个有初始夹角的长方体，并添加该 Joint 即可。
