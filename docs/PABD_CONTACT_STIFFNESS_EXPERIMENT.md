# PABD PF/EE Contact Stiffness Experiment

## 1. 要验证的两个假设

### H1：碰撞四面体体积是合适的接触质量

当前实现为

```text
V_hat = |det(p1-p0, p2-p0, p3-p0)| / 6 * d_hat / d
k     = beta * rho * V_hat / h^2
```

它至少满足两个必要条件：

- 量纲正确：`rho * V_hat` 是质量。
- 整体缩放 `x -> s*x`、`d_hat -> s*d_hat` 时，`V_hat -> s^3*V_hat`，与刚体质量同阶缩放。

但它还没有自动满足网格无关性。PF 的 `V_hat` 使用目标三角形面积，却由源点产生候选；只细分源网格时，候选数量可能增加而每个候选的面积权重不减。EE 还有

```text
V_hat_EE = |e0 x e1| * d_hat / 6
```

因此刚度会随夹角按 `sin(theta)` 下降。平行 EE 被丢弃没有问题，但近似平行的 EE 仍可能过软。

### H2：PF(VF) 刚度应为 EE 的 5 倍

代码中的正式类型名是 PF，和这里说的 VF 是同一种点面接触。现在定义

```text
k_PF = r * k_base
k_EE =     k_base
```

并扫描 `r = 1, 2, 5, 10`。

`r=5` 没有直接的几何推导。对边长同为 `L` 的直角 PF 和正交 EE，二者都有

```text
V_hat = L^2 * d_hat / 6
```

所以 5 倍若有效，它补偿的是候选采样、网格连接方式或 PF/EE 对系统约束覆盖率的差异，而不是四面体体积公式本身。

## 2. 已加入的 A/B 开关

Viewer 中：

- `self contact measure`: `effective mass` / `contact tetra volume`
- `PF(VF) stiffness / EE`: `0 ... 10`

生产默认值为 `effective mass`。`tetra_volume` 仅保留为显式 A/B 实验模式。

批处理环境变量：

```powershell
$env:CHYSX_PABD_SELF_CONTACT_MEASURE = 'tetra_volume' # or effective_mass
$env:CHYSX_PABD_PF_STIFFNESS_SCALE = '5'
$env:CHYSX_PABD_SELF_BETA = '16'
$env:CHYSX_PABD_CONTACT_FEATURES = 'both' # both, pf/vf, ee, or none
$env:CHYSX_PABD_DEBUG_INTERVAL = '1'
```

这些开关不修改场景的 `dt`、global-local 次数或 `PCG=50`。

## 3. 动态对照实验

自动脚本：

```powershell
.\tests\run_pabd_contact_stiffness_experiment.ps1
```

快速版：

```powershell
.\tests\run_pabd_contact_stiffness_experiment.ps1 -Quick
```

结果写到 `output/pabd_contact_stiffness_experiment.csv`。

实验场景：

| 场景 | 用途 |
|---|---|
| `CUDA PABD: Stacked Blocks` | PF/EE 混合、持续承重、观察堆叠压缩和抖动 |
| `CUDA PABD: Tetra EE Cross` | EE-only 负对照；改变 PF 比例后结果应基本不变 |
| `CUDA PABD: Rigid-IPC Chain Net 4x4` | 多体、多方向、碰撞拓扑不断变化 |
| `CUDA PABD: Torque Gear Line 2` | 高曲率表面和持续切向运动，检查传动及 hinge 误差 |

每个场景舍弃前 180 帧，再统计：

- 全程 `maxPen / contact_gap`
- 舍弃前 180 帧后的稳态 `maxPen / contact_gap` 与 `rmsPen / contact_gap`
- 活跃 PF/EE 数量
- PF/EE 平均 `k` 和平均 `V_hat`
- `minY`、齿轮角速度和最大 hinge endpoint error
- 平均帧时间、P95、NaN、接触容量溢出

### 公平比较 effective mass 与 tetra volume

不能直接固定同一个 beta 比较。二者定义的“单接触质量”数值不同，应分两步：

1. 保留同 beta 的原始结果，观察量级差异。
2. 在 `Stacked Blocks` 的参考帧把平均或总接触刚度标定一致，再把该 beta 原样用于其他场景。

当前短跑中，`beta=16` 时体积模式平均单接触刚度约 `4e3`，有效质量模式约 `5.2e4`；脚本暂用 `effective beta=1.25` 作为第一版标定值。

## 4. 几何不变性实验

动态场景只能说明“当前资产上好不好用”，还必须补下面三组几何实验，才能判断公式是否可发表为通用接触权重。

### A. 尺度扫描

将同一个双刚体接触整体缩放为 `s = 0.5, 1, 2, 4`，同时缩放 `contact_gap` 和 `thickness`，保持密度、时间步长和 beta 不变。

合格标准：`penetration/contact_gap` 的变化小于 10%。

### B. 网格细分扫描

使用同一个 ABD 控制四面体和相同总质量，只改变碰撞表面：每个 box 面分别使用 `2, 8, 32, 128` 个三角形。

必须分别测试：

- 只细分 PF 的源点一侧。
- 只细分目标三角形一侧。
- 两侧同步细分。

合格标准：总接触刚度、平衡压缩量及归一化穿透随细分层级变化小于 10%。只要“单侧细分”明显变硬，就说明原始碰撞四面体体积不是完整的面积求积权重，需要改成源点 nodal area / edge support area 一类的权重。

### C. EE 夹角扫描

固定边长、质量和间距，只改变两条边夹角 `theta = 15, 30, 45, 60, 90` 度。

当前公式预言 `k_EE proportional sin(theta)`。需要判断这种衰减是否符合目标离散模型；若归一化穿透随角度显著恶化，应把“几何覆盖面积”和“约束强度”分开，而不是继续用单个 PF/EE 魔法比例修补。

## 5. 判定规则

`PF/EE=5` 只有同时满足以下条件才保留为默认值：

1. 在至少两个 PF/EE 混合训练场景中显著降低归一化穿透或抖动。
2. 在未参与选择的 chain/gear 场景中仍改善，而不是只对 box 网格有效。
3. EE-only 场景结果不变，证明开关没有误乘 EE。
4. 无 NaN、无容量溢出，PCG 代价和平均帧时间增幅可接受。
5. 网格细分后最优比例不发生大幅漂移。

若最优比例随资产、细分或接触角度变化，`5` 就只能作为场景调参。此时更值得继续研究的是一致的 surface quadrature/contact patch measure，而不是寻找另一个全局常数。

## 6. 第一轮实测结果

测试条件：Release、每帧采样、每组 3 次、600 帧（EE Cross 为 360 帧），保持各场景原有 `dt`、global-local 次数和 `PCG=50`。下表为 3 次均值。

| 场景/模式 | PF/EE | 全程 maxPen/gap | 稳态 RMS/gap | 平均 k | mean ms | 关键结果 |
|---|---:|---:|---:|---:|---:|---|
| Stack, tetra volume | 1 | 0.460 | 0.059 | 3888 | 1.624 | 基线 |
| Stack, tetra volume | 2 | 0.425 | 0.047 | 4689 | 1.613 | 稳态穿透下降 |
| Stack, tetra volume | 5 | 0.452 | 0.029 | 7360 | 1.616 | 稳态 RMS 约为基线一半 |
| Stack, tetra volume | 10 | 0.421 | 0.020 | 11815 | 1.604 | 仍继续改善，没有在 5 处出现最优点 |
| Stack, effective mass, beta=16 | 1 | 0.165 | 0.067 | 51769 | 1.583 | 刚度约大 13 倍，但固定 50 次 PCG 下稳态反而略差 |
| Stack, effective mass, beta=1.25 | 1 | 0.458 | 0.057 | 4058 | 1.590 | 与体积模式标定到同量级后，结果基本相当 |
| EE Cross, EE-only | 1 | 3.605 | 0.011 | 481 | 1.579 | 发生瞬时深穿透并最终掉穿 |
| EE Cross, EE-only | 5 | 3.605 | 0.011 | 481 | 1.580 | 与比例 1 数值完全一致，证明 PF 倍率未污染 EE |
| Chain4, tetra volume | 1 | 2.500 | 0.772 | 118 | 1.294 | 基线 |
| Chain4, tetra volume | 5 | 2.499 | 0.657 | 147 | 1.269 | 稳态 RMS 下降约 15%，瞬时最大穿透未改善 |
| Chain4, tetra volume | 10 | 2.491 | 0.567 | 190 | 1.315 | 继续改善 |
| Gear2, tetra volume | 1 | 10.994 | 2.768 | 45 | 1.089 | 最终齿轮速度比 0.714 |
| Gear2, tetra volume | 5 | 10.997 | 3.278 | 85 | 1.086 | 穿透更差，最终速度比 1.774，明显过补偿 |
| Gear2, tetra volume | 10 | 9.107 | 1.822 | 91 | 1.085 | 最终速度比 0.939，在该扫描中最好 |

所有组均 `nan=0`、`overflow=0`。倍率本身没有可测的固定 kernel 成本；表中的时间差主要来自接触轨迹和系统噪声。

### 当前结论

1. `rho*V_hat` 的量纲与整体尺度缩放逻辑是对的，单元测试也验证了长度缩放 `s` 后体积按 `s^3` 缩放。
2. 在当前固定 box 网格上，把 effective mass 的 beta 标定到相同平均刚度后，它与 tetra volume 的结果接近；这组实验还不能证明 tetra volume 更优。
3. `PF=5*EE` 对 Stack 和 Chain4 有利，但对 Gear2 不利，而且 Stack/Chain 在 10 倍时仍继续改善，因此 5 不是跨场景常数。
4. EE-only Cross 会发生 `3.6*gap` 的瞬时穿透并掉穿。混合场景中提高 PF 权重可能只是在掩盖 EE 路径过软或短时漏接触，而不是修好了 EE。
5. 是否保留碰撞四面体体积，最终必须由“单侧网格细分”和“EE 夹角扫描”决定；这是下一轮优先级最高的实验。

原始数据位于 `output/pabd_contact_stiffness_experiment.csv`。

## 7. Chain Net 8x8 长期回归

在保持该场景当前 `dt=0.0033`、global-local 次数和 `PCG=30` 不变的条件下，对两种接触度量各运行 5000 帧：

| 模式 | frame 1000 minY | frame 2500 minY | frame 5000 minY | 结果 |
|---|---:|---:|---:|---|
| effective mass | -2.145 | -2.997 | -4.226 | 约 4000 帧后平台化，链网仍被锚点维持 |
| tetra volume | -3.279 | -5.426 | -376.137 | 约 2500 帧开始脱链，之后自由落体 |

体积模式中单接触平均刚度仅约 `50...115`，有效质量模式约 `1.2e5...1.4e5`。在现有 `beta=16` 下，两种 measure 的数量级并不兼容，碰撞四面体体积权重是 8x8 掉链的直接诱因。因此默认已恢复为 effective mass；体积模式不能直接沿用同一个 beta 投入链网场景。
