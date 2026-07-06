"""
正交势能 E(A) = k * || A^T A - I ||_F^2 的正定性实验。

约定:
  - A 是 3x3 仿射矩阵 (按列展平为 9 维向量 vec(A), 列优先 Fortran order)。
  - S = A^T A - I
  - E(A) = k * tr(S^2) = k * sum_i (sigma_i^2 - 1)^2,  sigma_i 是 A 的奇异值

本脚本验证四件事:
  1. 解析梯度  grad = 4k A S         vs  有限差分梯度
  2. 解析 Hessian 二次型 Q(X)        vs  有限差分 Hessian
  3. SVD 坐标系下 9 个特征值的闭式公式 vs 数值 Hessian 的特征值
  4. 何时不正定: 沿 s=sI 扫描 + 随机矩阵统计 + PSD 区域边界

注: 用 A^T A 与用 A A^T 的能量值完全相同 (都依赖奇异值),
    Hessian 谱也完全相同, 所以结论对两种写法通用。
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=120)
K = 1.0  # stiffness, 不影响符号, 只整体缩放


# ---------------------------------------------------------------------------
# 能量 / 梯度 / Hessian 二次型
# ---------------------------------------------------------------------------
def energy(A, k=K):
    S = A.T @ A - np.eye(3)
    return k * np.sum(S * S)


def grad_analytic(A, k=K):
    S = A.T @ A - np.eye(3)
    return 4.0 * k * (A @ S)


def hess_quadform_analytic(A, X, k=K):
    """返回 Q(X) = (1/2) d^2/de^2 E(A + eX). 即 x^T H x / 2 不是; 见下方说明.

    实际上我们让 H 满足  E ~ E0 + <g,dA> + 1/2 <dA, H dA>,
    于是沿单位方向 X 的曲率 (二阶方向导数 E'') = <X, H X> = 2 Q(X)。
    这里返回 E'' 直接, 方便和数值 Hessian 的 x^T H x 对齐。
    """
    S = A.T @ A - np.eye(3)
    dS = X.T @ A + A.T @ X
    term_gn = np.sum(dS * dS)              # ||X^T A + A^T X||_F^2  (Gauss-Newton, >=0)
    term_curv = np.sum(S * (X.T @ X))      # <S, X^T X>            (可能为负)
    Q = k * term_gn + 2.0 * k * term_curv  # = (1/2) E''
    return 2.0 * Q                          # 返回 E'' = x^T H x


# ---------------------------------------------------------------------------
# 数值参考: 有限差分梯度 & Hessian (对 vec(A), 列优先)
# ---------------------------------------------------------------------------
def grad_numeric(A, k=K, h=1e-6):
    g = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            Ap = A.copy(); Ap[i, j] += h
            Am = A.copy(); Am[i, j] -= h
            g[i, j] = (energy(Ap, k) - energy(Am, k)) / (2 * h)
    return g


def hess_numeric(A, k=K, h=1e-4):
    """9x9 数值 Hessian, 基于解析梯度做中心差分 (更稳)。索引 = vec(A) 列优先。"""
    n = 9
    H = np.zeros((n, n))
    base_vecs = [np.zeros((3, 3)) for _ in range(n)]
    for idx in range(n):
        i, j = idx % 3, idx // 3      # 列优先: idx = i + 3*j
        E = np.zeros((3, 3)); E[i, j] = 1.0
        base_vecs[idx] = E
    for idx in range(n):
        E = base_vecs[idx]
        gp = grad_analytic(A + h * E, k)
        gm = grad_analytic(A - h * E, k)
        dg = (gp - gm) / (2 * h)          # = H * e_idx (重排成 3x3)
        H[:, idx] = dg.flatten(order='F')
    return 0.5 * (H + H.T)                # 数值对称化


# ---------------------------------------------------------------------------
# SVD 坐标系下的 9 个特征值闭式公式
# ---------------------------------------------------------------------------
def hess_eigs_closed_form(A, k=K):
    sig = np.linalg.svd(A, compute_uv=False)  # sigma_1 >= sigma_2 >= sigma_3 >= 0
    eigs = []
    labels = []
    # 3 个 stretch 模态 (改变单个奇异值)
    for i in range(3):
        eigs.append(4 * k * (3 * sig[i] ** 2 - 1))
        labels.append(f"stretch  sig{i}={sig[i]:.3f}")
    # 3 个对称剪切 + 3 个旋转 模态 (i<j 平面)
    for i in range(3):
        for j in range(i + 1, 3):
            si, sj = sig[i], sig[j]
            eigs.append(4 * k * (si**2 + sj**2 + si * sj - 1))
            labels.append(f"sym-shear ({i},{j})")
            eigs.append(4 * k * (si**2 + sj**2 - si * sj - 1))
            labels.append(f"rotation  ({i},{j})")
    return np.array(eigs), labels, sig


# ---------------------------------------------------------------------------
# 工具: 随机扰动方向 / 随机仿射矩阵
# ---------------------------------------------------------------------------
def random_X(rng):
    X = rng.standard_normal((3, 3))
    return X / np.linalg.norm(X)


def rand_rotation(rng):
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# ===========================================================================
# 实验 1: 梯度与 Hessian 二次型的解析 vs 数值
# ===========================================================================
def exp1_verify_derivatives():
    print("=" * 78)
    print("实验 1: 解析梯度 / Hessian 二次型  vs  有限差分")
    print("=" * 78)
    rng = np.random.default_rng(0)
    max_g, max_h = 0.0, 0.0
    for _ in range(200):
        A = rng.standard_normal((3, 3)) * rng.uniform(0.2, 1.5)
        eg = np.max(np.abs(grad_analytic(A) - grad_numeric(A)))
        max_g = max(max_g, eg)
        for _ in range(5):
            X = random_X(rng)
            # 数值 E'' 用能量二阶差分
            h = 1e-4
            Epp = (energy(A + h * X) - 2 * energy(A) + energy(A - h * X)) / h**2
            ea = abs(hess_quadform_analytic(A, X) - Epp)
            max_h = max(max_h, ea)
    print(f"  梯度       最大误差: {max_g:.3e}")
    print(f"  Hessian二次型 最大误差: {max_h:.3e}")
    print("  -> 解析公式正确\n" if max_g < 1e-5 and max_h < 1e-2 else "  -> 有偏差, 检查!\n")


# ===========================================================================
# 实验 2: 数值 Hessian 特征值  vs  SVD 闭式公式
# ===========================================================================
def exp2_verify_eigs():
    print("=" * 78)
    print("实验 2: 数值 9x9 Hessian 特征值  vs  SVD 闭式公式")
    print("=" * 78)
    rng = np.random.default_rng(1)
    max_err = 0.0
    for _ in range(300):
        A = rng.standard_normal((3, 3)) * rng.uniform(0.1, 1.8)
        Hn = hess_numeric(A)
        en = np.sort(np.linalg.eigvalsh(Hn))
        ec, _, _ = hess_eigs_closed_form(A)
        ec = np.sort(ec)
        max_err = max(max_err, np.max(np.abs(en - ec)))
    print(f"  排序后特征值最大误差: {max_err:.3e}")
    print("  -> 闭式公式给出了全部 9 个特征值\n" if max_err < 1e-3
          else "  -> 有偏差, 检查!\n")

    # 展示一个具体例子
    print("  示例 A = diag(1.2, 0.8, 0.3):")
    A = np.diag([1.2, 0.8, 0.3])
    ec, labels, sig = hess_eigs_closed_form(A)
    order = np.argsort(ec)
    for idx in order:
        flag = "  <-- 负曲率!" if ec[idx] < -1e-9 else ""
        print(f"    {labels[idx]:>18s} : eig = {ec[idx]:+.4f}{flag}")
    print()


# ===========================================================================
# 实验 3: 沿 A = s * I 扫描, 找符号翻转的阈值
# ===========================================================================
def exp3_scale_sweep():
    print("=" * 78)
    print("实验 3: A = s*I 扫描 (R=I), 观察各模态曲率随 s 变化")
    print("=" * 78)
    print("  预测阈值:")
    print("    stretch / sym-shear 转负:  s < 1/sqrt(3) = 0.5774")
    print("    rotation           转负:  s < 1")
    print(f"  {'s':>6} | {'E(s)':>9} | {'lambda_min(H)':>14} | {'PSD?':>5} | 首先转负的模态")
    print("  " + "-" * 70)
    for s in [0.0, 0.2, 0.4, 0.5, 0.5774, 0.6, 0.8, 0.9, 0.99, 1.0, 1.2, 1.5]:
        A = s * np.eye(3)
        ec, labels, _ = hess_eigs_closed_form(A)
        lam_min = np.min(ec)
        psd = "yes" if lam_min > -1e-9 else "NO"
        neg = labels[int(np.argmin(ec))] if lam_min < -1e-9 else "-"
        print(f"  {s:6.4f} | {energy(A):9.4f} | {lam_min:14.5f} | {psd:>5} | {neg}")
    print()
    # 解析: A=sI 时三类曲率
    print("  闭式 (A=sI): stretch=sym-shear=4k(3s^2-1),  rotation=4k(s^2-1)")
    print("  => 哪怕只是均匀缩小 s<1, rotation 模态先出现负曲率!\n")


# ===========================================================================
# 实验 4: 随机矩阵统计 + 最小奇异值与 PSD 的关系
# ===========================================================================
def exp4_random_statistics():
    print("=" * 78)
    print("实验 4: 随机仿射矩阵, 最小奇异值 sigma_min 与 PSD 的关系")
    print("=" * 78)
    rng = np.random.default_rng(7)
    N = 20000
    bins = {}
    psd_count = 0
    # 收集: 在不同 sigma_min 区间, PSD 比例
    edges = np.linspace(0, 1.6, 17)
    counts = np.zeros(len(edges) - 1)
    psd_in_bin = np.zeros(len(edges) - 1)
    for _ in range(N):
        scale = rng.uniform(0.1, 1.6)
        A = rng.standard_normal((3, 3))
        A = A / np.linalg.svd(A, compute_uv=False)[0] * scale  # 控制最大奇异值=scale
        ec, _, sig = hess_eigs_closed_form(A)
        smin = sig[2]
        is_psd = np.min(ec) > -1e-9
        psd_count += is_psd
        b = np.searchsorted(edges, smin) - 1
        if 0 <= b < len(counts):
            counts[b] += 1
            psd_in_bin[b] += is_psd
    print(f"  总样本 {N}, PSD 比例 = {psd_count / N:.3f}")
    print(f"  {'sigma_min 区间':>16} | {'样本数':>6} | {'PSD 比例':>8}")
    print("  " + "-" * 42)
    for b in range(len(counts)):
        if counts[b] == 0:
            continue
        rng_str = f"[{edges[b]:.2f},{edges[b+1]:.2f})"
        print(f"  {rng_str:>16} | {int(counts[b]):6d} | {psd_in_bin[b]/counts[b]:8.3f}")
    print("\n  结论: sigma_min 远大于 1 -> 几乎总是 PSD;")
    print("        sigma_min < 1 -> rotation 模态易出现负曲率; sigma_min<0.577 几乎必有负曲率。\n")


# ===========================================================================
# 实验 5: PSD 区域的精确边界 (闭式条件)
# ===========================================================================
def exp5_psd_region():
    print("=" * 78)
    print("实验 5: 全局 PSD 的充要条件 (奇异值)")
    print("=" * 78)
    print("  Hessian PSD  <=>  下面 9 个量全部 >= 0:")
    print("    (a) stretch  : 3*sig_i^2 - 1 >= 0                  (每个 i)")
    print("    (b) sym-shear: sig_i^2 + sig_j^2 + sig_i*sig_j - 1 >= 0  (每对 i<j)")
    print("    (c) rotation : sig_i^2 + sig_j^2 - sig_i*sig_j - 1 >= 0  (每对 i<j)  <- 最易破坏")
    print()
    print("  特例:")
    print("    A=0    : 所有曲率 = -4k  -> 负定 (强负曲率, 鞍点/极大)")
    print("    A=sI   : rotation=4k(s^2-1) 在 s<1 即负; stretch 在 s<0.577 才负")
    print("    A 正交 : 全部 >=0, 且 rotation 模态恰为 0 (so(3) 零空间), 半正定奇异")
    print()
    # 数值确认正交点零空间维数
    rng = np.random.default_rng(3)
    R = rand_rotation(rng)
    Hn = hess_numeric(R)
    eig = np.sort(np.linalg.eigvalsh(Hn))
    nzero = int(np.sum(np.abs(eig) < 1e-6))
    print(f"  随机旋转 R 处 Hessian 特征值:\n    {eig}")
    print(f"    零特征值个数 = {nzero}  (= dim so(3) = 3, 对应无穷小旋转)\n")


if __name__ == "__main__":
    exp1_verify_derivatives()
    exp2_verify_eigs()
    exp3_scale_sweep()
    exp4_random_statistics()
    exp5_psd_region()
