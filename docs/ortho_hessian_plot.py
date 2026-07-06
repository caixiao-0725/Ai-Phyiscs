"""画出正交势能 Hessian 三类模态曲率, 以及 PSD 区域边界。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = 1.0
out = r"d:\physics\newton-ChysX\ChysX\docs\ortho_hessian_curvature.png"

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ---- 左: A = sI, 三类曲率 vs s ----
ax = axes[0]
s = np.linspace(0, 1.6, 400)
stretch = 4 * K * (3 * s**2 - 1)          # 也是 sym-shear (A=sI 时相等)
rotation = 4 * K * (s**2 - 1)
ax.axhline(0, color="k", lw=0.8)
ax.plot(s, stretch, label=r"stretch & sym-shear: $4k(3s^2-1)$", lw=2)
ax.plot(s, rotation, label=r"rotation: $4k(s^2-1)$", lw=2, color="crimson")
ax.axvline(1/np.sqrt(3), ls="--", color="gray", alpha=0.7)
ax.axvline(1.0, ls="--", color="crimson", alpha=0.7)
ax.text(1/np.sqrt(3), 12.5, r"$s=1/\sqrt{3}$", rotation=90, va="top", fontsize=9)
ax.text(1.0, 12.5, r"$s=1$", rotation=90, va="top", fontsize=9, color="crimson")
ax.fill_between(s, -100, 100, where=(s < 1.0), color="red", alpha=0.05)
ax.set_xlabel("scale s   (A = s I)")
ax.set_ylabel("Hessian eigenvalue (curvature)")
ax.set_title("A = sI: rotation mode turns negative once s<1 (red = not PSD)")
ax.set_ylim(-6, 14)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

# ---- 右: (sig_i, sig_j) 平面内 rotation 模态符号 ----
ax = axes[1]
g = np.linspace(0, 1.6, 300)
SI, SJ = np.meshgrid(g, g)
rot = SI**2 + SJ**2 - SI*SJ - 1          # rotation 模态 (除以 4k)
sym = SI**2 + SJ**2 + SI*SJ - 1          # sym-shear 模态
cs = ax.contourf(SI, SJ, np.sign(rot), levels=[-1.5, 0, 1.5],
                 colors=["#ffcccc", "#cce5ff"], alpha=0.8)
ax.contour(SI, SJ, rot, levels=[0], colors="crimson", linewidths=2)
ax.contour(SI, SJ, sym, levels=[0], colors="navy", linewidths=1.5, linestyles="--")
ax.plot([1], [1], "k*", ms=15, label="orthogonal (1,1)")
ax.set_xlabel(r"$\sigma_i$")
ax.set_ylabel(r"$\sigma_j$")
ax.set_title("one (i,j) plane: red=rotation=0, blue dashed=sym-shear=0\nred region: rotation negative curvature")
ax.legend(loc="upper right")
ax.set_aspect("equal")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(out, dpi=130)
print("saved", out)
