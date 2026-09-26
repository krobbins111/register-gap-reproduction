"""Concept figure + animation: what the two learned maps do to a collection.

A 2-D cartoon that is a faithful instance of the model: five caption/image
pairs, a text map phi_T(x) = normalize(x + W x + b_T) and an image map
phi_I(x) = normalize(x + W x + b_I) with a common stretch A = I + W that
amplifies one axis and suppresses the other, and different offsets. The
"after" arrangement is designed first (pairs adjacent, spread along the
amplified axis); the "raw" arrangement is its exact pre-image under the maps,
so the raw picture shows what the model implies: both clouds compressed along
the amplified axis (crowded), the small pair offsets blown up along the
suppressed axis (mismatched neighbours), and a global offset between the
caption and image clouds (the register/modality offset the biases remove).

The animation moves every point along x_t = x + t (W x + b), t in [0, 1],
which is exactly the map turned on gradually (the zero-initialised W and b of
training grow from 0).

Outputs  results/figs/concept_maps[_collapse].pdf/.png   (three panels: raw, maps, after)
         results/figs/concept_maps[_collapse].gif        (raw -> after, ~4 s loop)
         'tidy' = five pairs on a line; 'collapse' = eight scattered pairs, image
         side stretched harder, so the raw gallery is a collapsed blob.
Usage    python analysis\\make_concept.py [--variant tidy|collapse|both] [--no-gif]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from config import RESULTS_ROOT  # noqa: E402
except Exception:  # standalone use
    RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS_ROOT.parent / "figs"; OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "figure.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.facecolor": SURF})

# ── the model instance ───────────────────────────────────────────────────────
def make_instance(variant="tidy"):
    """Two variants of the same construction. 'tidy': five pairs on a line
    (the first figure). 'collapse': eight pairs scattered irregularly after
    the maps, with a stronger image-side stretch, so the raw gallery is a
    severe collapse (a blob) and the raw captions a tight cone; nearest
    images are wrong for most captions."""
    m = {}
    theta = np.deg2rad(22)
    U = np.array([np.cos(theta), np.sin(theta)]); V = np.array([-np.sin(theta), np.cos(theta)])
    R = np.stack([U, V], axis=1)
    if variant == "tidy":
        sT = (1.8, 0.35); sI = (1.8, 0.35)
        b_T = np.array([0.16, -0.14]); b_I = np.array([-0.07, 0.05])
        N = 5; centre = np.array([0.05, -0.05])
        along = (np.arange(N) - (N - 1) / 2) * 0.36
        eps_q = np.array([0.04, -0.03, 0.035, -0.04, 0.03]); eps_g = -eps_q * 0.8; SIDE = 0.10
        Q_after = centre + np.outer(along, U) + np.outer(eps_q + SIDE, V)
        G_after = centre + np.outer(along + 0.05, U) + np.outer(eps_g - SIDE, V)
        suffix = ""
    else:
        rng = np.random.default_rng(11)
        sT = (2.2, 0.40); sI = (3.4, 0.32)          # the image side is stretched harder: it starts more collapsed
        b_T = np.array([0.28, -0.02]); b_I = np.array([-0.15, -0.05])
        N = 8; centre = np.array([0.0, -0.02])
        # irregular positions after the maps: spread along U with uneven gaps, scattered along V
        along = np.linspace(-1.0, 1.0, N) + rng.uniform(-0.07, 0.07, N)
        v_q = rng.uniform(-0.16, 0.16, N); v_g = v_q + rng.uniform(-0.04, 0.04, N)
        SIDE = 0.15                                   # caption above / image below the true-pair line
        Q_after = centre + np.outer(along, U) + np.outer(v_q + SIDE, V)
        G_after = centre + np.outer(along + rng.uniform(-0.03, 0.06, N), U) + np.outer(v_g - SIDE, V)
        suffix = "_collapse"
    A_T = R @ np.diag(sT) @ R.T; A_I = R @ np.diag(sI) @ R.T
    m.update(theta=theta, U=U, V=V, N=N, suffix=suffix,
             W_T=A_T - np.eye(2), W_I=A_I - np.eye(2), b_T=b_T, b_I=b_I, S_AMP=max(sT[0], sI[0]), S_SUP=min(sT[1], sI[1]),
             Q_after=Q_after, G_after=G_after,
             Q_raw=(Q_after - b_T) @ np.linalg.inv(A_T).T, G_raw=(G_after - b_I) @ np.linalg.inv(A_I).T)
    return m


M = make_instance("tidy")


def at(t, X, W, b):
    """x_t = x + t (W x + b): the map turned on gradually."""
    return X + t * (X @ W.T + b)


def nn_correct(Q, G):
    d = ((Q[:, None, :] - G[None, :, :]) ** 2).sum(-1)
    return int((d.argmin(1) == np.arange(len(Q))).sum())


# ── drawing ─────────────────────────────────────────────────────────────────
def draw_space(ax, Q, G, t, title=None, show_axes=True, caption=None):
    theta, U, V, N, S_AMP, S_SUP = M["theta"], M["U"], M["V"], M["N"], M["S_AMP"], M["S_SUP"]
    ax.set_xlim(-1.65, 1.65); ax.set_ylim(-1.25, 1.25); ax.set_aspect("equal"); ax.axis("off")
    # the unit-ish sphere, stretched with t to hint at the re-metric
    w, h = 2.9, 2.2
    ax.add_patch(Ellipse((0, 0), w * (1 + 0.12 * t), h * (1 - 0.12 * t), angle=np.rad2deg(theta) * t,
                         fill=False, ec=GRID, lw=1.2))
    for i in range(N):
        ax.plot([Q[i, 0], G[i, 0]], [Q[i, 1], G[i, 1]], color="#b3b2ab", lw=1.6, zorder=2)
    ms = 95 if N <= 5 else 70
    ax.scatter(Q[:, 0], Q[:, 1], s=ms, color=BLUE, edgecolors="white", lw=1.2, zorder=3)
    ax.scatter(G[:, 0], G[:, 1], s=ms, color=ORANGE, marker="s", edgecolors="white", lw=1.2, zorder=3)
    if show_axes:
        la = 1.05 * (1 + (S_AMP - 1) * t) / S_AMP + 0.45      # visual lengths
        ls = 0.85 * (1 - (1 - S_SUP) * t) + 0.12
        o = np.array([-1.15, -1.0])
        ax.add_patch(FancyArrowPatch(o, o + la * U, arrowstyle="-|>", mutation_scale=12, color=INK, lw=1.4))
        ax.text(*(o + 0.5 * la * U + 0.1 * V), "amplified axis", fontsize=7.5, color=INK, ha="center", va="bottom",
                rotation=np.rad2deg(theta), rotation_mode="anchor")
        o2 = np.array([1.05, 0.35])
        ax.add_patch(FancyArrowPatch(o2, o2 + ls * V, arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1.2))
        ax.text(*(o2 + ls * V + 0.06 * V), "suppressed axis", fontsize=7.5, color=MUTED, ha="center", va="bottom")
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, pad=6)
    if caption:
        ax.text(0, -1.2, caption, fontsize=7.8, color=INK2, ha="center", va="bottom")


def legend(ax, loc="lower center", ncols=1, **kw):
    ax.scatter([], [], s=60, color=BLUE, label="caption embedding")
    ax.scatter([], [], s=60, color=ORANGE, marker="s", label="image embedding")
    ax.plot([], [], color="#c9c8c0", lw=1.3, label="true pair")
    ax.legend(loc=loc, frameon=False, fontsize=7.2, handletextpad=0.5, borderaxespad=0.1, labelspacing=0.3,
              ncols=ncols, **kw)


def fig_static(clean=False):
    """clean=True: the paper form. No grey explanatory text (panel captions,
    training note, parameter count); titles, the two map equations, the
    side labels, the amplified / suppressed axis arrows and the legend stay.
    Saved as concept_maps<suffix>_clean."""
    fig = plt.figure(figsize=(7.2, 2.55 if clean else 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.78, 1], left=0.01, right=0.99, top=0.88 if clean else 0.9,
                          bottom=0.10 if clean else 0.1, wspace=0.03)
    Q_raw, G_raw, Q_after, G_after, N = M["Q_raw"], M["G_raw"], M["Q_after"], M["G_after"], M["N"]
    ax = fig.add_subplot(gs[0]); draw_space(ax, Q_raw, G_raw, 0.0, "Frozen encoder", show_axes=False,
                                             caption=None if clean else f"nearest image is the true one for {nn_correct(Q_raw, G_raw)} of {N} captions")
    cq, cg = Q_raw.mean(0), G_raw.mean(0)
    ax.text(min(cq[0], -0.2), min(Q_raw[:, 1].max() + 0.2, 1.12), "captions: tight cone", fontsize=8, color=BLUE,
            fontweight="bold", ha="center", va="bottom")
    ax.text(G_raw[:, 0].min() - 0.14, cg[1] - 0.1, "images: collapsed" if M["suffix"] else "images: crowded",
            fontsize=8, color=ORANGE, fontweight="bold", ha="right", va="center")
    # middle: the maps
    am = fig.add_subplot(gs[1]); am.axis("off"); am.set_xlim(0, 1); am.set_ylim(0, 1)
    ya, yt, y1, y2 = (0.36, 0.90, 0.72, 0.58) if clean else (0.5, 0.92, 0.80, 0.70)
    am.add_patch(FancyArrowPatch((0.08, ya), (0.92, ya), arrowstyle="-|>", mutation_scale=26, color=BLUE, lw=5))
    am.text(0.5, yt, "two linear maps, one per side", fontsize=8.6, fontweight="bold", color=INK, ha="center", va="top")
    am.text(0.5, y1, r"$\phi_t(q) = \mathrm{normalize}(q + W_t q + b_t)$", fontsize=8.6, ha="center", color=BLUE)
    am.text(0.5, y2, r"$\phi_i(g) = \mathrm{normalize}(g + W_i g + b_i)$", fontsize=8.6, ha="center", color=ORANGE)
    if not clean:
        am.text(0.5, 0.38, "trained on the collection's own\n(caption, image) pairs with a\ncontrastive loss; $W, b$ start at 0,\nso training starts at the identity",
                fontsize=7.4, ha="center", va="top", color=INK2, linespacing=1.35)
        am.text(0.5, 0.02, "$\\sim d^2$ parameters, seconds on one GPU", fontsize=7.2, ha="center", va="bottom", color=MUTED)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=7, label="caption embedding"),
               plt.Line2D([], [], marker="s", ls="", color=ORANGE, ms=7, label="image embedding"),
               plt.Line2D([], [], color="#b3b2ab", lw=1.6, label="true pair")]
    fig.legend(handles=handles, loc="lower center", ncols=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.6, handletextpad=0.5)
    ax2 = fig.add_subplot(gs[2]); draw_space(ax2, Q_after, G_after, 1.0, "After the learned maps",
                                              caption=None if clean else f"nearest image is the true one for {nn_correct(Q_after, G_after)} of {N} captions")
    name = f"concept_maps{M['suffix']}" + ("_clean" if clean else "")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print(f"[concept] -> {OUT / (name + '.pdf')}")


def fig_gif(n_frames=48, hold=14, fps=16):
    frames = []
    ts = np.concatenate([np.zeros(hold), (1 - np.cos(np.linspace(0, np.pi, n_frames))) / 2, np.ones(hold)])
    for t in ts:
        fig = plt.figure(figsize=(5.2, 4.2), dpi=110)
        ax = fig.add_axes([0.02, 0.08, 0.96, 0.8])
        Q, G = at(t, M["Q_raw"], M["W_T"], M["b_T"]), at(t, M["G_raw"], M["W_I"], M["b_I"])
        draw_space(ax, Q, G, t, show_axes=t > 0.02,
                   caption=f"nearest image is the true one for {nn_correct(Q, G)} of {M['N']} captions")
        legend(ax, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.06), columnspacing=1.2)
        title = "Frozen encoder" if t < 0.02 else ("After the learned maps" if t > 0.98 else "Applying the maps")
        fig.text(0.5, 0.955, title, fontsize=12, fontweight="bold", ha="center", va="center")
        fig.text(0.5, 0.905, r"$x_t = x + t\,(W x + b)$" + f"      t = {t:.2f}", fontsize=8.5, ha="center", color=INK2)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(buf.copy()))
        plt.close(fig)
    gp = OUT / f"concept_maps{M['suffix']}.gif"
    frames[0].save(gp, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=False)
    print(f"[concept] -> {gp} ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--variant", choices=["tidy", "collapse", "both"], default="both")
    ap.add_argument("--clean", action="store_true", help="also save the paper form without the grey explanatory text (*_clean)")
    a = ap.parse_args()
    global M
    for v in (["tidy", "collapse"] if a.variant == "both" else [a.variant]):
        M = make_instance(v)
        print(f"[concept] {v}: raw nearest-image correct {nn_correct(M['Q_raw'], M['G_raw'])}/{M['N']}; "
              f"after: {nn_correct(M['Q_after'], M['G_after'])}/{M['N']}")
        fig_static()
        if a.clean:
            fig_static(clean=True)
        if not a.no_gif:
            fig_gif()


if __name__ == "__main__":
    main()
