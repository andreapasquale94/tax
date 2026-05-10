#!/usr/bin/env python3
"""Plot the on-split snapshots produced by ellipticOrbit.

Reads `split_ic.csv`, `split_phase.csv`, `ads_final_leaves.csv`, and
`flow_image.csv` written by the C++ example and produces a 2-panel figure:

    left  — IC-space partition history.  The full IC box is shown as the
            unit square in normalised (δ_y, δ_vy) coordinates.  For each
            split event we shade the parent rectangle (the leaf about to
            be split) and overlay the two children with the split axis
            highlighted.  The colour of each split tracks the split index
            so the eye can follow the order in which the algorithm
            refines the partition.
    right — phase-space pushforward at t = T_orbit of the same boxes.
            The single-flow polygon (the image of the full IC box pushed
            forward without splitting) is drawn behind for context.  Each
            split event contributes three polygons: the parent (large,
            often wraps around itself near apoapsis) and the two
            children, which together cover the same region but are each
            individually well-approximated.

A side panel shows the converged piecewise approximation: the final ADS
leaves at T_orbit, every leaf coloured separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def _polygon(samples: np.ndarray, x_field: str = "x", y_field: str = "y") -> np.ndarray:
    order = np.argsort(samples["sample_idx"])
    return np.column_stack([samples[x_field][order], samples[y_field][order]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory containing the CSV files written by ellipticOrbit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: <script-dir>/ellipticBoxOnSplit.png)",
    )
    args = parser.parse_args()

    out_path = args.output or Path(__file__).resolve().parent / "ellipticBoxOnSplit.png"

    ref = _load(args.data_dir / "orbit_reference.csv")
    split_ic = _load(args.data_dir / "split_ic.csv")
    split_ph = _load(args.data_dir / "split_phase.csv")
    final = _load(args.data_dir / "ads_final_leaves.csv")
    flow_img = _load(args.data_dir / "flow_image.csv")

    split_ids = sorted(set(split_ic["split_idx"].astype(int).tolist()))
    n_splits = len(split_ids)

    fig = plt.figure(figsize=(16.0, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.6, 1.6])
    ax_ic = fig.add_subplot(gs[0, 0])
    ax_ph = fig.add_subplot(gs[0, 1])
    ax_fin = fig.add_subplot(gs[0, 2])

    cmap = plt.get_cmap("plasma")
    if n_splits > 0:
        colors = [cmap(0.05 + 0.85 * i / max(n_splits - 1, 1)) for i in range(n_splits)]
    else:
        colors = []

    # ----- IC-space split history -----------------------------------------
    ax_ic.add_patch(patches.Rectangle((-1.0, -1.0), 2.0, 2.0,
                                      fill=False, ec="black", lw=1.6,
                                      label="full IC box"))
    for idx, sid in enumerate(split_ids):
        sel = split_ic["split_idx"] == sid
        for role in ("parent", "left", "right"):
            r = sel & (split_ic["role"] == role)
            if not np.any(r):
                continue
            dy_lo = float(split_ic["dy_lo"][r][0])
            dy_hi = float(split_ic["dy_hi"][r][0])
            dvy_lo = float(split_ic["dvy_lo"][r][0])
            dvy_hi = float(split_ic["dvy_hi"][r][0])
            if role == "parent":
                ax_ic.add_patch(patches.Rectangle(
                    (dy_lo, dvy_lo), dy_hi - dy_lo, dvy_hi - dvy_lo,
                    facecolor=colors[idx], alpha=0.20,
                    edgecolor=colors[idx], linewidth=0.8))
            else:
                ax_ic.add_patch(patches.Rectangle(
                    (dy_lo, dvy_lo), dy_hi - dy_lo, dvy_hi - dvy_lo,
                    fill=False, edgecolor=colors[idx], linewidth=1.4))
    ax_ic.set_xlim(-1.05, 1.05)
    ax_ic.set_ylim(-1.05, 1.05)
    ax_ic.set_aspect("equal", adjustable="box")
    ax_ic.set_xlabel(r"$\delta_{y_0}$")
    ax_ic.set_ylabel(r"$\delta_{v_{y_0}}$")
    ax_ic.set_title(f"IC partition over {n_splits} split events\n"
                    "(filled = parent, outlined = children)")
    ax_ic.grid(True, alpha=0.25)

    # ----- Phase-space view -----------------------------------------------
    flow_poly = _polygon(flow_img)
    all_x = [flow_poly[:, 0]]
    all_y = [flow_poly[:, 1]]
    for sid in split_ids:
        sel = split_ph["split_idx"] == sid
        for role in ("parent", "left", "right"):
            r = sel & (split_ph["role"] == role)
            if np.any(r):
                poly = _polygon(split_ph[r])
                all_x.append(poly[:, 0])
                all_y.append(poly[:, 1])
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    pad_x = 0.10 * (all_x.max() - all_x.min() + 1e-12)
    pad_y = 0.10 * (all_y.max() - all_y.min() + 1e-12)
    xlim = (all_x.min() - pad_x, all_x.max() + pad_x)
    ylim = (all_y.min() - pad_y, all_y.max() + pad_y)

    # Reference orbit + primary only if visible in the zoomed window.
    ax_ph.plot(ref["x"], ref["y"], "k-", lw=1.0, alpha=0.4, label="reference orbit")
    if xlim[0] <= 0.0 <= xlim[1] and ylim[0] <= 0.0 <= ylim[1]:
        ax_ph.plot([0.0], [0.0], "*", color="orange", ms=14, zorder=5, label="primary")

    ax_ph.fill(flow_poly[:, 0], flow_poly[:, 1],
               facecolor="lightgrey", edgecolor="dimgrey",
               alpha=0.45, linewidth=0.9, zorder=1,
               label="single flow polygon (no split)")

    for idx, sid in enumerate(split_ids):
        sel = split_ph["split_idx"] == sid
        for role in ("parent", "left", "right"):
            r = sel & (split_ph["role"] == role)
            if not np.any(r):
                continue
            poly = _polygon(split_ph[r])
            if role == "parent":
                ax_ph.plot(poly[:, 0], poly[:, 1], color=colors[idx],
                           linewidth=1.0, alpha=0.8, zorder=2)
            else:
                ax_ph.fill(poly[:, 0], poly[:, 1],
                           facecolor=colors[idx], alpha=0.30,
                           edgecolor=colors[idx], linewidth=0.6, zorder=3)

    ax_ph.set_xlim(xlim)
    ax_ph.set_ylim(ylim)
    ax_ph.set_aspect("equal", adjustable="box")
    ax_ph.set_xlabel("x(t = T_orbit)")
    ax_ph.set_ylabel("y(t = T_orbit)")
    ax_ph.set_title("Phase-space at one orbital period\n"
                    "(thin lines = parent images, filled = children)")
    ax_ph.grid(True, alpha=0.25)
    ax_ph.legend(loc="upper right", fontsize=8, framealpha=0.95)

    # Colour bar by split index.
    if n_splits > 1:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=0.0, vmax=float(n_splits - 1))
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax_ph, fraction=0.04, pad=0.02)
        cb.set_label("split index", fontsize=9)

    # ----- Converged ADS leaves -------------------------------------------
    leaf_ids = sorted(set(final["leaf_idx"].astype(int).tolist()))

    leaf_x = [flow_poly[:, 0]]
    leaf_y = [flow_poly[:, 1]]
    for li in leaf_ids:
        sel = final["leaf_idx"] == li
        poly = _polygon(final[sel])
        leaf_x.append(poly[:, 0])
        leaf_y.append(poly[:, 1])
    leaf_x = np.concatenate(leaf_x)
    leaf_y = np.concatenate(leaf_y)
    pad_x = 0.10 * (leaf_x.max() - leaf_x.min() + 1e-12)
    pad_y = 0.10 * (leaf_y.max() - leaf_y.min() + 1e-12)
    fxlim = (leaf_x.min() - pad_x, leaf_x.max() + pad_x)
    fylim = (leaf_y.min() - pad_y, leaf_y.max() + pad_y)

    ax_fin.plot(ref["x"], ref["y"], "k-", lw=1.0, alpha=0.4, label="reference orbit")
    if fxlim[0] <= 0.0 <= fxlim[1] and fylim[0] <= 0.0 <= fylim[1]:
        ax_fin.plot([0.0], [0.0], "*", color="orange", ms=14, zorder=5, label="primary")
    ax_fin.fill(flow_poly[:, 0], flow_poly[:, 1],
                facecolor="lightgrey", edgecolor="dimgrey",
                alpha=0.45, linewidth=0.9, zorder=1,
                label="single flow polygon")

    leaf_cmap = plt.get_cmap("tab20")
    for k, li in enumerate(leaf_ids):
        sel = final["leaf_idx"] == li
        poly = _polygon(final[sel])
        ax_fin.fill(poly[:, 0], poly[:, 1],
                    facecolor=leaf_cmap(k % 20), alpha=0.65,
                    edgecolor="black", linewidth=0.6, zorder=3)

    ax_fin.set_xlim(fxlim)
    ax_fin.set_ylim(fylim)
    ax_fin.set_aspect("equal", adjustable="box")
    ax_fin.set_xlabel("x(t = T_orbit)")
    ax_fin.set_ylabel("y(t = T_orbit)")
    ax_fin.set_title(f"Converged ADS partition\n({len(leaf_ids)} leaves at T_orbit)")
    ax_fin.grid(True, alpha=0.25)
    ax_fin.legend(loc="upper right", fontsize=8, framealpha=0.95)

    fig.suptitle(
        "Snapshots driven by ADS split events — Kepler orbit (a=1, e=0.5, full period)",
        fontsize=12,
    )

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
