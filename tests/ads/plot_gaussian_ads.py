#!/usr/bin/env python3
"""
Visualise the ADS result for f(x) = exp(-x^2) on [-3, 3].

Usage (from the build directory after running the testAdsGaussian test):
    python3 <repo>/tests/ads/plot_gaussian_ads.py [gaussian_ads.csv]

The CSV has columns:
    center, halfWidth, c0, c1, ..., c10

Each row describes a subdomain [center-hw, center+hw] and the coefficients
of the degree-10 Taylor polynomial in the normalised variable
    delta = (x - center) / halfWidth,  delta in [-1, 1].

So  f(x) ≈ sum_{k=0}^{10} c_k * delta^k  for x in [center-hw, center+hw].
"""

import sys
import csv
import math
import pathlib

# ---------------------------------------------------------------------------
# Try to import matplotlib; fall back to a plain-text summary if unavailable.
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (works everywhere)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not found — printing a text summary instead.")


# ---------------------------------------------------------------------------
# Horner evaluation of c[0] + c[1]*x + ... + c[N]*x^N
# ---------------------------------------------------------------------------
def horner(coeffs, x):
    result = 0.0
    for c in reversed(coeffs):
        result = result * x + c
    return result


# ---------------------------------------------------------------------------
# Read CSV
# ---------------------------------------------------------------------------
def read_csv(path):
    subdomains = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            center     = float(row["center"])
            half_width = float(row["halfWidth"])
            # coefficient columns: c0, c1, ..., cN
            coeffs = []
            i = 0
            while f"c{i}" in row:
                coeffs.append(float(row[f"c{i}"]))
                i += 1
            subdomains.append((center, half_width, coeffs))
    return subdomains


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    csv_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "gaussian_ads.csv")
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print("Run the testAdsGaussian test first to generate it.")
        sys.exit(1)

    subdomains = read_csv(csv_path)
    print(f"Loaded {len(subdomains)} subdomains from {csv_path}")

    # ------------------------------------------------------------------
    # Text summary: max error per subdomain
    # ------------------------------------------------------------------
    print(f"\n{'Subdomain':>22}   {'N coeffs':>8}   {'max |err|':>12}")
    print("-" * 50)
    total_max_err = 0.0
    for center, hw, coeffs in sorted(subdomains, key=lambda s: s[0] - s[1]):
        lo, hi = center - hw, center + hw
        # 200-point grid over this subdomain
        xs = [lo + (hi - lo) * i / 199 for i in range(200)]
        max_err = 0.0
        for x in xs:
            delta  = (x - center) / hw
            approx = horner(coeffs, delta)
            exact  = math.exp(-x * x)
            max_err = max(max_err, abs(approx - exact))
        total_max_err = max(total_max_err, max_err)
        print(f"  [{lo:+.4f}, {hi:+.4f}]   {len(coeffs):>8}   {max_err:>12.3e}")
    print("-" * 50)
    print(f"  Global max error: {total_max_err:.3e}\n")

    if not HAS_MPL:
        return

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                              gridspec_kw={"height_ratios": [2, 1]})

    # ---- Top panel: f(x) and the ADS piecewise polynomial ----
    ax = axes[0]
    xs_fine = [i * 6.0 / 999 - 3.0 for i in range(1000)]
    ys_exact = [math.exp(-x * x) for x in xs_fine]
    ax.plot(xs_fine, ys_exact, "k-", linewidth=2, label=r"$f(x)=e^{-x^2}$ (exact)", zorder=5)

    cmap = plt.get_cmap("tab20")
    for k, (center, hw, coeffs) in enumerate(subdomains):
        lo, hi = center - hw, center + hw
        xs_sub = [lo + (hi - lo) * i / 49 for i in range(50)]
        ys_sub = [horner(coeffs, (x - center) / hw) for x in xs_sub]
        color  = cmap(k % 20)
        ax.plot(xs_sub, ys_sub, "-", color=color, linewidth=1.5, alpha=0.8)
        ax.axvline(lo, color="grey", linewidth=0.4, linestyle="--", alpha=0.5)

    # Right boundary of last subdomain
    ax.axvline(max(c + h for c, h, _ in subdomains),
               color="grey", linewidth=0.4, linestyle="--", alpha=0.5)

    ax.set_title(
        rf"ADS of $f(x)=e^{{-x^2}}$ on $[-3,3]$, "
        rf"order $N=10$, tolerance $\varepsilon=10^{{-5}}$ "
        rf"→ {len(subdomains)} subdomains",
        fontsize=11,
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.legend()
    ax.set_xlim(-3, 3)

    # ---- Bottom panel: pointwise absolute error ----
    ax2 = axes[1]
    xs_err, ys_err = [], []
    for center, hw, coeffs in subdomains:
        lo, hi = center - hw, center + hw
        for i in range(50):
            x = lo + (hi - lo) * i / 49
            delta  = (x - center) / hw
            approx = horner(coeffs, delta)
            exact  = math.exp(-x * x)
            xs_err.append(x)
            ys_err.append(abs(approx - exact))

    # Sort by x for a clean line
    pairs = sorted(zip(xs_err, ys_err))
    xs_err = [p[0] for p in pairs]
    ys_err = [p[1] for p in pairs]

    ax2.semilogy(xs_err, ys_err, "b.", markersize=2, alpha=0.6)
    ax2.axhline(1e-5, color="red", linewidth=1, linestyle="--", label="tolerance $10^{-5}$")
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$|f - p|$")
    ax2.set_title("Pointwise absolute error")
    ax2.legend()
    ax2.set_xlim(-3, 3)

    plt.tight_layout()
    out_png = csv_path.with_suffix(".png")
    fig.savefig(out_png, dpi=150)
    print(f"Plot saved to {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
