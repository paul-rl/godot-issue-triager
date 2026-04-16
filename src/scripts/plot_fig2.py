"""
plot_fig2.py: Figure 2: Coverage-Accuracy Tradeoff for Component Routing.

"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE   = "#2166ac"
ORANGE = "#d6604d"


def load_baseline(path: str) -> tuple[list, list]:
    """
    Load baseline curve. Restricts to coverage >= 0.25 (below this,
    threshold is so high that F1 estimates are noisy on very few issues).
    Returns smoothed (coverages, micro_f1_covered).
    """
    with open(path) as f:
        raw = json.load(f)

    pts = [
        (pt["coverage"], pt["micro_f1_covered"])
        for pt in raw
        if 0.25 <= pt["coverage"] <= 0.97
    ]
    pts.sort(key=lambda x: x[0])

    covs = np.array([p[0] for p in pts])
    f1s  = np.array([p[1] for p in pts])

    # Light smoothing to remove jaggedness from the dense 301-point grid
    # Use a uniform window of 7 points (~2% of coverage range)
    kernel = np.ones(7) / 7
    f1s_smooth = np.convolve(f1s, kernel, mode="same")
    # Fix edge effects: keep raw values at the edges
    f1s_smooth[:3]  = f1s[:3]
    f1s_smooth[-3:] = f1s[-3:]

    return covs.tolist(), f1s_smooth.tolist()


def load_llm(path: str) -> tuple[list, list]:
    """Deduplicated, sorted ascending LLM curve."""
    with open(path) as f:
        raw = json.load(f)
    seen = set()
    pts = []
    for c, a in zip(raw["coverages"], raw["accuracies"]):
        key = round(c, 4)
        if key not in seen:
            seen.add(key)
            pts.append((c, a))
    pts.sort(key=lambda x: x[0])
    return [p[0] for p in pts], [p[1] for p in pts]


def find_op_point(path: str) -> tuple[float, float]:
    """(coverage, micro_f1_covered) at the tuned threshold ~0.515."""
    with open(path) as f:
        raw = json.load(f)
    op = min(raw, key=lambda x: abs(x["threshold"] - 0.515))
    return op["coverage"], op["micro_f1_covered"]


def find_at_coverage(path: str, target: float) -> float:
    """micro_f1_covered at the point closest to target coverage."""
    with open(path) as f:
        raw = json.load(f)
    pt = min(raw, key=lambda x: abs(x["coverage"] - target))
    return pt["micro_f1_covered"]


def plot(baseline_path: str, llm_path: str, out_stem: str) -> None:
    b_cov, b_f1   = load_baseline(baseline_path)
    l_cov, l_f1   = load_llm(llm_path)
    op_cov, op_f1 = find_op_point(baseline_path)

    llm_start_cov = l_cov[0]   # ~0.197
    llm_start_f1  = l_f1[0]    # ~0.666
    base_at_20    = find_at_coverage(baseline_path, llm_start_cov)  # ~0.737

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    # ── Curves ────────────────────────────────────────────────────────────────
    ax.plot(b_cov, b_f1, color=BLUE,   linewidth=2.2, zorder=3,
            label="TF-IDF Baseline")
    ax.plot(l_cov, l_f1, color=ORANGE, linewidth=2.2, zorder=3,
            label=r"LLM (Gemini 2.5 Flash Lite, $N{=}20$, $\tau{=}0.8$)")

    # ── Reference line: baseline operating point F1 ────────────────────────
    ax.axhline(op_f1, color=BLUE, linewidth=0.9, linestyle="--",
               alpha=0.65, zorder=2)
    ax.text(0.99, op_f1 + 0.003,
            f"Baseline op. $\\mu$F1 = {op_f1:.3f}",
            fontsize=7.5, color=BLUE, alpha=0.85, ha="right", va="bottom")

    # ── Reference line: LLM high-confidence F1 ────────────────────────────
    ax.axhline(llm_start_f1, color=ORANGE, linewidth=0.9, linestyle="--",
               alpha=0.65, zorder=2)
    ax.text(0.99, llm_start_f1 - 0.006,
            f"LLM high-conf. $\\mu$F1 = {llm_start_f1:.3f}",
            fontsize=7.5, color=ORANGE, alpha=0.85, ha="right", va="top")

    # ── Baseline operating point marker ───────────────────────────────────
    ax.scatter([op_cov], [op_f1], color=BLUE, s=65, zorder=5)
    ax.annotate(
        f"Baseline op. point\ncov={op_cov:.2f}, $\\mu$F1={op_f1:.3f}",
        xy=(op_cov, op_f1),
        xytext=(op_cov - 0.20, op_f1 + 0.022),
        fontsize=7.5, color=BLUE, ha="center",
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1),
    )

    # ── Two markers at 20% coverage: baseline (diamond) and LLM (circle) ──
    ax.scatter([llm_start_cov], [base_at_20],
               color=BLUE, s=60, marker="D", zorder=5)
    ax.scatter([llm_start_cov], [llm_start_f1],
               color=ORANGE, s=60, zorder=5)


    ax.annotate(
        f"Baseline @ 20% cov.\n$\\mu$F1={base_at_20:.3f}",
        xy=(llm_start_cov, base_at_20),
        xytext=(llm_start_cov + 0.10, base_at_20 + 0.010),
        fontsize=7.5, color=BLUE, ha="left",
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1),
    )


    ax.annotate(
        f"LLM @ 20% cov.\n$\\mu$F1={llm_start_f1:.3f}",
        xy=(llm_start_cov, llm_start_f1),
        xytext=(llm_start_cov + 0.10, llm_start_f1 - 0.018),
        fontsize=7.5, color=ORANGE, ha="left",
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1),
    )

    # ── Gap label between the two 20% markers ──────────────────────────────
    gap = base_at_20 - llm_start_f1
    mid_y = (base_at_20 + llm_start_f1) / 2
    gap_x = llm_start_cov + 0.016
    # Vertical line connecting the two markers
    ax.plot([gap_x, gap_x], [llm_start_f1 + 0.002, base_at_20 - 0.002],
            color="gray", lw=0.9, zorder=4)
    # Small horizontal ticks at top and bottom
    tick = 0.005
    ax.plot([gap_x - tick, gap_x + tick], [base_at_20 - 0.002]*2,
            color="gray", lw=0.9, zorder=4)
    ax.plot([gap_x - tick, gap_x + tick], [llm_start_f1 + 0.002]*2,
            color="gray", lw=0.9, zorder=4)
    ax.text(gap_x + 0.008, mid_y,
            f"$\\Delta${gap:.3f}", fontsize=7.5, color="gray",
            ha="left", va="center")

    # ── Note about LLM curve start ─────────────────────────────────────────
    ax.text(0.50, 0.572,
            "LLM curve begins at 20% coverage "
            "(exact-set consensus, $\\tau{=}0.8$, $N{=}20$)",
            fontsize=7, color="gray", ha="center", style="italic",
            transform=ax.transData)

    # ── Axes ───────────────────────────────────────────────────────────────
    ax.set_xlabel(
        r"Coverage (fraction of issues with $\geq$1 predicted label)",
        fontsize=10,
    )
    ax.set_ylabel("Micro-F1 on covered samples", fontsize=10)
    ax.set_title("Coverage\u2013Accuracy Tradeoff: Component Routing",
                 fontsize=11, fontweight="bold")

    ax.set_xlim(0.18, 1.02)
    ax.set_ylim(0.565, 0.77)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
    ax.grid(True, linestyle=":", alpha=0.4, zorder=1)

    # Legend placed in upper right to avoid overlap with note at bottom
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    fig.tight_layout(pad=1.2)

    for ext in (".pdf", ".png"):
        path = out_stem + ext
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure 2: Coverage-Accuracy Tradeoff.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline",
                        default="final_results/coverage_curve_baseline.json")
    parser.add_argument("--llm",
                        default="final_results/coverage_curve_llm.json")
    parser.add_argument("--out",
                        default="final_results/fig2_coverage_accuracy",
                        help="Output path stem (.pdf and .png appended)")
    args = parser.parse_args()
    plot(args.baseline, args.llm, args.out)


if __name__ == "__main__":
    main()