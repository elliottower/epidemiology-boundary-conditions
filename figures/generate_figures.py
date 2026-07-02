"""Generate figures for the clinical geometry paper (paper_v3a)."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTDIR = Path(__file__).parent

# Match psych paper style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# ── Fig 1: Bimodal Q dot plot ──────────────────────────────────────────────

def fig1_q_dotplot():
    ms_data = [
        ("HLA-DRB1", 23.1, "non-transport", True),
        ("Latitude/vitD", 23.7, "non-transport", True),
        ("Sex ratio", 39.8, "non-transport", True),
        ("EBV→MS", 0.67, "transport", True),
        ("Smoking→MS", 0.60, "transport", True),
        ("VitD→MS", 0.63, "transport", True),
        ("BMI→MS", 0.31, "transport", True),
    ]
    ad_data = [
        ("APOE4", 33.0, "non-transport", True),
        ("Sex×tau", 15.7, "non-transport", True),
        ("BMI→AD", 14.1, "non-transport", True),
        ("Ancestry", 6.6, "non-transport", False),
        ("Age×lecanemab", 4.7, "non-transport", False),
        ("T2D→AD", 3.1, "non-transport", False),
        ("CRP→AD", 0.66, "transport", True),
        ("TREM2", 0.70, "transport", True),
        ("Alcohol→AD", 0.43, "transport", True),
        ("Education→AD", 0.28, "transport", True),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ms_color = "#2E86AB"
    ad_color = "#A23B72"
    miss_color = "#999999"

    y_positions = []
    labels = []
    colors = []
    markers = []
    q_vals = []

    all_data = [("MS", ms_data), ("AD", ad_data)]
    y = 0
    for domain, pairs in all_data:
        for name, q, expected, correct in reversed(pairs):
            y_positions.append(y)
            labels.append(name)
            q_vals.append(q)
            if not correct:
                colors.append(miss_color)
                markers.append("x")
            else:
                colors.append(ms_color if domain == "MS" else ad_color)
                markers.append("o")
            y += 1
        y += 1  # gap between domains

    for i, (yp, qv, c, m) in enumerate(zip(y_positions, q_vals, colors, markers)):
        ms = 90 if m == "o" else 80
        lw = 1.5 if m == "o" else 2.5
        ax.scatter(qv, yp, c=c, s=ms, marker=m, linewidths=lw,
                   edgecolors=c if m == "x" else "white", zorder=5)

    ax.set_xscale("log")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)

    # threshold line
    ax.axvline(x=7.815, color="#E8E8E8", linewidth=12, zorder=0)
    ax.axvline(x=7.815, color="#666666", linewidth=0.8, linestyle="--", zorder=1,
               label=r"$\chi^2_{0.05,3}=7.81$")

    # gap annotation
    ax.annotate("", xy=(23.1, 4.5), xytext=(0.67, 4.5),
                arrowprops=dict(arrowstyle="<->", color=ms_color, lw=1.5))
    ax.text(3.5, 4.8, r"34.6$\times$ gap", ha="center", fontsize=9, color=ms_color,
            fontstyle="italic")

    ax.annotate("", xy=(14.1, 12.5), xytext=(0.70, 12.5),
                arrowprops=dict(arrowstyle="<->", color=ad_color, lw=1.5))
    ax.text(2.8, 12.8, r"20$\times$ gap", ha="center", fontsize=9, color=ad_color,
            fontstyle="italic")

    # domain labels
    ax.text(0.08, 3.0, "MS", fontsize=13, fontweight="bold", color=ms_color,
            transform=ax.get_yaxis_transform(), ha="right")
    ax.text(0.08, 11.0, "AD", fontsize=13, fontweight="bold", color=ad_color,
            transform=ax.get_yaxis_transform(), ha="right")

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ms_color,
                   markersize=8, label="MS (correct)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ad_color,
                   markersize=8, label="AD (correct)"),
        plt.Line2D([0], [0], marker="x", color=miss_color, markersize=8,
                   linestyle="None", markeredgewidth=2, label="Underpowered miss"),
        plt.Line2D([0], [0], color="#666666", linestyle="--", linewidth=0.8,
                   label=r"$\alpha=0.05$ threshold"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              framealpha=0.9, edgecolor="#cccccc")

    ax.set_xlabel(r"Cochran $Q$ statistic (log scale)", fontsize=12)
    ax.set_title(r"$H^1$ classification: transport vs. non-transport MR pairs",
                 fontsize=13, pad=12)
    ax.set_xlim(0.15, 80)

    fig.savefig(OUTDIR / "fig1_q_dotplot.png")
    fig.savefig(OUTDIR / "fig1_q_dotplot.pdf")
    plt.close(fig)
    print(f"Fig 1: {OUTDIR / 'fig1_q_dotplot.png'}")


# ── Fig 2: AD DAG with edge Q values ──────────────────────────────────────

def fig2_dag():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.2, 5.2)
    ax.set_aspect("equal")
    ax.axis("off")

    nodes = {
        "Amyloid": (3.5, 4.5),
        "Tau": (0.8, 0.8),
        "Cognition": (6.2, 0.8),
    }
    node_r = 0.6

    edge_data = [
        ("Amyloid", "Tau", 23.8, "Mechanism switching", "#C0392B"),
        ("Tau", "Cognition", 30.3, "Dose-response", "#D35400"),
        ("Amyloid", "Cognition", 1.5, "Stable bypass", "#27AE60"),
    ]

    # Label positions (manually placed to avoid overlap)
    label_pos = {
        ("Amyloid", "Tau"): (1.2, 3.1),
        ("Tau", "Cognition"): (3.5, 0.35),
        ("Amyloid", "Cognition"): (5.8, 3.1),
    }

    for src, dst, q, label, color in edge_data:
        x0, y0 = nodes[src]
        x1, y1 = nodes[dst]
        lw = 2.5 if q < 5 else 3.5 + np.log10(q) * 1.5
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    shrinkA=node_r * 55, shrinkB=node_r * 55,
                                    connectionstyle="arc3,rad=0.0"))

        lx, ly = label_pos[(src, dst)]
        ax.text(lx, ly, f"{label}\n$Q = {q}$", ha="center", va="center",
                fontsize=10, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, alpha=0.95, linewidth=1.2))

    for name, (x, y) in nodes.items():
        circle = plt.Circle((x, y), node_r, facecolor="white", edgecolor="#2C3E50",
                             linewidth=2.5, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, name, ha="center", va="center", fontsize=12,
                fontweight="bold", color="#2C3E50", zorder=11)

    legend_elements = [
        mpatches.Patch(facecolor="#C0392B", alpha=0.8, label="Heterogeneous (switching)"),
        mpatches.Patch(facecolor="#D35400", alpha=0.8, label="Heterogeneous (dose-response)"),
        mpatches.Patch(facecolor="#27AE60", alpha=0.8, label="Homogeneous (stable)"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", fontsize=9,
              ncol=3, framealpha=0.9, edgecolor="#cccccc",
              bbox_to_anchor=(0.5, -0.05))

    ax.set_title("Per-edge sheaf $Q$ test on ADNI longitudinal data",
                 fontsize=13, pad=10)

    fig.savefig(OUTDIR / "fig2_dag_edges.png")
    fig.savefig(OUTDIR / "fig2_dag_edges.pdf")
    plt.close(fig)
    print(f"Fig 2: {OUTDIR / 'fig2_dag_edges.png'}")


# ── Fig 3: Sensitivity analysis ───────────────────────────────────────────

def fig3_sensitivity():
    alphas = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    # At 0.01 and 0.05: 14/17 = 0.824, specificity 1.0
    # At 0.10 and above: 16/17 = 0.941
    accuracy = [0.824, 0.824, 0.824, 0.824, 0.824, 0.941, 0.941, 0.941]
    sensitivity = [0.667, 0.667, 0.667, 0.667, 0.667, 0.889, 0.889, 0.889]
    specificity = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(alphas, accuracy, "o-", color="#2E86AB", linewidth=2, markersize=7,
            label="Accuracy", zorder=5)
    ax.plot(alphas, sensitivity, "s--", color="#A23B72", linewidth=2, markersize=6,
            label="Sensitivity", zorder=5)
    ax.plot(alphas, specificity, "D-", color="#27AE60", linewidth=2, markersize=6,
            label="Specificity", zorder=5)

    ax.axvline(x=0.10, color="#E8E8E8", linewidth=10, zorder=0)

    ax.set_xlabel(r"Significance threshold $\alpha$", fontsize=12)
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title("Sensitivity analysis across classification thresholds", fontsize=13, pad=12)
    ax.set_ylim(0.55, 1.05)
    ax.set_xlim(-0.005, 0.215)

    ax.legend(fontsize=10, framealpha=0.9, edgecolor="#cccccc", loc="center right")
    ax.axhline(y=1.0, color="#cccccc", linewidth=0.5, linestyle=":", zorder=0)

    fig.savefig(OUTDIR / "fig3_sensitivity.png")
    fig.savefig(OUTDIR / "fig3_sensitivity.pdf")
    plt.close(fig)
    print(f"Fig 3: {OUTDIR / 'fig3_sensitivity.png'}")


# ── Fig 4: Boundary conditions decision diagram ──────────────────────────

def fig4_boundary():
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.7, 4.0)
    ax.axis("off")

    def draw_box(x, y, w, h, text, color, fontsize=9, bold=False):
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.12", facecolor=color,
            edgecolor="#2C3E50", linewidth=1.5)
        ax.add_patch(rect)
        fw = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight=fw, color="#2C3E50")

    def arrow(x0, y0, x1, y1, label=""):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=1.3))
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx - 0.22, my, label, fontsize=8, color="#666666",
                    ha="center", va="center", fontstyle="italic")

    # Row 1 (y=3.55): conditions
    draw_box(2.0, 3.55, 2.6, 0.55, "Subspace-valued\ndata?", "#FFF9E6", fontsize=9)
    draw_box(5.0, 3.55, 2.6, 0.55, "Cyclic\nconstraints?", "#FFF9E6", fontsize=9)
    draw_box(8.0, 3.55, 2.6, 0.55, "Edge-specific\nheterogeneity?", "#FFF9E6", fontsize=9)

    # Row 2 (y=2.5): geometric methods
    draw_box(2.0, 2.5, 2.5, 0.55,
             "Grassmannian holonomy\n(Berry phase)", "#D5E8D4", fontsize=9, bold=True)
    draw_box(5.0, 2.5, 2.5, 0.55,
             "Sheaf cohomology\n($H^1$ obstruction)", "#DAE8FC", fontsize=9, bold=True)
    draw_box(8.0, 2.5, 2.5, 0.55,
             "Per-edge\nsheaf $Q$ test", "#E1D5E7", fontsize=9, bold=True)

    # Row 3 (y=1.45): standard fallbacks
    draw_box(2.0, 1.45, 2.5, 0.55,
             "Scalar projection\n(partial correlation)", "#F8CECC", fontsize=9)
    draw_box(5.0, 1.45, 2.5, 0.55,
             "Cochran's $Q$\n(pairwise test)", "#F8CECC", fontsize=9)
    draw_box(8.0, 1.45, 2.5, 0.55,
             "Global $Q$ test\n($K$-means)", "#F8CECC", fontsize=9)

    # Arrows: condition → geometric (Yes)
    arrow(2.0, 3.275, 2.0, 2.78, "Yes")
    arrow(5.0, 3.275, 5.0, 2.78, "Yes")
    arrow(8.0, 3.275, 8.0, 2.78, "Yes")

    # Arrows: geometric → standard (No)
    arrow(2.0, 2.225, 2.0, 1.73, "No")
    arrow(5.0, 2.225, 5.0, 1.73, "No")
    arrow(8.0, 2.225, 8.0, 1.73, "No")

    ax.set_title("Boundary conditions: when geometric methods help",
                 fontsize=13, pad=8, fontweight="bold")

    geo_patch = mpatches.Patch(facecolor="#D5E8D4", edgecolor="#2C3E50",
                               label="Geometric method (advantage)")
    std_patch = mpatches.Patch(facecolor="#F8CECC", edgecolor="#2C3E50",
                               label="Standard method (sufficient)")
    cond_patch = mpatches.Patch(facecolor="#FFF9E6", edgecolor="#2C3E50",
                                label="Testable condition")
    ax.legend(handles=[cond_patch, geo_patch, std_patch], loc="lower center",
              ncol=3, fontsize=9, framealpha=0.9, edgecolor="#cccccc",
              bbox_to_anchor=(0.5, -0.02))

    fig.savefig(OUTDIR / "fig4_boundary_conditions.png")
    fig.savefig(OUTDIR / "fig4_boundary_conditions.pdf")
    plt.close(fig)
    print(f"Fig 4: {OUTDIR / 'fig4_boundary_conditions.png'}")


if __name__ == "__main__":
    fig1_q_dotplot()
    fig2_dag()
    fig3_sensitivity()
    fig4_boundary()
    print("done")
