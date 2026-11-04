"""Render the audited grid as one descriptive heatmap (no inferential claims).

Requires matplotlib; pass --qa-scripts to the directory containing
audit_panel_alignment.py. All data come from the saved analysis JSON.
Contract: compare mean final validation accuracy across all 20 cells; single
quantitative grid, 180 x 120 mm, n=5 fixed-seed repeats per cell. Full sample SD
and raw runs are supplied in the companion report/JSON; this color map encodes
means only. Highlight the two observed 5/5 optimum cells, without declaring
statistical superiority. No observations are excluded.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-scripts", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    data = json.loads((ROOT / "docs/temperature_top_p_grid_analysis_20261103.json").read_text())
    cells = data["cells"]
    ts = sorted({c["temperature"] for c in cells})
    ps = sorted({c["top_p"] for c in cells})
    vals = np.array([[next(c["valid_pct"]["mean"] for c in cells
                     if c["temperature"] == t and c["top_p"] == p) for p in ps] for t in ts])
    assert vals.shape == (4, 5) and sum(c["n"] for c in cells) == 100
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(180/25.4, 120/25.4))
    fig.subplots_adjust(left=.13, right=.78, bottom=.24, top=.83)
    im = ax.imshow(vals, cmap="Blues", vmin=91.50, vmax=91.61, aspect="auto")
    ax.set_xticks(range(len(ps)), [str(p) for p in ps])
    ax.set_yticks(range(len(ts)), [str(t) for t in ts])
    ax.set_xlabel("top_p", labelpad=8)
    ax.set_ylabel("Temperature", labelpad=8)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0, pad=7)
    for c in cells:
        if c["valid_hits"] == 5:
            ax.text(ps.index(c["top_p"]), ts.index(c["temperature"]), "5/5 hits",
                    ha="center", va="center", color="white", fontsize=10, weight="bold")
    cax = fig.add_axes([.83, .24, .025, .59])
    cb = fig.colorbar(im, cax=cax, ticks=[91.50,91.55,91.60])
    cb.set_label("Mean best validation accuracy (%)", labelpad=10)
    cb.outline.set_visible(False)
    fig.text(.13, .92, "Temperature × top_p: final search quality", fontsize=13, weight="bold")
    fig.text(.13, .08, "CIFAR-10 · 100 true queries · 30 generations · seed 42", fontsize=9)
    fig.text(.13, .035, "5 repeats/cell; means only. Sample SD and all runs are in the companion report.", fontsize=8)
    out = ROOT / "docs/temperature_top_p_grid_20261103"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(out)+".alignment.json",
                                       exclude_axes=[cax], strict=True)
    fig.savefig(str(out)+".pdf")
    fig.savefig(str(out)+".svg")
    fig.savefig(str(out)+".png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
