"""Build thesis Figures 4.1 and 4.2 from training metrics.jsonl files.

Figure 4.1: skipped (zero-variance) GRPO steps per run.
Figure 4.2: mean training-rollout citation precision per 32-step block, R0 s8 vs. R1 s8.

Writes PDF + PNG and a JSON of the plotted values to docs/thesis/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "thesis" / "figures"
BLOCK = 32

RUNS = {
    "R0 s7": ROOT / "results/2026-10-10-batch/r0_outcome_seed7/metrics.jsonl",
    "R0 s8": ROOT / "results/2026-10-10-batch/r0_outcome_seed8/metrics.jsonl",
    "R1 s7": ROOT / "results/2026-10-10-grpo-run2/metrics.jsonl",
    "R1 s8": ROOT / "results/2026-10-10-batch/r1_full_seed8/metrics.jsonl",
    "R2 s7": ROOT / "results/2026-10-10-batch/r2_gated_seed7/metrics.jsonl",
    "R2 s8": ROOT / "results/2026-10-10-batch/r2_gated_seed8/metrics.jsonl",
}

# Okabe-Ito (colorblind-safe); hatching gives a second, non-color cue.
STYLE = {
    "R0": {"color": "#D55E00", "hatch": "//", "marker": "s", "linestyle": "--"},
    "R1": {"color": "#0072B2", "hatch": "", "marker": "o", "linestyle": "-"},
    "R2": {"color": "#009E73", "hatch": "..", "marker": "^", "linestyle": ":"},
}


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def skipped_steps(rows: list[dict]) -> tuple[int, int]:
    return sum(row["skipped_reason"] is not None for row in rows), len(rows)


def precision_by_block(rows: list[dict]) -> list[dict]:
    blocks: dict[int, list[float]] = {}
    for row in rows:
        block = (row["step"] - 1) // BLOCK
        values = [
            r["citation_precision"]
            for r in row.get("rollouts", [])
            if r.get("citation_precision") is not None
        ]
        blocks.setdefault(block, []).extend(values)
    return [
        {
            "steps": f"{b * BLOCK + 1}-{(b + 1) * BLOCK}",
            "mean_precision": sum(v) / len(v) if v else None,
            "n_valid_rollouts": len(v),
        }
        for b, v in sorted(blocks.items())
    ]


def system_diagram() -> None:
    """Figure 3.1: one training episode and the GRPO update."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.0, 3.6), layout="constrained")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
                                    facecolor=color, edgecolor="#333333", linewidth=1))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontweight="bold", fontsize=8.5)
        ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=7, linespacing=1.35)

    def arrow(start, end, label="", offset=(0, 1.4), style="-|>", rad=0.0):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=11,
                                     color="#333333", linewidth=1.1,
                                     connectionstyle=f"arc3,rad={rad}"))
        if label:
            mx, my = (start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1]
            ax.text(mx, my, label, ha="center", va="bottom", fontsize=6.8, family="monospace")

    box(1, 30, 17, 18, "Question", "distractor setting\n10 paragraphs\n(2 gold + 8 distractors)", "#F2F2F2")
    box(27, 30, 21, 18, "Agent (policy)", "Qwen2.5-7B-Instruct\n+ LoRA (r=16)\none action per turn", "#DCEBF7")
    box(58, 30, 21, 18, "SearchEnvironment", "BM25 over the 10\nparagraphs, top-3 docs\nmax 3 searches", "#FBE6D9")
    box(58, 3, 21, 18, "VEGR reward", "EM + F1 + coverage\n+ precision + recall\n− 0.1·cost − 0.1·dup\ninvalid → −1", "#DDF1E8")
    box(27, 3, 21, 18, "GRPO update", "G = 8 rollouts\ngroup-normalized advantage\nclipped objective, ε = 0.2\nzero-variance groups skipped", "#EDE7F6")
    box(83, 30, 16, 18, "Gold labels", "answer\nsupporting facts\n(title, sentence)", "#F2F2F2")

    arrow((18.6, 39), (26.4, 39))
    arrow((48.6, 44), (57.4, 44), "SEARCH[q]", offset=(0, 0.5))
    arrow((48.6, 38), (57.4, 38), "ANSWER[a]", offset=(0, 0.5))
    arrow((57.4, 32), (48.6, 32), "results", offset=(0, 0.5))
    arrow((68.5, 29.4), (68.5, 21.6), "finished\nepisode", offset=(-6.5, -2.5))
    arrow((91, 29.4), (79.6, 15), rad=-0.2)
    arrow((57.4, 12), (48.6, 12), "reward ×8")
    arrow((37.5, 21.6), (37.5, 29.4), "update LoRA", offset=(-7.5, -1.5))
    fig.savefig(OUT / "fig3_1_system.pdf")
    fig.savefig(OUT / "fig3_1_system.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    system_diagram()
    data = {name: load(path) for name, path in RUNS.items()}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # Figure 4.1
    skipped = {name: skipped_steps(rows) for name, rows in data.items()}
    fig, ax = plt.subplots(figsize=(5.5, 3.0), layout="constrained")
    names = list(skipped)
    for i, name in enumerate(names):
        count, total = skipped[name]
        style = STYLE[name[:2]]
        ax.bar(i, count, color=style["color"], hatch=style["hatch"], edgecolor="white", width=0.7)
        ax.text(i, count + 3, f"{count}", ha="center", va="bottom")
    ax.set_xticks(range(len(names)), names)
    ax.set_ylim(0, 256)
    ax.set_ylabel(f"Skipped steps (of {skipped[names[0]][1]})")
    ax.set_title("Zero-variance GRPO steps skipped during training", loc="left")
    fig.savefig(OUT / "fig4_1_skipped_steps.pdf")
    fig.savefig(OUT / "fig4_1_skipped_steps.png", dpi=300)
    plt.close(fig)

    # Figure 4.2
    curves = {name: precision_by_block(data[name]) for name in ("R0 s8", "R1 s8")}
    fig, ax = plt.subplots(figsize=(5.5, 3.0), layout="constrained")
    for name, blocks in curves.items():
        style = STYLE[name[:2]]
        xs = [i for i, b in enumerate(blocks) if b["mean_precision"] is not None]
        ys = [blocks[i]["mean_precision"] for i in xs]
        ax.plot(xs, ys, color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], linewidth=2, markersize=6)
        ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=style["color"])
    labels = [b["steps"] for b in curves["R0 s8"]]
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_xlim(-0.3, len(labels) + 0.2)
    ax.set_ylim(-0.02, 1)
    ax.set_xlabel("Training steps (32-step blocks)")
    ax.set_ylabel("Mean citation precision\n(valid training rollouts)")
    ax.set_title("Citation precision during training, seed 8", loc="left")
    fig.savefig(OUT / "fig4_2_citation_precision.pdf")
    fig.savefig(OUT / "fig4_2_citation_precision.png", dpi=300)
    plt.close(fig)

    provenance = {
        "sources": {name: str(path.relative_to(ROOT)) for name, path in RUNS.items()},
        "fig4_1_skipped_steps": {name: {"skipped": c, "total": t} for name, (c, t) in skipped.items()},
        "fig4_2_citation_precision": curves,
    }
    (OUT / "figures_data.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance["fig4_1_skipped_steps"]))
    for name, blocks in curves.items():
        print(name, [(b["steps"], round(b["mean_precision"], 3) if b["mean_precision"] is not None else None, b["n_valid_rollouts"]) for b in blocks])


if __name__ == "__main__":
    main()
