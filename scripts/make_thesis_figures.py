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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
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
