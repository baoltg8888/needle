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


EPISODES = {
    "B0": ROOT / "results/2026-10-10-grpo-smoke/b0_episodes.jsonl",
    "R0 s7": ROOT / "results/2026-10-10-batch/r0_outcome_seed7/episodes.jsonl",
    "R0 s8": ROOT / "results/2026-10-10-batch/r0_outcome_seed8/episodes.jsonl",
    "R1 s7": ROOT / "results/2026-10-10-grpo-run2/grpo_run2_episodes.jsonl",
    "R1 s8": ROOT / "results/2026-10-10-batch/r1_full_seed8/episodes.jsonl",
    "R2 s7": ROOT / "results/2026-10-10-batch/r2_gated_seed7/episodes.jsonl",
    "R2 s8": ROOT / "results/2026-10-10-batch/r2_gated_seed8/episodes.jsonl",
}

# Seed-9 holdout results: raw files are still on the pod. Values are copied from
# docs/thesis/writer-brief.md section 4.1 (from the pod's batch2.log) until downloaded.
SEED9_FROM_BRIEF = {
    "R1 s9": {"em_all": 136 / 300, "citation_precision": 0.754},  # brief: 45.3%
    "R2 s9": {"em_all": 115 / 300, "citation_precision": 0.702},  # brief: 38.3%
}

FAILURE_KINDS = ("evidence", "protocol", "budget")


def episode_table(path: Path) -> dict[str, dict]:
    return {row["question_id"]: row for row in load(path)}


def em_all(row: dict) -> float:
    if row["status"] != "completed":
        return 0.0
    return float(row["metrics"]["answer_exact_match"])


def run_summary(rows: dict[str, dict]) -> dict:
    done = [r for r in rows.values() if r["status"] == "completed"]
    return {
        "em_all": sum(em_all(r) for r in rows.values()) / len(rows),
        "citation_precision": sum(r["metrics"]["citation_precision"] for r in done) / len(done),
        "completed": len(done),
        "failures": {k: sum(r.get("failure_kind") == k for r in rows.values()) for k in FAILURE_KINDS},
    }


def paired_bootstrap(a: dict[str, dict], b: dict[str, dict], resamples: int = 10_000,
                     seed: int = 0) -> dict:
    import numpy as np

    ids = sorted(set(a) & set(b))
    diff = np.array([em_all(a[q]) - em_all(b[q]) for q in ids])
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(ids), size=(resamples, len(ids)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"delta": float(diff.mean()) * 100, "low": float(low) * 100,
            "high": float(high) * 100, "n": len(ids), "resamples": resamples}


def method_of(name: str) -> str:
    return name[:2]


def ch4_figures() -> dict:
    episodes = {name: episode_table(path) for name, path in EPISODES.items()}
    summaries = {name: run_summary(rows) for name, rows in episodes.items()}
    points = {name: {"em_all": s["em_all"], "citation_precision": s["citation_precision"],
                     "source": "episodes.jsonl"} for name, s in summaries.items()}
    for name, values in SEED9_FROM_BRIEF.items():
        points[name] = {**values, "source": "writer-brief §4.1 (raw files on pod)"}
    style_b0 = {"color": "#666666", "marker": "D", "hatch": "xx", "linestyle": "-"}

    def style(name):
        return style_b0 if name == "B0" else STYLE[method_of(name)]

    # Figure 4.3: EM vs. citation precision.
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(5.5, 3.6), layout="constrained")
    for name, p in points.items():
        st = style(name)
        hollow = name in SEED9_FROM_BRIEF
        ax.scatter(p["em_all"] * 100, p["citation_precision"], s=46, marker=st["marker"],
                   facecolor="white" if hollow else st["color"], edgecolor=st["color"],
                   linewidth=1.5, zorder=3)
    for name, offset in {"B0": (8, -3), "R0 s8": (8, -3), "R0 s7": (8, -9)}.items():
        p = points[name]
        ax.annotate(name, (p["em_all"] * 100, p["citation_precision"]), xytext=offset,
                    textcoords="offset points", fontsize=7.5)
    ax.annotate("R1 and R2, all seeds", (43.5, 0.78), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=7.5)
    handles = [Line2D([], [], linestyle="", marker=s["marker"], color=s["color"], markersize=6, label=l)
               for s, l in [(style_b0, "B0 (un-tuned)"), (STYLE["R0"], "R0 outcome-only"),
                            (STYLE["R1"], "R1 VEGR"), (STYLE["R2"], "R2 gated")]]
    handles.append(Line2D([], [], linestyle="", marker="o", markerfacecolor="white",
                          markeredgecolor="#333333", markersize=6, label="hollow = seed 9"))
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="center left")
    ax.set_xlim(15, 55)
    ax.set_ylim(-0.04, 1.0)
    ax.set_xlabel("EM on holdout (%), failures count as 0")
    ax.set_ylabel("Citation precision\n(completed episodes)")
    ax.set_title("Same accuracy, different grounding", loc="left")
    fig.savefig(OUT / "fig4_3_em_vs_precision.pdf")
    fig.savefig(OUT / "fig4_3_em_vs_precision.png", dpi=300)
    plt.close(fig)

    # Figure 4.4: EM per seed for each reward.
    fig, ax = plt.subplots(figsize=(5.0, 3.0), layout="constrained")
    methods = ["R0", "R1", "R2"]
    seed_points = {}
    for i, method in enumerate(methods):
        names = [n for n in points if method_of(n) == method]
        values = [points[n]["em_all"] * 100 for n in names]
        seed_points[method] = dict(zip(names, values))
        st = STYLE[method]
        for n, v in zip(names, values):
            hollow = n in SEED9_FROM_BRIEF
            x = i + {"s7": -0.12, "s8": 0.0, "s9": 0.12}[n.split()[1]]
            ax.scatter(x, v, s=46, marker=st["marker"], facecolor="white" if hollow else st["color"],
                       edgecolor=st["color"], linewidth=1.5, zorder=3)
            ax.annotate(n.split()[1], (x, v), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7)
        mean = sum(values) / len(values)
        ax.hlines(mean, i - 0.25, i + 0.25, color="#333333", linewidth=1.5)
        ax.annotate(f"mean {mean:.1f}", (i - 0.25, mean), xytext=(-4, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=7)
    ax.axhline(points["B0"]["em_all"] * 100, color="#666666", linestyle=":", linewidth=1)
    ax.text(2.45, points["B0"]["em_all"] * 100 + 0.8, "B0", color="#666666", fontsize=7.5)
    ax.set_xticks(range(3), ["R0 (outcome-only)\n2 seeds [TODO s9]", "R1 (VEGR)\n3 seeds", "R2 (gated)\n3 seeds"])
    ax.set_xlim(-0.7, 2.7)
    ax.set_ylim(15, 55)
    ax.set_ylabel("EM on holdout (%)")
    ax.set_title("Holdout EM per seed", loc="left")
    fig.savefig(OUT / "fig4_4_em_per_seed.pdf")
    fig.savefig(OUT / "fig4_4_em_per_seed.png", dpi=300)
    plt.close(fig)

    # Figure 4.5: paired bootstrap forest plot.
    comparisons = [("R0 s7", "B0"), ("R0 s8", "B0"), ("R1 s7", "B0"), ("R1 s8", "B0"),
                   ("R2 s7", "B0"), ("R2 s8", "B0"), ("R1 s7", "R0 s7"), ("R1 s8", "R0 s8"),
                   ("R2 s7", "R1 s7"), ("R2 s8", "R1 s8")]
    boot = {f"{a} − {b}": paired_bootstrap(episodes[a], episodes[b]) for a, b in comparisons}
    fig, ax = plt.subplots(figsize=(5.5, 3.6), layout="constrained")
    labels = list(boot)
    for y, label in enumerate(reversed(labels)):
        r = boot[label]
        color = STYLE[method_of(label)]["color"]
        ax.errorbar(r["delta"], y, xerr=[[r["delta"] - r["low"]], [r["high"] - r["delta"]]],
                    fmt=STYLE[method_of(label)]["marker"], color=color, capsize=3, markersize=5)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.axhline(3.5, color="#BBBBBB", linewidth=0.8, linestyle="--")
    ax.text(-8.5, 3.75, "vs. B0 (RQ1)", fontsize=7, ha="left", va="bottom", color="#555555")
    ax.text(15, 3.25, "between rewards (RQ2/RQ3)", fontsize=7, ha="left", va="top", color="#555555")
    ax.set_yticks(range(len(labels)), list(reversed(labels)))
    ax.set_xlabel("ΔEM (points), paired bootstrap 95% CI")
    ax.set_title("Paired differences in holdout EM", loc="left")
    fig.savefig(OUT / "fig4_5_bootstrap.pdf")
    fig.savefig(OUT / "fig4_5_bootstrap.png", dpi=300)
    plt.close(fig)

    # Figure 4.6: failure breakdown.
    fig, ax = plt.subplots(figsize=(5.5, 3.0), layout="constrained")
    names = list(summaries)
    kind_style = {"evidence": ("#CC79A7", ""), "protocol": ("#E69F00", "//"), "budget": ("#56B4E9", "..")}
    bottoms = [0] * len(names)
    for kind in FAILURE_KINDS:
        values = [summaries[n]["failures"][kind] for n in names]
        color, hatch = kind_style[kind]
        ax.bar(range(len(names)), values, bottom=bottoms, color=color, hatch=hatch,
               edgecolor="white", width=0.7, label=kind)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    for i, total in enumerate(bottoms):
        ax.text(i, total + 2, str(total), ha="center", fontsize=7.5)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("Failed episodes (of 300)")
    ax.set_title("Failed holdout episodes by type", loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(OUT / "fig4_6_failures.pdf")
    fig.savefig(OUT / "fig4_6_failures.png", dpi=300)
    plt.close(fig)

    # Figure 4.7: cited sentence index, R1 s8.
    cited, gold = {}, {}
    for row in episodes["R1 s8"].values():
        if row["status"] != "completed":
            continue
        for c in row["citations"]:
            index = int(c.rsplit("|", 1)[1])
            cited[index] = cited.get(index, 0) + 1
    fig, ax = plt.subplots(figsize=(5.0, 2.8), layout="constrained")
    xs = sorted(cited)
    ax.bar(xs, [cited[x] for x in xs], color=STYLE["R1"]["color"], edgecolor="white", width=0.7)
    for x in xs:
        ax.text(x, cited[x] + 3, str(cited[x]), ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_ylim(0, max(cited.values()) * 1.12)
    ax.set_xlabel("Sentence index within the cited paragraph")
    ax.set_ylabel("Citations")
    ax.set_title("Which sentences R1 (seed 8) cites", loc="left")
    fig.savefig(OUT / "fig4_7_cited_index.pdf")
    fig.savefig(OUT / "fig4_7_cited_index.png", dpi=300)
    plt.close(fig)

    return {"points": points, "seed_points": seed_points, "bootstrap": boot,
            "failures": {n: summaries[n]["failures"] for n in names},
            "cited_sentence_index_r1_s8": cited}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    system_diagram()
    ch4 = ch4_figures()
    data = {name: load(path) for name, path in RUNS.items()}

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
        "episode_sources": {name: str(path.relative_to(ROOT)) for name, path in EPISODES.items()},
        "fig4_3_to_4_7": ch4,
    }
    print(json.dumps({k: {kk: round(vv, 1) for kk, vv in v.items() if kk != "n" and kk != "resamples"} for k, v in ch4["bootstrap"].items()}))
    print(json.dumps(ch4["failures"]), json.dumps(ch4["cited_sentence_index_r1_s8"]))
    (OUT / "figures_data.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance["fig4_1_skipped_steps"]))
    for name, blocks in curves.items():
        print(name, [(b["steps"], round(b["mean_precision"], 3) if b["mean_precision"] is not None else None, b["n_valid_rollouts"]) for b in blocks])


if __name__ == "__main__":
    main()
