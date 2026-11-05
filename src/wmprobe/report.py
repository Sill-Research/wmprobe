"""Render results.json into report.md plus figures/.

uv run python -m wmprobe.report runs/full
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical order
PLAIN_COLOR, WM_COLOR = SERIES[0], SERIES[1]
PREFERRED_WM = ["kgw_g0.25_d2", "synthid_n2", "synthid_n4", "kgw_g0.25_d4", "gumbel_h4"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 6,
    }
)


def _fmt(x, digits: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def _plain(conds) -> list[str]:
    return [c for c in conds if c.startswith("plain")]


def _wm_series(conds) -> list[str]:
    chosen = [c for c in PREFERRED_WM if c in conds]
    return (chosen + [c for c in conds if not c.startswith("plain") and c not in chosen])[:3]


def _finite(xs) -> np.ndarray:
    return np.array([x for x in xs if x is not None and np.isfinite(x)], dtype=float)


def _rate_panel(ax, per: dict, conds, levels: list[str], key: str, alpha: float, xlabel: str):
    """Rejection rate vs a size axis: worst plain condition (false positives) + watermarks."""
    x = [int(level) for level in levels]

    def rate(c, level):
        return per[c][key][level] if key else per[c][level]["reject01"]

    worst = [max(rate(c, lv) for c in _plain(conds)) for lv in levels]
    series = [("plain, worst temperature", worst)] + [
        (c, [rate(c, lv) for lv in levels]) for c in _wm_series(conds)
    ]
    for (name, ys), color in zip(series, SERIES, strict=False):
        ax.plot(x, ys, "-o", color=color, label=name, markeredgecolor=SURFACE)
    ax.axhline(alpha, color=INK2, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks(x, [str(v) for v in x])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=8, loc="center right", labelcolor=INK2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("rejection rate")


def _box_panel(ax, per: dict, conds, field: str, title: str, symlog: bool = False):
    data = [_finite(per[c][field]) for c in conds]
    bp = ax.boxplot(
        data,
        orientation="horizontal",
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": INK},
        flierprops={"markersize": 3},
    )
    for patch, c in zip(bp["boxes"], conds, strict=True):
        patch.set_facecolor(PLAIN_COLOR if c.startswith("plain") else WM_COLOR)
        patch.set_alpha(0.75)
        patch.set_edgecolor(SURFACE)
    ax.set_yticks(range(1, len(conds) + 1), conds)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, linewidth=0.8)
    if symlog:
        ax.set_xscale("symlog", linthresh=3)
    ax.set_title(title, loc="left")


def render(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    r = json.loads((run_dir / "results.json").read_text())
    figs = run_dir / "figures"
    figs.mkdir(exist_ok=True)
    proxies = list(r["test_a"])
    conds = list(r["test_a"][proxies[0]]["per_condition"])
    plain = _plain(conds)
    md = [
        "# Keyless watermark detection — results",
        "",
        f"Generator: GPT-2. Proxies: {', '.join(proxies)}. "
        f"Corpus: {r['config']['n_prompts']} texts × {r['config']['n_new']} tokens per condition; "
        f"Tests A/B skip the first {r['config']['burn_in']} tokens (hidden-prompt burn-in).",
        "AUC: watermarked condition (positive) vs a plain condition (negative). "
        "Rejection rate: fraction of trials whose test fires (one-sided).",
        "",
    ]

    # ---- keyed
    md += [
        "## Keyed baselines",
        "",
        _table(
            ["condition", "mean z", "AUC vs plain_t1.0"],
            [
                [c, _fmt(np.mean(v["z"])), _fmt(v.get("auc_vs_plain_t1.0"), 3)]
                for c, v in r["keyed"].items()
            ],
        ),
        "",
    ]

    # ---- Test A
    md += ["## Test A — single-text likelihood deficit", ""]
    fig, axes = plt.subplots(
        1,
        len(proxies),
        figsize=(4.6 * len(proxies), 0.32 * len(conds) + 1.4),
        squeeze=False,
        sharey=True,
    )
    for ax, p in zip(axes[0], proxies, strict=True):
        a = r["test_a"][p]
        _box_panel(ax, a["per_condition"], conds, "z", f"proxy: {p}")
        ax.set_xlabel("Z per text (negative = less likely than expected)")
        full = max(r["config"]["lengths"])
        rows = []
        for c in conds:
            aucs = [_fmt(a["auc"].get(c, {}).get(pl, {}).get(str(full)), 3) for pl in plain]
            rows.append([c, _fmt(a["per_condition"][c]["mean_z"])] + aucs)
        md += [
            f"**Proxy {p}** (AUC at {full} tokens)",
            "",
            _table(["condition", "mean Z"] + [f"AUC vs {pl}" for pl in plain], rows),
            "",
        ]
    fig.tight_layout()
    fig.savefig(figs / "test_a_z.png", dpi=150)
    plt.close(fig)

    negs = [pl for pl in ("plain_t1.0", "plain_t1.2") if pl in plain] or plain[:1]
    fig, axes = plt.subplots(
        len(proxies),
        len(negs),
        figsize=(4.8 * len(negs), 3.0 * len(proxies)),
        squeeze=False,
        sharey=True,
    )
    lengths = [str(L) for L in r["config"]["lengths"]]
    for i, p in enumerate(proxies):
        for j, neg in enumerate(negs):
            ax = axes[i][j]
            for c, color in zip(_wm_series(conds), SERIES[1:], strict=False):
                ys = [r["test_a"][p]["auc"][c][neg][L] for L in lengths]
                xs = [int(L) for L in lengths]
                ax.plot(xs, ys, "-o", color=color, label=c, markeredgecolor=SURFACE)
            ax.axhline(0.5, color=INK2, linestyle="--", linewidth=1)
            ax.set_xscale("log")
            ax.set_xticks([int(L) for L in lengths], lengths)
            ax.minorticks_off()
            ax.set_ylim(-0.02, 1.02)
            ax.legend(frameon=False, fontsize=8, loc="best", labelcolor=INK2)
            ax.set_title(f"proxy {p}: watermark vs {neg}", loc="left")
            ax.set_xlabel("text length (tokens)")
            ax.set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig(figs / "test_a_auc_vs_length.png", dpi=150)
    plt.close(fig)
    md += [
        "![Test A Z](figures/test_a_z.png)",
        "",
        "![Test A AUC vs length](figures/test_a_auc_vs_length.png)",
        "",
    ]

    # ---- Test B
    md += ["## Test B — corpus bigram residuals (temperature fitted out)", ""]
    sizes = [str(m) for m in r["config"]["corpus_sizes"]]
    fig, axes = plt.subplots(
        1, len(proxies), figsize=(5.2 * len(proxies), 3.2), squeeze=False, sharey=True
    )
    for ax, p in zip(axes[0], proxies, strict=True):
        per = r["test_b"][p]["per_condition"]
        sz = [m for m in sizes if all(m in per[c]["reject05"] for c in conds)]
        _rate_panel(ax, per, conds, sz, "reject05", 0.05, "corpus size (texts)")
        ax.set_title(f"proxy {p}: sign-flip p < 0.05", loc="left")
        rows = [
            [
                c,
                _fmt(1 / per[c]["s_hat"]),
                _fmt(per[c]["full_z"], 1),
                _fmt(per[c]["full_p"], 4),
                _fmt(per[c]["full_p_no_burn_in"], 4),
            ]
            + [_fmt(per[c]["reject05"].get(m)) for m in sz]
            for c in conds
        ]
        md += [
            f"**Proxy {p}**",
            "",
            _table(
                ["condition", "fitted τ", "z (full)", "p (full)", "p (full, no burn-in)"]
                + [f"reject @ {m} texts" for m in sz],
                rows,
            ),
            "",
        ]
    fig.tight_layout()
    fig.savefig(figs / "test_b_rejection.png", dpi=150)
    plt.close(fig)
    md += ["![Test B rejection](figures/test_b_rejection.png)", ""]

    # ---- Test C
    md += ["## Test C — query access, context × answer interaction", ""]
    ns = [str(n) for n in r["config"]["c_queries"]]
    fig, axes = plt.subplots(
        1, len(proxies), figsize=(5.2 * len(proxies), 3.2), squeeze=False, sharey=True
    )
    for ax, p in zip(axes[0], proxies, strict=True):
        per = r["test_c"][p]["per_condition"]
        _rate_panel(ax, per, conds, ns, "", 0.01, "queries per prompt variant")
        ax.set_title(f"proxy {p}: p < 0.01", loc="left")
        rows = [
            [c]
            + [_fmt(per[c][n]["reject01"]) for n in ns]
            + [_fmt(per[c][ns[-1]]["answer_set_size"], 1)]
            for c in conds
        ]
        md += [
            f"**Proxy {p}** (separators: {r['test_c'][p]['separators']})",
            "",
            _table(
                ["condition"] + [f"reject @ N={n}" for n in ns] + [f"answers |W| @ N={ns[-1]}"],
                rows,
            ),
            "",
        ]
    fig.tight_layout()
    fig.savefig(figs / "test_c_rejection.png", dpi=150)
    plt.close(fig)
    md += ["![Test C rejection](figures/test_c_rejection.png)", ""]

    # ---- Test D
    md += ["## Test D — query access, diversity collapse", ""]
    fig, axes = plt.subplots(
        1,
        len(proxies),
        figsize=(4.6 * len(proxies), 0.32 * len(conds) + 1.4),
        squeeze=False,
        sharey=True,
    )
    for ax, p in zip(axes[0], proxies, strict=True):
        per = r["test_d"][p]["per_condition"]
        _box_panel(ax, per, conds, "z", f"proxy: {p}", symlog=True)
        ax.set_xlabel("Z per prompt (symlog; positive = too little diversity)")
        rows = [
            [c, _fmt(1 / per[c]["s_hat"]), _fmt(per[c]["pooled_z"], 1), _fmt(per[c]["reject05"])]
            for c in conds
        ]
        md += [
            f"**Proxy {p}**",
            "",
            _table(["condition", "fitted τ", "pooled z", "per-prompt reject @ 5%"], rows),
            "",
        ]
    fig.tight_layout()
    fig.savefig(figs / "test_d_z.png", dpi=150)
    plt.close(fig)
    md += ["![Test D](figures/test_d_z.png)", ""]

    path = run_dir / "report.md"
    path.write_text("\n".join(md))
    return path


if __name__ == "__main__":
    print(render(Path(sys.argv[1])))
