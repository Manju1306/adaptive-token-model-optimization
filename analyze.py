"""
Builds the paper's two tables and two figures from results/runs.jsonl (no GPU needed).

    python analyze.py --res results          # -> results/table1.csv, table2.csv, tables.md, fig1_pareto.*, fig2_allocation.*

All statistics are on the held-out TEST split; the operating point lambda* was selected on TRAIN.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

import katmo as K

# palette (validated reference instance): series 1-3, neutrals, sequential blue ramp
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
REFN = f"static:{K.cfg_name(K.REF)}"


def load(res):
    rows = [json.loads(l) for l in open(f"{res}/runs.jsonl")]
    meta = json.load(open(f"{res}/meta.json"))
    return rows, meta


def pick(rows, method, lam=None, split="test", ds=None):
    out = [r for r in rows if r["method"] == method and r["split"] == split and (lam is None or r["lam"] == lam)
           and (ds is None or r["dataset"] == ds)]
    return sorted(out, key=lambda r: r["task_id"])


def recost(r, w_in):
    """Recompute a task's cost under a different input-token weight from the stored per-step tokens."""
    return sum(K.step_cost(s["cfg"].split("@")[0], s["n_in"], s["n_out"], w_in) for s in r["steps"])


def stats(rr, ref, w_in=None, n_boot=2000, seed=0):
    ids = [r["task_id"] for r in ref]
    m = {r["task_id"]: r for r in rr}
    rr = [m[i] for i in ids if i in m]
    ref = [r for r in ref if r["task_id"] in m]
    s = np.array([r["success"] for r in rr], float)
    s0 = np.array([r["success"] for r in ref], float)
    c = np.array([recost(r, w_in) if w_in else r["cost"] for r in rr])
    c0 = np.array([recost(r, w_in) if w_in else r["cost"] for r in ref])
    rng = np.random.default_rng(seed)
    bs = [s[idx].mean() for idx in (rng.integers(0, len(s), len(s)) for _ in range(n_boot))]
    # exact McNemar (discordant pairs) for success vs reference; Wilcoxon signed-rank for cost
    b, cc = int(((s == 1) & (s0 == 0)).sum()), int(((s == 0) & (s0 == 1)).sum())
    p_succ = binomtest(b, b + cc, 0.5).pvalue if b + cc > 0 else 1.0
    p_cost = wilcoxon(c, c0).pvalue if np.any(c != c0) else 1.0
    steps = [st for r in rr for st in r["steps"]]
    return {"n": len(s), "success": 100 * s.mean(), "ci_lo": 100 * np.percentile(bs, 2.5), "ci_hi": 100 * np.percentile(bs, 97.5),
            "rel_cost": 100 * c.sum() / c0.sum(), "tokens": float(np.mean([r["n_in"] + r["n_out"] for r in rr])),
            "out_tokens": float(np.mean([r["n_out"] for r in rr])),
            "esc_rate": 100 * np.mean([st["esc_from"] is not None for st in steps]) if steps else 0.0,
            "p_success": p_succ, "p_cost": p_cost}


def frontier(rows, method, ds=None, split="test"):
    ref = pick(rows, REFN, split=split, ds=ds)
    pts = []
    for lam in sorted({r["lam"] for r in rows if r["method"] == method}):
        rr = pick(rows, method, lam, split, ds)
        if rr:
            st = stats(rr, ref, n_boot=1)
            pts.append((st["rel_cost"], st["success"], lam))
    return pts  # ordered by lambda (cost-quality path)


def iso_cost(rows, method, ds=None, tol=1.0):
    """Lowest relative cost on the TEST frontier whose success is within `tol` points of the reference."""
    ref = pick(rows, REFN, ds=ds)
    target = 100 * np.mean([r["success"] for r in ref]) - tol
    ok = [c for c, s, _ in frontier(rows, method, ds) if s >= target]
    return min(ok) if ok else float("nan")


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def build_tables(rows, meta, out):
    lam = meta["lambda_star"]
    methods = [("Static S@1024", "static:S@1024", None), ("Static M@1024", "static:M@1024", None),
               ("Static L@512", "static:L@512", None), ("Static L@1024 (reference)", REFN, None),
               ("Model routing only", "route-model", lam["route-model"]),
               ("Token budgeting only", "route-budget", lam["route-budget"]),
               ("Query-level joint (model+budget)", "route-joint", lam["route-joint"]),
               ("KATMO (step-level, this work)", "katmo", lam["katmo"])]
    fam = {"route-model": "route-model", "route-budget": "route-budget", "route-joint": "route-joint", "katmo": "katmo"}
    t1 = []
    for name, m, l in methods:
        row = {"Method": name}
        for ds, tag in (("gsm8k", "GSM8K"), ("hotpotqa", "HotpotQA")):
            st = stats(pick(rows, m, l, ds=ds), pick(rows, REFN, ds=ds))
            row[f"{tag} success (%)"] = f"{st['success']:.1f}"
            row[f"{tag} cost (%)"] = f"{st['rel_cost']:.1f}"
        st = stats(pick(rows, m, l), pick(rows, REFN))
        row.update({"Overall success (%) [95% CI]": f"{st['success']:.1f} [{st['ci_lo']:.1f}, {st['ci_hi']:.1f}]",
                    "Overall cost (% of ref.)": f"{st['rel_cost']:.1f}", "Tokens/task": f"{st['tokens']:.0f}",
                    "Iso-success cost (%)": (f"{iso_cost(rows, fam[m]):.1f}" if m in fam else "-"),
                    "p (success vs ref.)": fmt_p(st["p_success"]) if m != REFN else "-",
                    "p (cost vs ref.)": fmt_p(st["p_cost"]) if m != REFN else "-"})
        t1.append(row)

    ls = lam["katmo"]
    t2 = []
    ref = pick(rows, REFN)
    def add(name, rr, note="", w_in=None):
        st = stats(rr, ref, w_in=w_in)
        t2.append({"Variant": name, "Success (%) [95% CI]": f"{st['success']:.1f} [{st['ci_lo']:.1f}, {st['ci_hi']:.1f}]",
                   "Cost (% of ref.)": f"{st['rel_cost']:.1f}", "Output tokens/task": f"{st['out_tokens']:.0f}",
                   "Escalated steps (%)": f"{st['esc_rate']:.1f}", "p (success vs ref.)": fmt_p(st["p_success"]), "Note": note})
    add("KATMO (full)", pick(rows, "katmo", ls), f"lambda*={ls:g} x C_ref (selected on train)")
    add("- without knowledge rules (R1-R6)", pick(rows, "katmo-norules", ls), "ablation")
    add("- without escalation (R6)", pick(rows, "katmo-noesc", ls), "ablation")
    add("Rules + cheapest admissible (lambda=0)", pick(rows, "katmo", 0.0), "no estimator trade-off")
    for frac in (0.75, 0.5):
        rr = pick(rows, f"katmo-B{frac}", ls)
        if rr:
            add(f"KATMO, task budget = {int(frac * 100)}% of ref. cost", rr, "budget sensitivity")
    for w in (0.1, 0.5, 1.0):
        add(f"KATMO, input-token weight w_in={w}", pick(rows, "katmo", ls), "price sensitivity", w_in=w)

    import csv
    for name, t in (("table1", t1), ("table2", t2)):
        with open(f"{out}/{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(t[0].keys())); w.writeheader(); w.writerows(t)
    with open(f"{out}/tables.md", "w") as f:
        if meta.get("mock"):
            f.write("> **MOCK DATA - CODE TEST ONLY. NOT FOR THE PAPER.**\n\n")
        for title, t in (("Table 1. Main results on the held-out test split", t1), ("Table 2. Ablation and sensitivity (test split)", t2)):
            f.write(f"### {title}\n\n| " + " | ".join(t[0]) + " |\n|" + "---|" * len(t[0]) + "\n")
            for r in t:
                f.write("| " + " | ".join(str(v) for v in r.values()) + " |\n")
            f.write("\n")
        e = meta.get("estimator", {})
        f.write(f"Step-value estimator: grouped 5-fold CV AUC = {e.get('step_auc')}, Brier = {e.get('step_brier')}, rows = {e.get('n_rows')}.\n")
    return t1, t2


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=7, width=0.6, length=3)
    ax.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)


def _mock_mark(fig, meta):
    if meta.get("mock"):
        fig.text(0.5, 0.5, "MOCK DATA - CODE TEST ONLY", ha="center", va="center", fontsize=22, color="#e34948",
                 alpha=0.25, rotation=20, transform=fig.transFigure)


def fig_pareto(rows, meta, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    shapes = {"S": "o", "M": "s", "L": "^"}
    for ax, (ds, title) in zip(axes, (("gsm8k", "(a) GSM8K: plan, reason, tool, verify, answer"),
                                      ("hotpotqa", "(b) HotpotQA: plan, retrieve, reason, verify, answer"))):
        _style(ax)
        ref = pick(rows, REFN, ds=ds)
        for cfg in K.CONFIGS:
            st = stats(pick(rows, f"static:{K.cfg_name(cfg)}", ds=ds), ref, n_boot=1)
            ax.scatter(st["rel_cost"], st["success"], marker=shapes[cfg[0]], s=10 + 9 * np.log2(cfg[1] / 64),
                       facecolor="#d6d5cf", edgecolor=MUTED, linewidth=0.6, zorder=2)
            if cfg == K.REF:
                ax.annotate("L@1024 (ref.)", (st["rel_cost"], st["success"]), xytext=(-4, 6), textcoords="offset points",
                            ha="right", fontsize=6.5, color=INK2)
        for m, col, ls, mk, lab in (("route-model", AQUA, ":", "D", "Model routing only"),
                                    ("route-budget", AQUA, "--", "v", "Token budgeting only"),
                                    ("route-joint", ORANGE, "-", "s", "Query-level joint"),
                                    ("katmo", BLUE, "-", "o", "KATMO (step-level)")):
            pts = frontier(rows, m, ds)
            if not pts:
                continue
            x, y = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(x, y, ls, color=col, linewidth=2 if m == "katmo" else 1.4, zorder=3)
            ax.scatter(x, y, marker=mk, s=16, color=col, edgecolor="white", linewidth=0.8, zorder=4)
            lst = meta["lambda_star"].get(m)
            star = [p for p in pts if p[2] == lst]
            if star and m in ("katmo", "route-joint"):
                ax.scatter(star[0][0], star[0][1], marker="*", s=120, color=col, edgecolor="white", linewidth=0.8, zorder=5)
        ax.set_xscale("log")
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if str(int(round(v)))[0] in "25" else ""))
        ax.set_xlabel("Inference cost (% of reference, log scale)", color=INK2, fontsize=7.5)
        ax.set_ylabel("Task success (%)", color=INK2, fontsize=7.5)
        ax.set_title(title, fontsize=7.5, color=INK, loc="left")
    handles = [Line2D([], [], color=BLUE, lw=2, marker="o", ms=4, label="KATMO (step-level)"),
               Line2D([], [], color=ORANGE, lw=1.4, marker="s", ms=4, label="Query-level joint"),
               Line2D([], [], color=AQUA, lw=1.4, ls="--", marker="v", ms=4, label="Token budgeting only"),
               Line2D([], [], color=AQUA, lw=1.4, ls=":", marker="D", ms=4, label="Model routing only"),
               Line2D([], [], color=MUTED, lw=0, marker="o", mfc="#d6d5cf", ms=4, label="Static S (1.5B)"),
               Line2D([], [], color=MUTED, lw=0, marker="s", mfc="#d6d5cf", ms=4, label="Static M (7B)"),
               Line2D([], [], color=MUTED, lw=0, marker="^", mfc="#d6d5cf", ms=4, label="Static L (32B)"),
               Line2D([], [], color=INK2, lw=0, marker="*", ms=8, label="Operating point $\\lambda^*$ (train)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _mock_mark(fig, meta)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/fig1_pareto.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def fig_allocation(rows, meta, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    lam = meta["lambda_star"]["katmo"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.18})
    cols = [K.cfg_name(c) for c in K.CONFIGS]
    steps = K.STEP_TYPES
    for k_ax, (ax, (ds, title)) in enumerate(zip(axes, (("gsm8k", "(a) GSM8K"), ("hotpotqa", "(b) HotpotQA")))):
        M = np.zeros((len(steps), len(cols)))
        esc = np.zeros(len(steps))
        for r in pick(rows, "katmo", lam, ds=ds):
            for st in r["steps"]:
                i = steps.index(st["step"])
                M[i, cols.index(st["cfg"])] += 1
                esc[i] += st["esc_from"] is not None
        n = M.sum(1, keepdims=True)
        P = 100 * M / np.maximum(n, 1)
        ax.imshow(P, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if P[i, j] >= 5:
                    ax.text(j, i, f"{P[i, j]:.0f}", ha="center", va="center", fontsize=6.3,
                            color="white" if P[i, j] > 55 else INK)
            ax.text(len(cols) - 0.2, i, f"{100 * esc[i] / max(n[i, 0], 1):.0f}%", ha="left", va="center",
                    fontsize=6.3, color=INK2)
        ax.text(len(cols) - 0.2, -0.75, "esc.", ha="left", va="center", fontsize=6.3, color=MUTED)
        ax.set_xticks(range(len(cols)), [str(c[1]) for c in K.CONFIGS], fontsize=5.8, color=INK2, rotation=90)
        if k_ax == 0:
            ax.set_yticks(range(len(steps)), [s.upper() for s in steps], fontsize=6.8, color=INK2)
        else:
            ax.set_yticks([])
        for b in (3.5, 7.5):
            ax.axvline(b, color="white", linewidth=2.5)
        for k, t in enumerate(K.TIERS):
            ax.text(1.5 + 4 * k, -0.75, f"{t} ({K.MODELS[t][1]:g}B)", ha="center", va="center", fontsize=6.6, color=INK)
        ax.set_xlabel("Output-token budget within model tier", color=INK2, fontsize=7)
        ax.set_title(title, fontsize=7.5, color=INK, loc="left", pad=16)
        ax.set_xlim(-0.5, len(cols) + 0.6)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label("Share of steps assigned (%)", fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=6.5, colors=INK2, length=2)
    cb.outline.set_visible(False)
    _mock_mark(fig, meta)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/fig2_allocation.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    a = ap.parse_args()
    rows, meta = load(a.res)
    build_tables(rows, meta, a.res)
    fig_pareto(rows, meta, a.res)
    fig_allocation(rows, meta, a.res)
    print(open(f"{a.res}/tables.md").read())
