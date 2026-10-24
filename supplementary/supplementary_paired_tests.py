"""Supplementary analysis for the KATMO manuscript (reads results/runs.jsonl written by experiment.py).

Usage:  python supplementary_paired_tests.py --res results

Prints (i) every method's cost-success frontier on the test split, (ii) paired tests of KATMO against the
reference, the query-level routers and the ablations, (iii) per-workload frontiers, and (iv) the lambda sweep on
the training split. These tests were run after the main analysis (analyze.py) and are not corrected for multiple
comparisons; they are descriptive.
"""
import argparse, json, collections
import numpy as np
from scipy.stats import binomtest, wilcoxon

ap = argparse.ArgumentParser(); ap.add_argument("--res", default="results"); a = ap.parse_args()
import gzip, os
_p = f"{a.res}/runs.jsonl"
rows = [json.loads(l) for l in (open(_p) if os.path.exists(_p) else gzip.open(_p + ".gz", "rt"))]
T = [r for r in rows if r["split"] == "test"]
Tr = [r for r in rows if r["split"] == "train"]
LAM = 1e6  # lambda* selected on the training split for every method


def get(m, lam=None, ds=None, pool=T):
    return {r["task_id"]: r for r in pool if r["method"] == m and (lam is None or r.get("lam") == lam)
            and (ds is None or r["dataset"] == ds)}


def point(d, ids, cref):
    return 100 * np.mean([d[i]["success"] for i in ids]), 100 * sum(d[i]["cost"] for i in ids) / cref


ref = get("static:L@1024"); ids = sorted(ref); cref = sum(ref[i]["cost"] for i in ids)
print(f"n test tasks = {len(ids)}; reference success = {100*np.mean([ref[i]['success'] for i in ids]):.1f}")

print("\n== Test-split frontiers (success %, cost % of reference) ==")
for m in ["katmo", "katmo-norules", "katmo-noesc", "route-model", "route-budget", "route-joint"]:
    print(m)
    for l in sorted({r["lam"] for r in T if r["method"] == m}):
        s, c = point(get(m, l), ids, cref); print(f"   lambda={l:<9g} success={s:5.1f} cost={c:5.1f}")
print("static grid")
for t in "SML":
    for b in (128, 256, 512, 1024):
        s, c = point(get(f"static:{t}@{b}"), ids, cref); print(f"   {t}@{b:<5} success={s:5.1f} cost={c:5.1f}")


def paired(x, y, name, rng):
    sx = np.array([x[i]["success"] for i in ids], float); sy = np.array([y[i]["success"] for i in ids], float)
    cx = np.array([x[i]["cost"] for i in ids]); cy = np.array([y[i]["cost"] for i in ids])
    b = int(((sx == 1) & (sy == 0)).sum()); c = int(((sx == 0) & (sy == 1)).sum())
    p = binomtest(b, b + c, .5).pvalue if b + c else 1.0
    pc = wilcoxon(cx, cy).pvalue if np.any(cx != cy) else 1.0
    bs_s, bs_c = [], []
    for _ in range(5000):
        ix = rng.integers(0, len(ids), len(ids))
        bs_s.append(100 * (sx[ix].mean() - sy[ix].mean()))
        bs_c.append(100 * (cx[ix].sum() - cy[ix].sum()) / cy[ix].sum())
    print(f"{name}: d_success={100*(sx.mean()-sy.mean()):+.1f} pts, 95% CI [{np.percentile(bs_s,2.5):.1f}, {np.percentile(bs_s,97.5):.1f}], "
          f"discordant {b}/{c}, McNemar p={p:.3f}; cost ratio={100*cx.sum()/cy.sum():.1f}% "
          f"(diff 95% CI [{np.percentile(bs_c,2.5):.1f}, {np.percentile(bs_c,97.5):.1f}] pts of comparator cost), Wilcoxon p={pc:.2g}")


print("\n== Paired comparisons on the 300 test tasks (KATMO at lambda*) ==")
rng = np.random.default_rng(0); K = get("katmo", LAM)
paired(K, ref, "KATMO vs static L@1024 (reference)", rng)
paired(K, get("static:L@128"), "KATMO vs static L@128", rng)
for m in ["route-joint", "route-model", "route-budget", "katmo-norules", "katmo-noesc"]:
    paired(K, get(m, LAM), f"KATMO vs {m}", rng)

tier = collections.Counter(s["cfg"].split("@")[0] for i in ids for s in K[i]["steps"]); n = sum(tier.values())
print("\nKATMO step share by tier at lambda*:", {k: round(100 * v / n, 1) for k, v in tier.items()})

print("\n== Per-workload frontiers ==")
for ds in ["gsm8k", "hotpotqa"]:
    rd = get("static:L@1024", ds=ds); di = sorted(rd); cr = sum(rd[i]["cost"] for i in di)
    print(ds, f"n={len(di)} reference success={100*np.mean([rd[i]['success'] for i in di]):.1f}")
    for m in ["katmo", "route-joint", "route-model", "route-budget"]:
        for l in sorted({r["lam"] for r in T if r["method"] == m}):
            s, c = point(get(m, l, ds), di, cr); print(f"   {m:12s} lambda={l:<9g} success={s:5.1f} cost={c:5.1f}")
    for k in ["static:S@1024", "static:M@128", "static:M@512", "static:M@1024", "static:L@128", "static:L@512"]:
        s, c = point(get(k, ds=ds), di, cr); print(f"   {k:14s} success={s:5.1f} cost={c:5.1f}")
ds = "hotpotqa"; di = sorted(get("static:L@1024", ds=ds))
ids_backup, ids = ids, di
for l in (0.8, 1.6):
    paired(get("katmo", l, ds), get("static:M@1024", ds=ds), f"HotpotQA KATMO(lambda={l}) vs static M@1024", rng)
    paired(get("katmo", l, ds), get("route-joint", l, ds), f"HotpotQA KATMO(lambda={l}) vs query-level joint(lambda={l})", rng)
ids = ids_backup

print("\n== Training-split lambda sweep (KATMO) ==")
tr = get("static:L@1024", pool=Tr); tids = sorted(tr); ctr = sum(tr[i]["cost"] for i in tids)
print(f"train reference success = {100*np.mean([tr[i]['success'] for i in tids]):.1f}")
for l in sorted({r["lam"] for r in Tr if r["method"] == "katmo"}):
    s, c = point(get("katmo", l, pool=Tr), tids, ctr); print(f"   lambda={l:<9g} success={s:5.1f} cost={c:5.1f}")
