"""Extra offline analyses for the KATMO manuscript (no GPU needed). Reads results/runs.jsonl.
Usage: python extra_analyses.py --res results
A: hindsight (oracle) headroom from the interventional step profiles (TRAIN split, descriptive upper bound)
B: price-ratio sensitivity of the measured costs (allocations are NOT re-optimised)
C: knowledge-rule firing frequencies at lambda* (TEST split)
D: best success reachable at fixed cost budgets, per method family (TEST frontier, descriptive)
"""
import argparse, json, collections
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--res", default="results"); a = ap.parse_args()
import gzip, os
_p = f"{a.res}/runs.jsonl"
rows = [json.loads(l) for l in (open(_p) if os.path.exists(_p) else gzip.open(_p + ".gz", "rt"))]
P = {"S": 1.5, "M": 7.0, "L": 32.0}; W = 0.25
def cost(tier, n_in, n_out, params=P, w=W): return params[tier] * (w * n_in + n_out) / 1000.0

# ---------------- A: hindsight headroom ----------------
print("== A. Hindsight headroom from interventional profiles (train, 300 tasks, upper bound ignoring interactions) ==")
prof = [r for r in rows if r["method"] == "profile"]
by = collections.defaultdict(dict)
for r in prof: by[(r["task_id"], r["step"])][r["cfg"]] = r
tasks = sorted({r["task_id"] for r in prof}); ds_of = {r["task_id"]: r["dataset"] for r in prof}
refok = {t: bool(by[(t, s)]["L@1024"]["ref_success"]) for (t, s) in by for _ in [0]}
def sc(r): return cost(r["cfg"].split("@")[0], r["n_in"], r["n_out"])
tot = collections.defaultdict(lambda: [0.0, 0.0, 0, 0, 0])  # oracle cost, ref cost, steps, steps_SM, steps_ok
for (t, s), d in by.items():
    rc = sc(d["L@1024"]); ok = [(sc(r), r["cfg"]) for r in d.values() if r["success"]]
    key = ds_of[t]
    if refok[t] and ok:
        best = min(ok); o = best[0]; tier = best[1][0]
    else:
        o = rc; tier = "L"
    for k in (key, "all"):
        x = tot[k]; x[0] += o; x[1] += rc; x[2] += 1; x[3] += tier in "SM"
print("(tasks the reference solves: oracle picks the cheapest single-step config that keeps the task solved; others unchanged)")
for k, x in tot.items(): print(f"  {k:9s} oracle cost = {100*x[0]/x[1]:5.1f}% of reference; steps where S or M suffices = {100*x[3]/x[2]:5.1f}% (n={x[2]})")
st = [r for r in rows if r["split"] == "train" and r["method"].startswith("static:")]
sby = collections.defaultdict(dict)
for r in st: sby[r["task_id"]][r["method"].split(":")[1]] = r
oc = rc_ = 0.0; solved = 0
for t, d in sby.items():
    ref = d["L@1024"]; ok = [r["cost"] for r in d.values() if r["success"]]
    if ref["success"] and ok: oc += min(ok); solved += 1
    else: oc += ref["cost"]
    rc_ += ref["cost"]
print(f"  task-level hindsight (cheapest static config per task, train): cost = {100*oc/rc_:.1f}% of reference at the reference's success")
trk = {r["task_id"]: r for r in rows if r["split"] == "train" and r["method"] == "katmo" and r["lam"] == 1e6}
trr = {t: d["L@1024"] for t, d in sby.items()}
print(f"  KATMO at lambda* on train: cost = {100*sum(trk[t]['cost'] for t in trr)/sum(trr[t]['cost'] for t in trr):.1f}% of reference")

# ---------------- B: price-ratio sensitivity ----------------
print("\n== B. Relative cost under different 32B:7B price ratios (allocations fixed; S=1.5B, M=7B priced by parameters) ==")
T = [r for r in rows if r["split"] == "test"]
def get(m, lam=None): return {r["task_id"]: r for r in T if r["method"] == m and (lam is None or r.get("lam") == lam)}
meths = {"KATMO": get("katmo", 1e6), "joint": get("route-joint", 1e6), "model-only": get("route-model", 1e6), "budget-only": get("route-budget", 1e6),
         "static L@128": get("static:L@128"), "static M@1024": get("static:M@1024")}
ref = get("static:L@1024"); ids = sorted(ref)
def total(d, params): return sum(cost(s["cfg"].split("@")[0], s["n_in"], s["n_out"], params) for i in ids for s in d[i]["steps"])
print("  ratio  " + "  ".join(f"{k:>13s}" for k in meths))
for rr in (1.5, 2, 3, 32 / 7, 8, 16):
    params = {"S": 1.5, "M": 7.0, "L": 7.0 * rr}; cr = total(ref, params)
    print(f"  {rr:5.2f}  " + "  ".join(f"{100*total(d, params)/cr:12.1f}%" for d in meths.values()))
chk = sum(s["cost"] for i in ids for s in meths["KATMO"][i]["steps"]); print(f"  check: recomputed baseline KATMO cost differs from logged by {100*abs(chk - total(meths['KATMO'], P))/chk:.3f}% (escalated steps sum tokens across attempts)")

# ---------------- C: rule firing ----------------
print("\n== C. Knowledge-rule firing at lambda* (test, KATMO, share of steps per step type) ==")
K = meths["KATMO"]; cnt = collections.defaultdict(collections.Counter); n = collections.Counter()
for i in ids:
    for s in K[i]["steps"]:
        n[s["step"]] += 1
        for r in s["rules"]: cnt[s["step"]][r] += 1
rules = ["R1", "R2", "R3", "R4", "R5", "R6", "BUDGET"]
print("  step     n   " + "  ".join(f"{r:>6s}" for r in rules))
for st_ in ["plan", "reason", "tool", "retrieve", "verify", "answer"]:
    if n[st_]: print(f"  {st_:7s} {n[st_]:4d}  " + "  ".join(f"{100*cnt[st_][r]/n[st_]:5.1f}%" for r in rules))

# ---------------- D: success at fixed cost budgets ----------------
print("\n== D. Best test success reachable at or below a cost budget (% of reference), per family (descriptive, test frontier) ==")
def pts(m): 
    out = []
    for l in sorted({r["lam"] for r in T if r["method"] == m}):
        d = get(m, l); out.append((100*sum(d[i]["cost"] for i in ids)/sum(ref[i]["cost"] for i in ids), 100*np.mean([d[i]["success"] for i in ids])))
    return out
fam = {"KATMO": pts("katmo"), "joint": pts("route-joint"), "model-only": pts("route-model"), "budget-only": pts("route-budget"),
       "static (12)": [(100*sum(get(f"static:{t}@{b}")[i]["cost"] for i in ids)/sum(ref[i]["cost"] for i in ids), 100*np.mean([get(f"static:{t}@{b}")[i]["success"] for i in ids])) for t in "SML" for b in (128,256,512,1024)]}
budgets = (5, 10, 20, 30, 50, 80, 90, 100)
print("  budget " + "  ".join(f"{b:>5d}%" for b in budgets))
for k, p in fam.items():
    print(f"  {k:11s}" + "  ".join(f"{max([s for c, s in p if c <= b], default=float('nan')):6.1f}" for b in budgets))
