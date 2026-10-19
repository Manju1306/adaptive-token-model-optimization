"""
Generates the KATMO dataset and evaluates every method end-to-end on real LLM executions.

Phases (all results are cached and resumable; re-running skips completed generations):
  1. profile  : interventional step profiling on the TRAIN split
                (reference trajectory at L@1024; for every step and every config, re-run that one step
                 with the config and complete the trajectory with the reference config)
  2. static   : the 12 static configurations, end-to-end, on TRAIN and TEST
  3. fit      : step-value estimator (from profiling) and task-level router estimator (from TRAIN static)
  4. routers  : query-level baselines (model-only, budget-only, joint model+budget) for a lambda grid
  5. katmo    : KATMO and ablations for a lambda grid; operating point lambda* chosen on TRAIN only
  6. budget   : KATMO at lambda* under task-level budgets

Usage on the GPU machine (after runpod.sh started the three vLLM servers):
    python experiment.py --n-train 150 --n-test 150
Local code test without GPU (synthetic tasks, MOCK outputs, never for the paper):
    python experiment.py --mock --n-train 12 --n-test 12 --out results_mock
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import statistics
import subprocess
import time
from collections import defaultdict

import numpy as np

import katmo as K

LAMS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 1e6]   # multiples of the mean reference task cost


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def load_tasks(n_train, n_test, seed, mock):
    rng = random.Random(seed)
    if mock:
        tasks = []
        for i in range(n_train + n_test):
            d = rng.random()
            tasks.append({"id": f"gsm-{i}", "dataset": "gsm8k", "question": "Q" * rng.randint(150, 500),
                          "answer": str(rng.randint(1, 500)), "difficulty": d})
            ctx = [(f"T{j}", ["sentence " * rng.randint(20, 60)]) for j in range(10)]
            tasks.append({"id": f"hot-{i}", "dataset": "hotpotqa", "question": "W" * rng.randint(60, 200),
                          "answer": f"entity {i}", "context": ctx, "support_titles": ["T0", "T1"],
                          "qtype_comparison": int(rng.random() < .2), "difficulty": d})
        g = [t for t in tasks if t["dataset"] == "gsm8k"]
        h = [t for t in tasks if t["dataset"] == "hotpotqa"]
        return g[:n_train] + h[:n_train], g[n_train:] + h[n_train:]

    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gi = rng.sample(range(len(gsm)), n_train + n_test)
    G = [{"id": f"gsm8k-{i}", "dataset": "gsm8k", "question": gsm[i]["question"],
          "answer": gsm[i]["answer"].split("####")[-1].strip().replace(",", "")} for i in gi]
    try:
        hp = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        rows = [{"_id": r["id"], "question": r["question"], "answer": r["answer"], "type": r["type"],
                 "context": list(zip(r["context"]["title"], r["context"]["sentences"])),
                 "support": list(dict.fromkeys(r["supporting_facts"]["title"]))} for r in hp]
    except Exception:
        import urllib.request
        url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
        raw = json.load(urllib.request.urlopen(url))
        rows = [{"_id": r["_id"], "question": r["question"], "answer": r["answer"], "type": r["type"],
                 "context": [(t, s) for t, s in r["context"]],
                 "support": list(dict.fromkeys(t for t, _ in r["supporting_facts"]))} for r in raw]
    hi = rng.sample(range(len(rows)), n_train + n_test)
    H = [{"id": f"hotpotqa-{rows[i]['_id']}", "dataset": "hotpotqa", "question": rows[i]["question"],
          "answer": rows[i]["answer"], "context": rows[i]["context"], "support_titles": rows[i]["support"],
          "qtype_comparison": int(rows[i]["type"] == "comparison")} for i in hi]
    return G[:n_train] + H[:n_train], G[n_train:] + H[n_train:]


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
class Sink:
    def __init__(self, path):
        self.path = path
        self.done = set()
        if os.path.exists(path):
            for line in open(path):
                r = json.loads(line)
                self.done.add(self.k(r))
        self.fh = open(path, "a")

    @staticmethod
    def k(r):
        return (r["task_id"], r["method"], str(r.get("lam")), r.get("split"), r.get("step"), r.get("cfg"))

    def write(self, r):
        if self.k(r) not in self.done:
            self.done.add(self.k(r))
            self.fh.write(json.dumps(r) + "\n")
            self.fh.flush()


def read_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


async def gather_progress(coros, label):
    t0, n, out = time.time(), len(coros), []
    for i, fut in enumerate(asyncio.as_completed(coros), 1):
        out.append(await fut)
        if i % max(1, n // 20) == 0 or i == n:
            print(f"  [{label}] {i}/{n}  {time.time() - t0:.0f}s", flush=True)
    return out


# ----------------------------------------------------------------------------------------------
# Phases
# ----------------------------------------------------------------------------------------------
async def phase_profile(llm, train, sink):
    ref_cfgs = {s: K.REF for s in K.STEP_TYPES}

    async def one(task):
        st_ref, res_ref, before = await K.run_fixed(llm, task, ref_cfgs)
        ref_sc = K.score(task, st_ref)
        steps = K.STEP_ORDER[task["dataset"]]
        jobs = []
        for i, step in enumerate(steps):
            chars = sum(len(m["content"]) for m in K.build_messages(task, step, before[i], 1024))
            for cfg in K.CONFIGS:
                async def run(i=i, step=step, cfg=cfg, chars=chars):
                    if cfg == K.REF:
                        r, sc = res_ref[i], ref_sc
                    else:
                        upd, r = await K.exec_step(llm, task, step, before[i], cfg)
                        st = dict(before[i]); st.update(upd)
                        st, _, _ = await K.run_fixed(llm, task, ref_cfgs, state=st, start=i + 1)
                        sc = K.score(task, st)
                    sink.write({"task_id": task["id"], "dataset": task["dataset"], "split": "train", "method": "profile",
                                "step": step, "cfg": K.cfg_name(cfg), "chars": chars, "n_in": r.n_in, "n_out": r.n_out,
                                "finish": r.finish, "ok": r.ok, "success": sc["success"], "ref_success": ref_sc["success"]})
                jobs.append(run())
        await asyncio.gather(*jobs)

    await gather_progress([one(t) for t in train], "profile")


async def phase_static(llm, tasks, split, sink):
    async def one(task, cfg):
        st, res, _ = await K.run_fixed(llm, task, {s: cfg for s in K.STEP_TYPES})
        sink.write(K.summarize(task, st, res, f"static:{K.cfg_name(cfg)}", None, split))
    await gather_progress([one(t, c) for t in tasks for c in K.CONFIGS], f"static-{split}")


def fit_models(train, prof_rows, static_rows):
    by_id = {t["id"]: t for t in train}
    ratio = {}
    for ds in ("gsm8k", "hotpotqa"):
        rr = [r for r in prof_rows if r["dataset"] == ds and r["cfg"] == K.cfg_name(K.REF)]
        ratio[ds] = sum(r["chars"] for r in rr) / max(1, sum(r["n_in"] for r in rr))
    X, y, groups = [], [], []
    nout_lists = defaultdict(list)
    cfg_of = {K.cfg_name(c): c for c in K.CONFIGS}
    for r in prof_rows:
        t = by_id[r["task_id"]]
        c = cfg_of[r["cfg"]]
        X.append(K.step_features(t, r["step"], r["chars"] / ratio[r["dataset"]], c))
        y.append(int(r["success"]))
        groups.append(r["task_id"])
        nout_lists[(r["dataset"], r["step"], c)].append(r["n_out"])
    nout = {k: float(np.median(v)) for k, v in nout_lists.items()}
    step_est = K.Estimator(K.step_features).fit(np.array(X), np.array(y))

    # grouped 5-fold CV quality of the step estimator (reported in the paper)
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.model_selection import GroupKFold
    Xa, ya, ga = np.array(X), np.array(y), np.array(groups)
    pr = np.zeros(len(ya))
    for tr, te in GroupKFold(n_splits=5).split(Xa, ya, ga):
        pr[te] = K.Estimator(K.step_features).fit(Xa[tr], ya[tr]).predict(Xa[te])
    metrics = {"step_auc": float(roc_auc_score(ya, pr)) if len(set(ya)) > 1 else None,
               "step_brier": float(brier_score_loss(ya, pr)), "n_rows": int(len(ya)), "ratio": ratio}

    TX, Ty = [], []
    for r in static_rows:
        t = by_id[r["task_id"]]
        c = cfg_of[r["method"].split(":")[1]]
        TX.append(K.task_features(t, c)); Ty.append(int(r["success"]))
    task_est = K.Estimator(K.task_features).fit(np.array(TX), np.array(Ty))
    task_cost = defaultdict(list)
    for r in static_rows:
        task_cost[(r["dataset"], r["method"].split(":")[1])].append(r["cost"])
    task_cost = {k: float(np.mean(v)) for k, v in task_cost.items()}
    return step_est, nout, ratio, task_est, task_cost, metrics


def phase_routers(tasks, split, static_rows, task_est, task_cost, c_ref, sink):
    look = {(r["task_id"], r["method"].split(":")[1]): r for r in static_rows}
    variants = {"route-model": [c for c in K.CONFIGS if c[1] == 1024],
                "route-budget": [c for c in K.CONFIGS if c[0] == "L"],
                "route-joint": list(K.CONFIGS)}
    for name, space in variants.items():
        for lam in LAMS:
            for t in tasks:
                p = task_est.predict([K.task_features(t, c) for c in space])
                s = [task_cost[(t["dataset"], K.cfg_name(c))] - lam * c_ref * pi for c, pi in zip(space, p)]
                c = space[int(np.argmin(s))]
                r = dict(look[(t["id"], K.cfg_name(c))])
                r.update(method=name, lam=lam, split=split, chosen=K.cfg_name(c))
                sink.write(r)


async def phase_katmo(llm, tasks, split, models, c_ref, lams, variants, sink, budgets=None):
    step_est, nout, ratio = models

    async def one(task, name, lam, B):
        rules, esc = {"katmo": (True, True), "katmo-norules": (False, False), "katmo-noesc": (True, False)}[name]
        pol = K.KATMOPolicy(step_est, nout, ratio, lam * c_ref, rules, esc, B)
        st, res = await K.run_policy(llm, task, pol)
        mname = name if B is None else f"{name}-B{budgets[task['dataset']]['frac']}"
        sink.write(K.summarize(task, st, res, mname, lam, split))

    jobs = []
    for name in variants:
        for lam in lams:
            for t in tasks:
                B = budgets[t["dataset"]]["B"] if budgets else None
                jobs.append(one(t, name, lam, B))
    await gather_progress(jobs, f"{'/'.join(variants)}-{split}")


def select_lambda(rows, method, target):
    """Cheapest lambda on TRAIN whose success >= target (pooled over datasets)."""
    best = None
    for lam in LAMS:
        rr = [r for r in rows if r["method"] == method and r["lam"] == lam and r["split"] == "train"]
        if not rr:
            continue
        s, c = np.mean([r["success"] for r in rr]), np.mean([r["cost"] for r in rr])
        if s >= target and (best is None or c < best[1]):
            best = (lam, c)
    return best[0] if best else LAMS[-1]


# ----------------------------------------------------------------------------------------------
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tolerance", type=float, default=0.01, help="allowed success drop vs reference on TRAIN")
    ap.add_argument("--out", default="results")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--concurrency", type=int, default=48)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    train, test = load_tasks(a.n_train, a.n_test, a.seed, a.mock)
    json.dump({"train": [t["id"] for t in train], "test": [t["id"] for t in test]}, open(f"{a.out}/split.json", "w"))
    llm = K.LLM(f"{a.out}/cache.jsonl", mock=a.mock, concurrency=a.concurrency)
    sink = Sink(f"{a.out}/runs.jsonl")

    print("Phase 1/6: interventional profiling (train)")
    await phase_profile(llm, train, sink)
    print("Phase 2/6: static configurations (train + test)")
    await phase_static(llm, train, "train", sink)
    await phase_static(llm, test, "test", sink)

    rows = read_jsonl(f"{a.out}/runs.jsonl")
    prof = [r for r in rows if r["method"] == "profile"]
    st_train = [r for r in rows if r["method"].startswith("static:") and r["split"] == "train"]
    st_test = [r for r in rows if r["method"].startswith("static:") and r["split"] == "test"]
    print("Phase 3/6: fitting estimators")
    step_est, nout, ratio, task_est, task_cost, metrics = fit_models(train, prof, st_train)
    ref_train = [r for r in st_train if r["method"] == f"static:{K.cfg_name(K.REF)}"]
    c_ref = float(np.mean([r["cost"] for r in ref_train]))
    target = float(np.mean([r["success"] for r in ref_train])) - a.tolerance
    print(f"  step-estimator CV AUC={metrics['step_auc']}, reference train success={target + a.tolerance:.3f}")

    print("Phase 4/6: query-level routers")
    phase_routers(train, "train", st_train, task_est, task_cost, c_ref, sink)
    phase_routers(test, "test", st_test, task_est, task_cost, c_ref, sink)

    print("Phase 5/6: KATMO + ablations")
    models = (step_est, nout, ratio)
    await phase_katmo(llm, train, "train", models, c_ref, LAMS, ["katmo"], sink)
    await phase_katmo(llm, test, "test", models, c_ref, LAMS, ["katmo", "katmo-norules", "katmo-noesc"], sink)

    rows = read_jsonl(f"{a.out}/runs.jsonl")
    lam_star = {m: select_lambda(rows, m, target) for m in ["katmo", "route-model", "route-budget", "route-joint"]}
    print(f"  lambda* (train) = {lam_star}")

    print("Phase 6/6: task-level budgets")
    for frac in (0.5, 0.75):
        budgets = {ds: {"frac": frac, "B": frac * float(np.mean([r["cost"] for r in ref_train if r["dataset"] == ds]))}
                   for ds in ("gsm8k", "hotpotqa")}
        await phase_katmo(llm, test, "test", models, c_ref, [lam_star["katmo"]], ["katmo"], sink, budgets)

    gpu = ""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    try:
        import vllm
        vv = vllm.__version__
    except Exception:
        vv = None
    meta = {"mock": a.mock, "n_train_per_dataset": a.n_train, "n_test_per_dataset": a.n_test, "seed": a.seed,
            "models": K.MODELS, "budgets": K.BUDGETS, "ref": K.cfg_name(K.REF), "w_in": K.W_IN, "lams": LAMS,
            "c_ref": c_ref, "target": target, "tolerance": a.tolerance, "lambda_star": lam_star,
            "estimator": metrics, "rules": {k: v.text for k, v in K.RULES.items()}, "gpu": gpu, "vllm": vv,
            "python": platform.python_version(), "llm_calls": llm.n_calls, "cache_hits": llm.n_hits,
            "unique_generations": len(llm.cache), "wall_seconds": time.time() - t0}
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=2, default=str)
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Unique generations: {len(llm.cache)}. Results in {a.out}/")


if __name__ == "__main__":
    asyncio.run(main())
