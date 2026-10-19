"""
KATMO: Knowledge-based Adaptive Token and Model Optimization for multi-step LLM agents.

This module contains the complete framework:
  * the configuration space (model tier x token budget) and cost model,
  * two multi-step agent pipelines (GSM8K math, HotpotQA multi-hop QA) with five step types,
  * the knowledge base of step-type rules (admissibility + escalation) with rule traces,
  * the data-driven step-value estimator,
  * the Lagrangian step-level allocator (KATMO) and the query-level baselines,
  * an async LLM client (vLLM OpenAI-compatible servers) with a persistent response cache.

A deterministic MOCK backend exists ONLY to test the code path without a GPU.
Mock outputs are flagged in every result file and must never be reported as findings.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import re
import string
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------------------------
# 1. Configuration space and cost model
# ----------------------------------------------------------------------------------------------
# tier -> (HF model id, nominal parameters in billions, local vLLM port)
MODELS = {
    "S": ("Qwen/Qwen2.5-1.5B-Instruct", 1.5, 8001),
    "M": ("Qwen/Qwen2.5-7B-Instruct", 7.0, 8002),
    "L": ("Qwen/Qwen2.5-32B-Instruct-AWQ", 32.0, 8003),
}
TIERS = ["S", "M", "L"]
BUDGETS = [128, 256, 512, 1024]
CONFIGS = [(m, b) for m in TIERS for b in BUDGETS]
REF = ("L", 1024)  # strongest static configuration = reference
W_IN = 0.25        # relative price of an input (prefill) token vs an output token


def cfg_name(cfg) -> str:
    return f"{cfg[0]}@{cfg[1]}"


def step_cost(tier: str, n_in: int, n_out: int, w_in: float = W_IN) -> float:
    """Compute-proportional cost: parameters (B) x weighted tokens / 1000."""
    return MODELS[tier][1] * (w_in * n_in + n_out) / 1000.0


# ----------------------------------------------------------------------------------------------
# 2. Agent pipelines: prompts, parsers, scorers
# ----------------------------------------------------------------------------------------------
STEP_ORDER = {
    "gsm8k": ["plan", "reason", "tool", "verify", "answer"],
    "hotpotqa": ["plan", "tool", "reason", "verify", "answer"],
}
STEP_TYPES = ["plan", "reason", "tool", "verify", "answer"]
SYSTEM = "You are a precise assistant operating one step of a multi-step agent. Keep your response under {b} tokens."

_NUM = r"-?\$?\d[\d,]*\.?\d*"


def _last_tag(text: str, tag: str):
    hits = re.findall(rf"{tag}\s*[:：]\s*(.+)", text, flags=re.I)
    return hits[-1].strip() if hits else None


def _to_num(s):
    if s is None:
        return None
    m = re.findall(_NUM, str(s))
    if not m:
        return None
    try:
        return float(m[-1].replace(",", "").replace("$", ""))
    except ValueError:
        return None


def _clip(s, n=1800):
    s = s or ""
    return s if len(s) <= n else s[:n] + " ..."


def run_python(code: str, timeout: float = 5.0):
    """Execute model-written code in an isolated interpreter and read `answer`."""
    wrapper = (
        "import resource,sys\n"
        "try:\n resource.setrlimit(resource.RLIMIT_AS,(1<<30,1<<30))\nexcept Exception: pass\n"
        f"{code}\n"
        "print('__ANS__', answer)\n"
    )
    try:
        r = subprocess.run([sys.executable, "-I", "-c", wrapper], capture_output=True, text=True, timeout=timeout)
        m = re.search(r"__ANS__ (.+)", r.stdout)
        return _to_num(m.group(1)) if m else None
    except Exception:
        return None


def _paras(task, idx=None):
    out = []
    for i, (t, sents) in enumerate(task["context"], 1):
        if idx is None or i in idx:
            out.append(f"[{i}] {t}: {''.join(sents)}")
    return "\n".join(out)


def build_messages(task: dict, step: str, st: dict, budget: int):
    q, ds = task["question"], task["dataset"]
    if ds == "gsm8k":
        if step == "plan":
            u = f"Problem:\n{q}\n\nWrite a short numbered plan of the calculations needed. Do not compute the final answer."
        elif step == "reason":
            u = f"Problem:\n{q}\n\nPlan:\n{_clip(st.get('plan'))}\n\nSolve the problem step by step following the plan. End with a line 'Answer: <number>'."
        elif step == "tool":
            u = (f"Problem:\n{q}\n\nPlan:\n{_clip(st.get('plan'))}\n\nWrite a short Python program that computes the final "
                 "answer and stores it in a variable named `answer`. Output only one ```python code block.")
        elif step == "verify":
            u = (f"Problem:\n{q}\n\nCandidate A (step-by-step reasoning): {st.get('reason_ans')}\nReasoning:\n{_clip(st.get('reason'))}\n\n"
                 f"Candidate B (Python program result): {st.get('tool_ans')}\n\nCheck the candidates against the problem and "
                 "decide the correct final answer. End with a line 'Final: <number>'.")
        else:
            u = (f"Problem:\n{q}\n\nVerification notes:\n{_clip(st.get('verify'))}\n\n"
                 "Reply with only the final numeric answer (digits only, no units or words).")
    else:
        if step == "plan":
            u = f"Question: {q}\n\nList the sub-questions (at most 3) that must be answered to answer this question. Do not answer them."
        elif step == "tool":
            u = (f"Question: {q}\n\nSub-questions:\n{_clip(st.get('plan'), 800)}\n\nParagraphs:\n{_paras(task)}\n\n"
                 "Which paragraphs are needed to answer the question? Reply with only a JSON list of at most 3 paragraph numbers, e.g. [2, 7].")
        elif step == "reason":
            u = (f"Question: {q}\n\nSub-questions:\n{_clip(st.get('plan'), 800)}\n\nEvidence:\n{st.get('evidence', '')}\n\n"
                 "Answer the question step by step using only the evidence. End with a line 'Answer: <short answer>'.")
        elif step == "verify":
            u = (f"Question: {q}\n\nEvidence:\n{st.get('evidence', '')}\n\nProposed answer: {st.get('reason_ans')}\n"
                 f"Reasoning:\n{_clip(st.get('reason'), 1500)}\n\nCheck whether the proposed answer is correct and supported by the "
                 "evidence; correct it if needed. End with a line 'Final: <short answer>'.")
        else:
            u = (f"Question: {q}\n\nVerification notes:\n{_clip(st.get('verify'))}\n\nReply with only the final short answer "
                 "(a name, number, date, or yes/no) and no explanation.")
    return [{"role": "system", "content": SYSTEM.format(b=budget)}, {"role": "user", "content": u}]


def parse_step(task: dict, step: str, text: str, finish: str):
    """Return (state_update, structurally_ok). Structural checks feed the escalation rule."""
    ds, text = task["dataset"], text or ""
    upd = {step: text}
    if step == "plan":
        ok = len(text.strip()) > 0 and finish != "length"
    elif step == "reason":
        a = _last_tag(text, "Answer")
        upd["reason_ans"] = (_to_num(a) if ds == "gsm8k" else a)
        ok = upd["reason_ans"] is not None
    elif step == "tool":
        if ds == "gsm8k":
            m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
            code = m.group(1) if m else ""
            upd["tool_ans"] = run_python(code) if code else None
            ok = upd["tool_ans"] is not None
        else:
            nums = [int(x) for x in re.findall(r"\d+", text.split("]")[0] if "[" in text else text)]
            idx = [i for i in dict.fromkeys(nums) if 1 <= i <= len(task["context"])][:3]
            upd["evidence"] = _paras(task, set(idx)) if idx else ""
            gold_titles = set(task.get("support_titles", []))
            sel = {task["context"][i - 1][0] for i in idx}
            upd["retrieval_recall"] = (len(sel & gold_titles) / len(gold_titles)) if gold_titles else None
            ok = len(idx) > 0
    elif step == "verify":
        f = _last_tag(text, "Final")
        upd["verify_ans"] = (_to_num(f) if ds == "gsm8k" else f)
        ok = upd["verify_ans"] is not None
    else:
        a = text.strip().splitlines()[0].strip() if text.strip() else ""
        upd["answer_ans"] = (_to_num(a) if ds == "gsm8k" else a)
        ok = upd["answer_ans"] not in (None, "") and (ds == "gsm8k" or len(a.split()) <= 20)
    return upd, bool(ok)


def _norm(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc)


def score(task: dict, st: dict):
    """Final prediction = ANSWER step output (the agent's committed answer)."""
    pred = st.get("answer_ans")
    if task["dataset"] == "gsm8k":
        g = float(task["answer"])
        ok = pred is not None and abs(pred - g) <= 1e-4 * max(1.0, abs(g))
        return {"pred": pred, "success": bool(ok), "em": float(ok), "f1": float(ok)}
    pred = pred or ""
    em = float(_norm(pred) == _norm(task["answer"]))
    f1 = f1_score(pred, task["answer"])
    return {"pred": pred, "success": bool(em == 1.0 or f1 >= 0.8), "em": em, "f1": f1}


# ----------------------------------------------------------------------------------------------
# 3. Knowledge base: step-type rules (fixed a priori, before any experiment was run)
# ----------------------------------------------------------------------------------------------
@dataclass
class Rule:
    rid: str
    text: str


RULES = {
    "R1": Rule("R1", "ANSWER steps only format a committed answer: token budget <= 256."),
    "R2": Rule("R2", "PLAN steps are outlines: token budget <= 512."),
    "R3": Rule("R3", "Program-writing TOOL steps need room for code: token budget >= 256."),
    "R4": Rule("R4", "Long-context steps (estimated input > 1500 tokens) exclude the smallest tier S."),
    "R5": Rule("R5", "A VERIFY step must not use a weaker tier than the REASON step it checks."),
    "R6": Rule("R6", "Escalate once on structural failure: truncated -> double budget, otherwise next tier."),
}
TIER_RANK = {t: i for i, t in enumerate(TIERS)}


def admissible(task, step, st, est_in, used_tiers):
    """Return (admissible configs, list of fired rule ids)."""
    A, fired = list(CONFIGS), []

    def keep(pred, rid):
        nonlocal A
        new = [c for c in A if pred(c)]
        if len(new) < len(A) and new:
            A = new
            fired.append(rid)

    if step == "answer":
        keep(lambda c: c[1] <= 256, "R1")
    if step == "plan":
        keep(lambda c: c[1] <= 512, "R2")
    if step == "tool" and task["dataset"] == "gsm8k":
        keep(lambda c: c[1] >= 256, "R3")
    if est_in > 1500:
        keep(lambda c: c[0] != "S", "R4")
    if step == "verify" and "reason" in used_tiers:
        r = TIER_RANK[used_tiers["reason"]]
        keep(lambda c: TIER_RANK[c[0]] >= r, "R5")
    return A, fired


def escalate(cfg, truncated, allowed):
    """R6: one escalation step inside the admissible set; None if already maximal."""
    t, b = cfg
    cands = []
    if truncated and b * 2 in BUDGETS:
        cands.append((t, b * 2))
    if TIER_RANK[t] + 1 < len(TIERS):
        cands.append((TIERS[TIER_RANK[t] + 1], b))
    if not truncated and b * 2 in BUDGETS:
        cands.append((t, b * 2))
    for c in cands:
        if c in allowed:
            return c
    return None


# ----------------------------------------------------------------------------------------------
# 4. LLM client with persistent cache (vLLM OpenAI-compatible servers, or MOCK)
# ----------------------------------------------------------------------------------------------
class LLM:
    def __init__(self, cache_path: str, mock: bool = False, concurrency: int = 48, host: str = "http://127.0.0.1"):
        self.mock, self.host, self.cache_path = mock, host, cache_path
        self.cache, self.inflight = {}, {}
        self.sem = {t: asyncio.Semaphore(concurrency) for t in TIERS}
        self.n_calls = self.n_hits = 0
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.cache[r["key"]] = r["val"]
                    except Exception:
                        pass
        self._fh = open(cache_path, "a")
        self._client = None

    @staticmethod
    def key(tier, budget, messages):
        return hashlib.sha1(json.dumps([MODELS[tier][0], budget, messages], sort_keys=True).encode()).hexdigest()

    async def call(self, tier, budget, messages, mock_ctx=None):
        k = self.key(tier, budget, messages)
        self.n_calls += 1
        if k in self.cache:
            self.n_hits += 1
            return dict(self.cache[k], cached=True)
        if k in self.inflight:
            return dict(await self.inflight[k], cached=True)
        fut = asyncio.get_running_loop().create_future()
        self.inflight[k] = fut
        try:
            val = self._mock(tier, budget, messages, mock_ctx) if self.mock else await self._http(tier, budget, messages)
            self.cache[k] = val
            self._fh.write(json.dumps({"key": k, "val": val}) + "\n")
            self._fh.flush()
            fut.set_result(val)
        except Exception as e:
            fut.set_exception(e)
            raise
        finally:
            self.inflight.pop(k, None)
        return dict(val, cached=False)

    async def _http(self, tier, budget, messages):
        import httpx
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=600)
        name, _, port = MODELS[tier]
        body = {"model": name, "messages": messages, "max_tokens": budget, "temperature": 0.0, "seed": 0}
        async with self.sem[tier]:
            for attempt in range(6):
                try:
                    t0 = time.time()
                    r = await self._client.post(f"{self.host}:{port}/v1/chat/completions", json=body)
                    r.raise_for_status()
                    j = r.json()
                    ch = j["choices"][0]
                    return {"text": ch["message"]["content"] or "", "finish": ch.get("finish_reason") or "stop",
                            "n_in": j["usage"]["prompt_tokens"], "n_out": j["usage"]["completion_tokens"],
                            "latency": time.time() - t0}
                except Exception:
                    if attempt == 5:
                        raise
                    await asyncio.sleep(2 ** attempt)

    # --- MOCK: deterministic, flagged synthetic behaviour for code testing only -----------------
    def _mock(self, tier, budget, messages, ctx):
        task, step = ctx
        rnd = random.Random(self.key(tier, budget, messages))
        size = {"S": 0, "M": 1, "L": 2}[tier]
        need = {"plan": 140, "reason": 380, "tool": 170, "verify": 260, "answer": 12}[step] * (1.15 - 0.15 * size)
        need = int(need * rnd.uniform(0.6, 1.4)) + 4
        trunc = need > budget
        hard = task.get("difficulty", 0.5)
        p = {"plan": .95, "reason": .55 + .17 * size, "tool": .6 + .15 * size, "verify": .6 + .15 * size, "answer": .9 + .04 * size}[step]
        p = max(0.02, p - 0.35 * hard * (2 - size) / 2)
        g = task["answer"]
        gold_seen = re.findall(r"(?:Answer|Final|result\)?)[^\n]*?:\s*([^\n]+)", messages[1]["content"])
        prev = gold_seen[-1].strip() if gold_seen else None
        right = rnd.random() < p
        if task["dataset"] == "gsm8k":
            val = g if right else (prev if prev and rnd.random() < 0.5 else str(float(g) + rnd.randint(1, 9)))
        else:
            val = g if right else (prev if prev and rnd.random() < 0.5 else "unknown entity")
        body = {"plan": "1. identify quantities\n2. combine them", "reason": f"Working...\nAnswer: {val}",
                "tool": f"```python\nanswer = {val if task['dataset'] == 'gsm8k' else 0}\n```", "verify": f"Checked.\nFinal: {val}",
                "answer": f"{val}"}[step]
        if task["dataset"] == "hotpotqa" and step == "tool":
            body = "[1, 2]" if right else "[5]"
        if trunc:
            body = body[: max(1, len(body) // 3)] if step != "answer" else body
        n_in = sum(len(m["content"]) for m in messages) // 4
        return {"text": body, "finish": "length" if trunc else "stop", "n_in": n_in,
                "n_out": min(need, budget), "latency": 0.001 * min(need, budget) * (1 + size), "mock": True}


# ----------------------------------------------------------------------------------------------
# 5. Features and the data-driven step-value estimator
# ----------------------------------------------------------------------------------------------
def est_tokens(messages, ratio):
    return int(sum(len(m["content"]) for m in messages) / ratio)


def step_features(task, step, est_in, cfg):
    t, b = cfg
    return [1.0 if task["dataset"] == "gsm8k" else 0.0, *[1.0 if step == s else 0.0 for s in STEP_TYPES],
            float(TIER_RANK[t]), math.log2(b), math.log1p(est_in), math.log1p(len(task["question"])),
            float(task.get("qtype_comparison", 0))]


def task_features(task, cfg):
    t, b = cfg
    return [1.0 if task["dataset"] == "gsm8k" else 0.0, float(TIER_RANK[t]), math.log2(b),
            math.log1p(len(task["question"])), float(task.get("qtype_comparison", 0)),
            math.log1p(sum(len("".join(s)) for _, s in task.get("context", [])))]


class Estimator:
    """P(task success | configuration) learned with gradient-boosted trees, plus output-token tables."""

    def __init__(self, featfn):
        self.featfn, self.clf, self.nout = featfn, None, {}

    def fit(self, X, y, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                  l2_regularization=1.0, random_state=seed)
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        import numpy as np
        return self.clf.predict_proba(np.asarray(X))[:, 1]


# ----------------------------------------------------------------------------------------------
# 6. Agent execution and policies
# ----------------------------------------------------------------------------------------------
@dataclass
class StepResult:
    step: str
    cfg: tuple
    n_in: int
    n_out: int
    cost: float
    finish: str
    ok: bool
    latency: float
    escalated_from: tuple | None = None
    rules: list = field(default_factory=list)


async def exec_step(llm, task, step, st, cfg):
    msgs = build_messages(task, step, st, cfg[1])
    r = await llm.call(cfg[0], cfg[1], msgs, mock_ctx=(task, step))
    upd, ok = parse_step(task, step, r["text"], r["finish"])
    res = StepResult(step, cfg, r["n_in"], r["n_out"], step_cost(cfg[0], r["n_in"], r["n_out"]), r["finish"], ok, r["latency"])
    return upd, res


async def run_fixed(llm, task, cfgs: dict, state=None, start=0):
    """Run steps[start:] with a fixed config per step type; returns (state, [StepResult], states_before)."""
    st = dict(state or {})
    res, before = [], []
    for step in STEP_ORDER[task["dataset"]][start:]:
        before.append(dict(st))
        upd, r = await exec_step(llm, task, step, st, cfgs[step])
        st.update(upd)
        res.append(r)
    return st, res, before


class KATMOPolicy:
    """Step-level allocator: argmin_a in A_k  c_hat(a) - lambda * p_hat(a), with KB rules and escalation."""

    def __init__(self, est: Estimator, nout: dict, ratio: dict, lam: float, use_rules=True, use_escalation=True,
                 task_budget=None):
        self.est, self.nout, self.ratio, self.lam = est, nout, ratio, lam
        self.use_rules, self.use_esc, self.B = use_rules, use_escalation, task_budget

    def choose(self, task, step, st, used, spent, remaining_steps):
        ds = task["dataset"]
        # estimate input tokens from the prompt actually built for this state
        est_in = est_tokens(build_messages(task, step, st, 1024), self.ratio[ds])
        A, fired = admissible(task, step, st, est_in, used) if self.use_rules else (list(CONFIGS), [])
        p = self.est.predict([step_features(task, step, est_in, c) for c in A])
        chat = [step_cost(c[0], est_in, min(c[1], self.nout.get((ds, step, c), c[1]))) for c in A]
        scores = [ch - self.lam * pi for ch, pi in zip(chat, p)]
        order = sorted(range(len(A)), key=lambda i: (scores[i], chat[i]))
        pick = A[order[0]]
        if self.B is not None:  # task-level budget: keep enough for the cheapest completion of the remaining steps
            floor = sum(step_cost("S", est_in, 64) for _ in range(remaining_steps))
            for i in order:
                if spent + chat[i] + floor <= self.B:
                    pick = A[i]
                    break
            else:
                pick = A[min(range(len(A)), key=lambda i: chat[i])]
                fired = fired + ["BUDGET"]
        return pick, A, fired


async def run_policy(llm, task, pol: KATMOPolicy):
    st, res, used, spent = {}, [], {}, 0.0
    steps = STEP_ORDER[task["dataset"]]
    for i, step in enumerate(steps):
        cfg, A, fired = pol.choose(task, step, st, used, spent, len(steps) - i - 1)
        upd, r = await exec_step(llm, task, step, st, cfg)
        r.rules = list(fired)
        if pol.use_esc and (not r.ok):
            nxt = escalate(cfg, r.finish == "length", A if pol.use_rules else CONFIGS)
            if nxt is not None:
                upd2, r2 = await exec_step(llm, task, step, st, nxt)
                r2.cost += r.cost
                r2.n_in += r.n_in
                r2.n_out += r.n_out
                r2.latency += r.latency
                r2.escalated_from, r2.rules = cfg, r.rules + ["R6"]
                upd, r = upd2, r2
        st.update(upd)
        used[step] = r.cfg[0]
        spent += r.cost
        res.append(r)
    return st, res


def summarize(task, st, res, method, lam=None, split="test"):
    sc = score(task, st)
    return {"task_id": task["id"], "dataset": task["dataset"], "split": split, "method": method, "lam": lam,
            **sc, "cost": sum(r.cost for r in res), "n_in": sum(r.n_in for r in res), "n_out": sum(r.n_out for r in res),
            "latency": sum(r.latency for r in res),
            "steps": [{"step": r.step, "cfg": cfg_name(r.cfg), "cost": r.cost, "n_in": r.n_in, "n_out": r.n_out,
                       "finish": r.finish, "ok": r.ok, "esc_from": cfg_name(r.escalated_from) if r.escalated_from else None,
                       "rules": r.rules} for r in res],
            "retrieval_recall": st.get("retrieval_recall")}
