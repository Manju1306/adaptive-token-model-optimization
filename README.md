# KATMO — Knowledge-based Adaptive Token and Model Optimization for Agentic AI

Code and data for the Short Communication
**"Cost-Efficient Agentic AI: A Knowledge-Based Decision Intelligence Framework for Adaptive Token and Model Optimization"**
(M. L. Cherukuri, submitted to *Knowledge-Based Systems*).

LLM agents solve tasks in steps (plan → reason → tool → verify → answer), but most deployments run every step with the
same large model and the same token limit. KATMO decides **per agent step** which **model tier** (1.5B / 7B / 32B) and
which **output-token budget** (128 / 256 / 512 / 1024) to use, combining

1. a **knowledge base of step-type rules** (admissible configurations + one-shot escalation, each decision logged with a rule trace),
2. a **data-driven step-value estimator** trained on interventional step profiles, and
3. a **Lagrangian allocator** `argmin_a ĉ(a) − λ·p̂(a)` whose multiplier λ is the price of one unit of task-success probability.

## Repository

| File | Purpose |
|---|---|
| `katmo.py` | Framework: configuration space, cost model, agent pipelines, knowledge rules, estimator, allocator, cached LLM client |
| `experiment.py` | Generates the dataset on real LLM executions and evaluates all methods (resumable) |
| `analyze.py` | Builds Table 1, Table 2, Figure 1 (cost–success frontiers), Figure 2 (step-level allocation) |
| `runpod.sh` | One-command run on a single 80 GB GPU (starts three vLLM servers, runs everything, packs results) |

## Knowledge base (fixed before any experiment)

| Rule | Content |
|---|---|
| R1 | ANSWER steps only format a committed answer: token budget ≤ 256 |
| R2 | PLAN steps are outlines: token budget ≤ 512 |
| R3 | Program-writing TOOL steps need room for code: budget ≥ 256 |
| R4 | Long-context steps (estimated input > 1500 tokens) exclude the 1.5B tier |
| R5 | A VERIFY step may not use a weaker tier than the REASON step it checks |
| R6 | Escalate once on structural failure: truncated → double budget, otherwise next tier |

## Experimental protocol

* **Tasks:** GSM8K test (math; tool step = Python execution) and HotpotQA distractor dev (multi-hop QA; tool step = paragraph retrieval). 150 train + 150 test tasks per dataset, seed 2026.
* **Models:** Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-32B-Instruct-AWQ served with vLLM, greedy decoding.
* **Cost:** compute-proportional, `params(B) × (w_in·input_tokens + output_tokens)/1000`, `w_in = 0.25` (sensitivity 0.1–1.0 reported).
* **Interventional profiling (train):** for each step and each of the 12 configurations, that single step is re-executed and the trajectory is completed with the reference configuration (32B@1024); the outcome labels the step-value estimator.
* **Baselines:** 12 static configurations; query-level routers choosing model only, budget only, or model+budget jointly per task.
* **Ablations:** without knowledge rules, without escalation, λ = 0, task-level budgets (50%, 75%).
* **Operating point:** λ* is selected on the train split (cheapest λ whose success is within 1 point of the reference); all reported numbers are on the held-out test split with bootstrap CIs, exact McNemar tests (success) and Wilcoxon signed-rank tests (cost).

## Reproduce

**GPU run (RunPod, 1× H100/A100 80 GB, ~$10–20):**
```bash
git clone https://github.com/Manju1306/adaptive-token-model-optimization.git
cd adaptive-token-model-optimization
bash runpod.sh smoke     # ~10 min sanity check
bash runpod.sh full      # paper run; resumable if interrupted
```
**Tables and figures only (CPU, from released results):**
```bash
pip install -r requirements.txt
python analyze.py --res results
```
**Code test without GPU** (synthetic tasks and deterministic MOCK outputs; every artifact is watermarked and must not be reported):
```bash
python experiment.py --mock --n-train 10 --n-test 10 --out results_mock && python analyze.py --res results_mock
```

## Data and licences

GSM8K (MIT), HotpotQA (CC BY-SA 4.0), Qwen2.5 1.5B/7B/32B (Apache-2.0). The generated step-level dataset
(`results/runs.jsonl`, and the full response cache released as an archive) is distributed under CC BY 4.0.
Model-written Python is executed in an isolated interpreter with a memory limit and timeout.

## Citation

Cherukuri, M. L. (2026). Cost-Efficient Agentic AI: A Knowledge-Based Decision Intelligence Framework for Adaptive
Token and Model Optimization. *Manuscript submitted to Knowledge-Based Systems.*
