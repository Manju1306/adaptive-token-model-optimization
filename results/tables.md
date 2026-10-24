### Table 1. Main results on the held-out test split

| Method | GSM8K success (%) | GSM8K cost (%) | HotpotQA success (%) | HotpotQA cost (%) | Overall success (%) [95% CI] | Overall cost (% of ref.) | Tokens/task | Iso-success cost (%) | p (success vs ref.) | p (cost vs ref.) |
|---|---|---|---|---|---|---|---|---|---|---|
| Static S@1024 | 68.7 | 3.4 | 28.0 | 4.2 | 48.3 [42.7, 54.0] | 3.9 | 2173 | - | <0.001 | <0.001 |
| Static M@1024 | 91.3 | 22.4 | 54.0 | 21.1 | 72.7 [67.7, 78.0] | 21.7 | 2352 | - | <0.001 | <0.001 |
| Static L@512 | 94.7 | 97.1 | 66.7 | 98.7 | 80.7 [76.3, 85.0] | 98.0 | 2330 | - | 1.000 | <0.001 |
| Static L@1024 (reference) | 96.0 | 100.0 | 65.3 | 100.0 | 80.7 [76.3, 85.0] | 100.0 | 2363 | - | - | - |
| Model routing only | 96.0 | 100.0 | 65.3 | 87.7 | 80.7 [76.0, 85.0] | 93.2 | 2359 | 93.2 | 1.000 | <0.001 |
| Token budgeting only | 96.0 | 99.1 | 65.3 | 97.6 | 80.7 [76.3, 85.0] | 98.3 | 2337 | 83.7 | 1.000 | <0.001 |
| Query-level joint (model+budget) | 96.0 | 99.1 | 64.7 | 86.0 | 80.3 [75.7, 84.7] | 91.9 | 2333 | 91.9 | 1.000 | <0.001 |
| KATMO (step-level, this work) | 94.0 | 94.5 | 66.0 | 80.0 | 80.0 [75.7, 84.3] | 86.5 | 2298 | 86.5 | 0.804 | <0.001 |

### Table 2. Ablation and sensitivity (test split)

| Variant | Success (%) [95% CI] | Cost (% of ref.) | Output tokens/task | Escalated steps (%) | p (success vs ref.) | Note |
|---|---|---|---|---|---|---|
| KATMO (full) | 80.0 [75.7, 84.3] | 86.5 | 386 | 2.2 | 0.804 | lambda*=1e+06 x C_ref (selected on train) |
| - without knowledge rules (R1-R6) | 80.0 [75.7, 84.3] | 84.4 | 381 | 0.0 | 0.791 | ablation |
| - without escalation (R6) | 81.3 [77.0, 85.7] | 83.7 | 367 | 0.0 | 0.791 | ablation |
| Rules + cheapest admissible (lambda=0) | 45.0 [39.7, 50.7] | 7.5 | 228 | 4.3 | <0.001 | no estimator trade-off |
| KATMO, task budget = 75% of ref. cost | 73.0 [68.0, 78.0] | 74.4 | 374 | 2.6 | <0.001 | budget sensitivity |
| KATMO, task budget = 50% of ref. cost | 68.0 [63.0, 73.0] | 56.1 | 356 | 2.8 | <0.001 | budget sensitivity |
| KATMO, input-token weight w_in=0.1 | 80.0 [75.7, 84.3] | 86.1 | 386 | 2.2 | 0.804 | price sensitivity |
| KATMO, input-token weight w_in=0.5 | 80.0 [75.7, 84.3] | 86.8 | 386 | 2.2 | 0.804 | price sensitivity |
| KATMO, input-token weight w_in=1.0 | 80.0 [75.7, 84.3] | 87.1 | 386 | 2.2 | 0.804 | price sensitivity |

Step-value estimator: grouped 5-fold CV AUC = 0.7277477856151459, Brier = 0.19427691494423743, rows = 18000.
