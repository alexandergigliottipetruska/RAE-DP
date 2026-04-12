# Optuna HP Sweep — Experiment Log

## Task: Square (NutAssemblySquare), bf16 tokens, RAE-DP

---

## Round 1: `v3_square_bf16_hp_sweep`

**Date**: 2026-04-10 to 2026-04-11

**Setup**:
- 15-17 lab PCs (dh2026pc04-pc20), RTX 4080 16GB each
- 300 epochs per trial, cosine LR schedule
- 50-episode eval every 10 epochs (30 eval points per trial)
- Objective: peak success rate (maximize)
- bf16 pre-computed tokens, cache_in_ram
- PostgreSQL Optuna DB on dh2026pc02

**Trials**: 62 completed, 6 pruned, 22 orphaned (from restart)

### Search Space

| Parameter | Range | Type |
|-----------|-------|------|
| lr | [5e-5, 5e-4] | log-uniform |
| lr_adapter | [1e-6, 1e-4] | log-uniform |
| d_model | {256, 384, 512} | categorical |
| n_layers | {6, 8, 10, 12} | discrete |
| n_cond_layers | {2, 4, 6} | discrete |
| p_drop_attn | [0.0, 0.3] | continuous |

### Fixed Settings

| Parameter | Value |
|-----------|-------|
| spatial_pool_size | 7 |
| use_flow_matching | True |
| batch_size | 64 |
| warmup_steps | 1000 |
| T_obs | 2 |
| T_pred | 10 |
| norm_mode | chi |
| no_amp | True (fp32 compute) |

### Top 5 Results (by weighted = 0.5*peak + 0.5*last10_avg)

| Trial | Peak | Last10 | Top5 | Weighted | lr | lr_adapter | d | nl | nc | pdrop |
|-------|------|--------|------|----------|-----|------------|---|----|----|-------|
| #61 | 100% | 95.8% | 98.4% | **0.979** | 1.7e-4 | 3.0e-5 | 384 | 8 | 4 | 0.130 |
| #57 | 100% | 95.4% | 98.8% | 0.977 | 2.8e-4 | 2.8e-5 | 384 | 6 | 4 | 0.132 |
| #52 | 100% | 94.6% | 98.4% | 0.973 | 3.2e-4 | 7.9e-6 | 384 | 6 | 4 | 0.128 |
| #30 | 100% | 94.2% | 98.4% | 0.971 | 2.4e-4 | 1.6e-5 | 384 | 6 | 6 | 0.213 |
| #58 | 100% | 94.0% | 98.4% | 0.970 | 2.9e-4 | 2.7e-5 | 384 | 6 | 4 | 0.048 |

### Key Findings

**Hyperparameter importance** (fANOVA, dashboard):
- lr: 0.36
- lr_adapter: 0.29
- n_layers: 0.17
- p_drop_attn: 0.07
- d_model: 0.06
- n_cond_layers: 0.06

**Architecture**:
- **d_model=384 dominates**: all top 19 trials (100% peak) use d=384. TPE sampled 46/62 trials as d=384.
- **n_layers=6 is optimal**: mean weighted 0.912 (n=34) vs nl=8 at 0.899 (n=14). Wider + shallower beats narrower + deeper.
- **n_cond_layers=4 dominates top-10**: 8/10 use nc=4. But nc barely matters (importance=0.06).

**Training recipe**:
- **lr sweet spot: 1.7e-4 to 3.2e-4**. Higher lr converges faster (r=+0.24).
- **lr_adapter very low (8e-6 to 3e-5)**: the RAE pre-trained adapter should barely be fine-tuned. Supports the hypothesis that RAE pre-training finds the right representation.
- **p_drop_attn ~0.13**: moderate dropout. Very low (<0.03) clearly hurts. Higher (~0.21) also works.

**Diminishing returns**:
- 73% of trials peak by epoch 149, 87% by epoch 199
- Mean SR declines by 0.01 from epoch 149 to 299
- Rankings at epoch 100 have r=0.67 with final rankings; epoch 150 has r=0.80

**Eval noise**: 50-episode evals have ~4% std. Peak SR is a biased estimator (selects for noise). Weighted metric (0.5*peak + 0.5*last10_avg) is the most robust metric (avg Spearman rho=0.834 with all other metrics).

**Toxic combinations**:
- d=256 + low lr (<1.3e-4) → consistently bad
- d=512 + high lr (>3e-4) → training collapse
- d=384 + very low dropout (<0.03) → underperforms

---

## Round 2: `v3_square_bf16_hp_sweep_r2`

**Date**: 2026-04-11 onwards

**Changes from Round 1**:
- Objective: **weighted metric** (0.5*peak + 0.5*last10_avg) instead of peak SR
- 200 epochs (was 300): captures 87% of peaks, saves 33% compute
- 100-episode evals (was 50): less noisy evaluations
- 15 eval envs (was 10)
- Narrowed search space based on R1 findings
- Added T_obs as new search parameter
- Added n_layers=4 (unexplored in R1)
- 3 trials per worker (was 5): 17 PCs × 3 = 51 trials
- New study name to keep R1 data separate

### Search Space

| Parameter | Range | Type | Rationale |
|-----------|-------|------|-----------|
| lr | [1.8e-4, 3.6e-4] | log-uniform | Narrowed to R1 sweet spot |
| lr_adapter | [8e-6, 2.8e-5] | log-uniform | Narrowed to R1 sweet spot |
| d_model | {384} | fixed | All R1 top trials use 384 |
| n_layers | {4, 6, 8} | categorical | Added 4 (unexplored), dropped 10/12 |
| n_cond_layers | 4 | fixed | Dominated R1 top-10 |
| p_drop_attn | [0.05, 0.22] | continuous | Very low (<0.03) hurts, narrowed |
| T_obs | {1, 2, 3} | categorical | **New**: does temporal context matter? |

### Fixed Settings (same as R1 except noted)

| Parameter | Value | Change from R1 |
|-----------|-------|----------------|
| d_model | 384 | was searched |
| n_cond_layers | 4 | was searched |
| num_epochs | 200 | was 300 |
| eval_full_episodes | 100 | was 50 |
| eval_n_envs | 15 | was 10 |

### Questions this round aims to answer
1. Does T_obs=1 (no temporal context) work as well as T_obs=2?
2. Does T_obs=3 (acceleration info) help for the harder Square task?
3. Is n_layers=4 (even shallower) better than 6?
4. With less noisy evals (100 episodes), do the same configs still win?
5. Does the weighted metric change which configs TPE prefers?

### Results

**Final**: 87 completed, 30 pruned (R2 stopped after 135 total trials when TPE plateaued)

#### Top 5 (old metric: 0.5*peak + 0.5*last10_avg)

| Rank | Trial | weighted | peak | last10 | T_obs | nl | lr | lr_adapter | pdrop |
|------|-------|----------|------|--------|-------|----|----|------------|-------|
| 1 | **#100** | **0.9725** | 0.99 | 0.955 | 2 | 6 | 3.25e-4 | 2.27e-5 | 0.188 |
| 2 | #61 | 0.9700 | 0.98 | 0.960 | 2 | 6 | 3.50e-4 | 2.34e-5 | 0.159 |
| 3 | #104 | 0.9700 | 0.98 | 0.960 | 2 | 6 | 2.94e-4 | 2.30e-5 | 0.211 |
| 4 | #26 | 0.9675 | 0.99 | 0.945 | 2 | **4** | 3.37e-4 | 8.63e-6 | 0.181 |
| 5 | #105 | 0.9670 | 0.99 | 0.944 | 2 | 6 | 2.72e-4 | 8.75e-6 | 0.175 |

**R2 champion**: trial #100, beat the R1 champion #61 (0.970 → 0.972). Very marginal improvement with more noisy-eval episodes (100 vs 50 in R1).

#### Key findings

**T_obs breakdown** (completed trials):
| T_obs | n | mean weighted | best | pruning rate |
|-------|---|--------------|------|--------------|
| 1 | 22 | 0.913 | 0.954 | 0% |
| **2** | **58** | **0.942** | **0.972** | 5% |
| 3 | 7 | 0.869 | 0.935 | **79%** (27/34) |

**T_obs=3 is catastrophic**: 79% pruning rate. The extra conditioning tokens (294 vs 196) cause training collapse with nl=6 specifically. Only nl=4 tolerates T_obs=3 (and even then, never wins).

**T_obs=2 wins decisively**: both highest mean (0.942) and best single trial (0.972).

**T_obs=1 is safe but suboptimal**: zero pruning but ceilings lower — the policy needs at least one past frame for velocity information.

**n_layers breakdown (T_obs=2 only)**:
| nl | n | mean | best |
|----|---|------|------|
| 4 | 9 | 0.934 | 0.968 |
| **6** | **35** | **0.944** | **0.972** |
| 8 | 14 | 0.940 | 0.953 |

**nl=6 is the clear winner**. nl=4 is close but never produces a champion. nl=8 is surprisingly worse than nl=6 (possibly overfitting the narrow task).

#### TPE convergence

- Trial 0-5: rapid improvement as TPE builds its model
- Trial 5-30: refinement (best: 0.953 → 0.958)
- Trial 30-100: flat plateau (0.958 → 0.972)
- Trial 100+: **17 trials with zero improvement**, TPE exploiting same local optimum

TPE effectively converged by trial ~100. Additional trials were just micro-perturbations in a narrow region (lr ≈ 2.7-3.5e-4, lr_adapter bimodal at ~8e-6 or ~2.3e-5).

#### Edge exploitation signals (critical for R3 design)

Top 10 R2 trials — parameter positions within the narrow R2 range:
- **lr**: 6/10 trials in upper half; best clustered at 2.8-3.5e-4 (near upper edge 3.6e-4)
- **lr_adapter**: **bimodal** — 6/10 near upper edge (2.0-2.7e-5), 4/10 near lower edge (7-10e-6). No middle ground.
- **p_drop_attn**: mostly in upper half (0.17-0.22, near the 0.22 cap)

**Interpretation**: The narrowed R2 ranges were too tight in lr_adapter particularly — TPE hit both edges, indicating the true optima extend beyond both bounds. The bimodal pattern also confirms there are **two distinct winning regimes**:
- **Low lr_adapter (~1e-5)**: adapter effectively frozen, denoiser does all the work
- **High lr_adapter (~2.5e-5)**: adapter co-adapts with the denoiser

Both achieve equivalent quality (~0.965). Within top-20 R1 trials, Spearman(lr, lr_adapter) ≈ -0.37, suggesting the two regimes may favor slightly different main lr values.

---

## Round 3: `v3_square_bf16_hp_sweep_r3` — Rapid Wide Search

**Date**: 2026-04-12 onwards

**Motivation**: R2 showed TPE is exploiting the narrowed ranges too tightly, hitting edges. R1/R2 ranges were too constrained to find the true optimum. Combined with the finding that 50 epochs is sufficient for rapid convergence (peak typically by epoch 25-50 from multi-seed runs), we can afford to run **many more trials with a much wider search space** in roughly the same wall-clock time.

### Changes from R2

| Aspect | R2 | R3 | Why |
|--------|-----|-----|-----|
| Study name | `..._r2` | `..._r3` | Separate data |
| num_epochs | 200 | **50** | Rapid iteration, TPE ranking converges by epoch ~100 anyway |
| eval_full_every_epoch | 10 | **5** | 10 evals per trial (from 20) |
| n_trials_per_worker | 3 | **10** | More trials per worker for exploration |
| Pruner | MedianPruner (50%) | **PercentilePruner (25%)** | More aggressive pruning for wide search |
| n_warmup_steps | 50 | **19** | Allow pruning after 4 evals in a 50-epoch run |
| Objective | 0.5*peak + 0.5*last10_avg | **0.3*peak + 0.7*overall_avg** | Less peak-sensitive; with 10 evals, peak is noisy |
| T_obs | searched {1, 2, 3} | **fixed at 2** | T_obs=3 is garbage, T_obs=1 is worse on average |
| n_cond_layers | fixed at 4 | fixed at 4 (same) | Already established |
| d_model | fixed at 384 | fixed at 384 (same) | Already established |

### Search Space (wide)

| Parameter | Range | Type | vs R2 |
|-----------|-------|------|-------|
| `lr` | **[1.0e-5, 1.0e-3]** | log-uniform | 2 orders wider (R2: [1.8e-4, 3.6e-4]) |
| `lr_adapter` | **[1.0e-7, 1.0e-4]** | log-uniform | 3 orders wider (R2: [8e-6, 2.8e-5]) |
| `n_layers` | {4, 6, 8} | categorical | same |
| `p_drop_attn` | [0.03, 0.25] | continuous | slightly wider |

### Trial Budget

- **~50 min per trial** (d=384, T_obs=2, bf16 cached tokens, 100-episode evals every 5 epochs)
- 17 workers × 10 trials each = **170 target trials**
- Expected wall-clock: ~9 hours per worker, with strong overlap → **~10-15 hours total**

### Questions R3 aims to answer

1. Where are the **true optima** for lr and lr_adapter when TPE has full freedom?
2. Is the **lr/lr_adapter bimodal pattern** real (two distinct regimes) or an artifact of R2's narrow range?
3. Can **50-epoch rapid training** reliably identify the best HP region? (Sweep efficiency validation.)
4. Does the new metric **0.3*peak + 0.7*avg** produce a stabler ranking than the old weighted?
5. With **PercentilePruner(25%)** instead of MedianPruner(50%), does pruning eliminate more bad trials earlier?

### Expected behavior

Given the evidence that the top R2 configs cluster tightly around (lr ≈ 3e-4, lr_adapter ≈ 1e-5 or 2.5e-5, pdrop ≈ 0.18), R3 TPE should:
- Spend the first ~30 random trials covering the wide space
- Converge toward the same region within 50-60 trials
- Find a slightly better optimum (maybe 0.975 → 0.98) by exploring the true edges of the optimum basin
- Possibly discover that lr just outside R2's range (e.g. 4-5e-4) works even better, or confirm 3e-4 is optimal

### Metric bug found after 24 trials — switching to R3b

After the first 24 R3 trials launched, inspection of their trajectories revealed a structural problem with the `0.3*peak + 0.7*overall_avg` objective. With `warmup_steps=1000` (~2.2 epochs) and `eval_full_every_epoch=5`, the first 2–3 eval points (epochs 4, 9, 14) are dominated by warmup noise, not by HP quality. Averaging them in rewards *fast-converging* HPs rather than *best-converging* HPs.

Concrete example from the running R3 trials:

| Trial | Trajectory | peak | overall_avg | 0.3·peak + 0.7·overall_avg |
|-------|-----------|------|-------------|------------------------------|
| #0 | 0.03, 0.24, 0.36, 0.55, 0.69 (still climbing) | 0.69 | 0.374 | **0.469** |
| #16 | 0.83, 0.95, 0.95, 0.97, 0.95, 0.91 | 0.97 | 0.927 | **0.940** |

Trial #0 may well plateau above 0.85 if it ran to completion, but its epoch-4 value of 0.03 is permanently baked into the overall average — no amount of final-plateau excellence can outweigh the warmup noise. TPE's posterior will systematically prefer HPs that warm up fast, which is orthogonal to what we actually want (final-plateau quality).

**Methodological finding**: for rapid 50-epoch sweeps with dense early evals and nontrivial warmup, use **tail-window averaging**, not overall averaging. The pruner can still report raw step-wise SR (that's a valid step-wise signal — at epoch 14, low SR legitimately means worse-so-far), but the final objective must use the post-convergence window.

R3's 24 in-flight trials were allowed to finish under the old metric and remain in the database as a separate record (`v3_square_bf16_hp_sweep_r3`). TPE for the continued sweep resumes under a new study name (`..._r3b`) with the fixed metric.

---

## Round 3b: `v3_square_bf16_hp_sweep_r3b` — fixed metric

**Date**: 2026-04-12 onwards

**Motivation**: R3's `overall_avg` biased TPE toward fast-converging HPs (see R3 "Metric bug found" note above). R3b restarts TPE cleanly with the corrected metric.

### Only change from R3

| Aspect | R3 | R3b |
|--------|-----|-----|
| Objective | 0.3·peak + 0.7·overall_avg | **0.3·peak + 0.7·last_6_avg** |
| Study name | `..._r3` | `..._r3b` |

Everything else identical: 50 epochs, eval every 5, 100-ep × 15-env evals, PercentilePruner(25%, n_warmup_steps=19), wide search space (`lr` [1e-5, 1e-3] log, `lr_adapter` [1e-7, 1e-4] log, `n_layers` {4,6,8}, `p_drop_attn` [0.03, 0.25]), locked architecture (d=384, T_obs=2, n_cond=4, n_head=6, spatial_pool=7, flow matching, chi norm, rot6d), 17 workers × 10 trials = 170 target.

### Why `last_6_avg` specifically

A 50-epoch run with `eval_full_every_epoch=5` gives 10 eval points (epochs 4, 9, 14, …, 49). The last 6 correspond to epochs 24, 29, 34, 39, 44, 49 — comfortably past warmup (which ends by epoch ~2.2) and past the rapid ramp-up phase seen in R3 trajectories. This leaves the tail window representing the converged plateau, matching the R1/R2 `last_10_avg` philosophy (last-half of training) while still respecting R3's rapid-iteration compute budget.

For pruned trials with fewer than 6 evals, the slice `rates[-6:]` gracefully falls back to the whole trajectory — same behavior as the old metric for pruned trials (which were never the main ranking signal anyway since they're pruned).

### Pruner is unchanged — and should be

Pruning uses raw step-wise SR via `trial.report(sr, epoch)` in `training/train_v3.py`. That's correct for pruning: at a fixed epoch, all trials have had equal training budget, so raw SR is the right comparison. The metric bias only matters for the final objective (what TPE's posterior learns from), which uses the returned `objective_value` from `objective()`.

### Questions R3b aims to answer (unchanged from R3 but now under a valid metric)

1. Where are the **true optima** for lr and lr_adapter with wide search + unbiased ranking?
2. Does the **bimodal lr_adapter pattern** from R2 persist when the range opens to 3 orders of magnitude?
3. Is the 50-epoch + last_6 averaging combination enough to reliably identify top-3 configs?
4. Does `last_6_avg` produce a materially different winning HP region than R1/R2's `last_10_avg` over 200-300 epochs?

