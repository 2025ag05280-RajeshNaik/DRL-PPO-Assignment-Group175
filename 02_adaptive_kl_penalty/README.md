# Adaptive KL penalty coefficient (paper Section 4, Eq. 8)

Paper reference: Section 4 of arXiv:1707.06347 - the KL-penalised objective `L_KLPEN = E[r*A - beta*KL(pi_old || pi_theta)]` and the rule: compute `d = E[KL]` after each update; if `d < d_targ/1.5` then `beta <- beta/2`; if `d > d_targ*1.5` then `beta <- beta*2`. The paper says `beta` "quickly adjusts", the initial `beta` "is not important in practice", and KL occasionally differs from `d_targ` but "these are rare". The same paper motivates the adaptive rule because it is hard to choose one fixed `beta` when problem characteristics change.

## What it does and why
A toy test of those claims. A 5-armed softmax policy takes 100 updates (50 epochs of gradient ascent on `L_KLPEN` per update, batch 64, `d_targ` = 0.01, 30 seeds). Every update starts from the same uniform policy, so only `beta` and the sampling noise change between updates. Half-way through, the reward scale is multiplied by 3 (problem characteristics change). It compares:

- the adaptive rule from three different initial values `beta0` = 0.1, 1 and 10;
- a **fixed** `beta` (the value the adaptive rule settled on in phase 1, then frozen).

## How to run
```bash
cd ppo_proof_of_work_3/02_adaptive_kl_penalty
pip install -r requirements.txt
python adaptive_kl_penalty.py
python adaptive_kl_penalty.py --d-targ 0.02 --jump 5 --seeds 50 --out output/other.png
```

## Result (default settings; statistics skip the first 15 updates of each phase)
| setting | phase | median d | in band | d > 2 d_targ | d < d_targ/2 | median beta |
|---|---|---|---|---|---|---|
| adaptive, beta0 = 0.1 | 1 (scale 1) | 0.0097 | 0.39 | 0.18 | 0.16 | 3.2 |
| adaptive, beta0 = 0.1 | 2 (scale 3) | 0.0113 | 0.36 | 0.15 | 0.25 | 6.4 |
| adaptive, beta0 = 1 | 1 | 0.0109 | 0.42 | 0.14 | 0.20 | 2.0 |
| adaptive, beta0 = 1 | 2 | 0.0098 | 0.58 | 0.15 | 0.12 | 8.0 |
| adaptive, beta0 = 10 | 1 | 0.0100 | 0.54 | 0.12 | 0.14 | 2.5 |
| adaptive, beta0 = 10 | 2 | 0.0087 | 0.30 | 0.25 | 0.19 | 10.0 |
| fixed beta = 2 | 1 | 0.0131 | 0.57 | 0.13 | 0.02 | 2.0 |
| fixed beta = 2 | 2 | **0.1065** | 0.00 | **1.00** | 0.00 | 2.0 |

("in band" = d within [d_targ/1.5, 1.5 d_targ].) After the jump the adaptive runs get back inside the band after a median of 2 to 4 updates; the fixed beta never does.

## What it supports and what it does not
- **Supported:** the median `d` sits at `d_targ` for all three initial `beta0`, and `beta` ends up at the same order of magnitude whatever the start. After the reward-scale jump the adaptive rule re-tunes `beta` (about 2-3 to about 6-10) within a few updates, whereas a fixed `beta` lets the KL grow 10 times above target. This agrees with the paper's reason for using an adaptive coefficient.
- **Not supported at the default settings:** "these deviations are rare". In this toy, only 30-58% of updates are inside the band, and 26-44% of updates are either above 2 x `d_targ` or below `d_targ`/2. The paper does not define "significantly different", and its batches are far larger than 64 samples (see the batch-size check below), so this is not a contradiction of the paper, only a result that depends on the setting.
- **Where adaptivity does not help:** in the stationary phase 1, the fixed `beta` (57% in band, median d = 0.0131) was at least as good as the adaptive runs (39-54% in band). The adaptive rule only pays off after the problem changes in phase 2.

## Batch-size check (why "rare" depends on the setting)
Re-running with `--batch 512` and `--batch 4096` (15 seeds each) gives the fraction of updates inside the band:

| beta0 | phase | batch 64 (default, 30 seeds) | batch 512 | batch 4096 |
|---|---|---|---|---|
| 0.1 | 1 | 0.39 | 0.27 | 0.10 |
| 0.1 | 2 | 0.36 | 0.60 | 0.92 |
| 1 | 1 | 0.42 | 0.91 | 1.00 |
| 1 | 2 | 0.58 | 0.93 | 1.00 |
| 10 | 1 | 0.54 | 0.92 | 1.00 |
| 10 | 2 | 0.30 | 0.07 | 0.00 |

- With less sampling noise, the in-band fraction rises to 0.9-1.00 for several starts, so out-of-band updates can be rare, as the paper says.
- It is not uniform: for some starting values (beta0 = 0.1 in phase 1, beta0 = 10 in phase 2) the fraction stays low even with a large batch. `beta` moves only in factors of 2, so it sits on a grid (0.1, 0.2, ..., 1.6, 3.2 for beta0 = 0.1) and can alternate between two values that straddle the band without ever landing inside it. This is our reading of the pattern; we did not test it separately.
- So the median `d` is near `d_targ` whatever `beta0` is, but how often `d` lands inside the band also depends on `beta0` through this grid. The paper's statement that the initial `beta` "is not important in practice" holds for where `d` settles, not for the in-band fraction.

## Caveats
- Toy bandit and a hand-written gradient optimiser (the inner step must be small enough to stay stable for large `beta`), not the paper's MuJoCo setup.
- Each update restarts from the same policy; this isolates the `beta` controller but is not a full training run. We first tried carrying the policy over between updates, but it collapsed to a nearly deterministic policy within a few updates (KL near 0 for the rest of the run), which leaves nothing for the controller to regulate; that is why the restart design and these settings (`lr` 0.1, 50 epochs, reward jump x3) were chosen.
- The fixed `beta` (2) is the value the adaptive rule settled on in phase 1. A different fixed value would fit phase 2 better but then miss phase 1.

Output: `output/adaptive_kl_penalty.png`.
