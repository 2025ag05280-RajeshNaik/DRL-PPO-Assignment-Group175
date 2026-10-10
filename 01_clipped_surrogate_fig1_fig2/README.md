# Clipped surrogate objective - Figure 1 and Figure 2 idea (paper Section 3)

Paper reference: Section 3, Eq. 6-7 (`L_CPI`, `L_CLIP`), Figure 1 and Figure 2 of arXiv:1707.06347.

## What it does and why
The clipped surrogate is PPO's core idea. This script makes its properties visible and checks them numerically:

1. **Figure 1 idea:** one term of `L_CLIP = min(r*A, clip(r, 1-eps, 1+eps)*A)` against the probability ratio `r`, for `A > 0` and `A < 0`. It asserts that the objective is capped at `1+eps` for `A > 0`, gives no extra reward below `1-eps` for `A < 0`, and keeps the unclipped (worse) term when the policy moves in the wrong direction.
2. **Figure 2 idea:** a toy 5-armed softmax policy takes one PPO update (10 epochs on `L_CLIP`, eps = 0.2, batch 256). The script then interpolates from `theta_old` (alpha = 0) to the updated `theta` (alpha = 1) and beyond (up to 1.5), and plots `L_CPI`, `E[clip(r)A]`, `L_CLIP` and `E[KL(pi_old || pi)]`.

## How to run
```bash
cd ppo_proof_of_work_3/01_clipped_surrogate_fig1_fig2
pip install -r requirements.txt
python clipped_surrogate_fig1_fig2.py
python clipped_surrogate_fig1_fig2.py --eps 0.1 --seed 3 --out output/other.png
```

## Result (default settings)
- `max(L_CLIP - L_CPI) = 0` over the whole sweep: `L_CLIP` is a lower bound of `L_CPI`.
- At alpha = 0.02 both equal 0.00100: they agree to first order near `theta_old`.
- At the PPO update point (alpha = 1): KL = 0.0179, `L_CPI` = 0.0505, `L_CLIP` = 0.0440.
- `L_CLIP` is maximal at alpha = 1.02. Beyond that `L_CPI` keeps growing (0.0756 at alpha = 1.5) while `L_CLIP` falls (0.0381): the clipped objective penalises an overly large update.

Output: `output/clipped_surrogate_fig1_fig2.png`.

## Link to the paper
The paper reports for Hopper-v1 that the update lands at KL of about 0.02 and `L_CLIP` is maximal there. Our toy gives KL = 0.018 and a peak at alpha = 1.02, the same qualitative picture.

## Caveats
- It is a toy softmax policy on a bandit, not Hopper-v1; it illustrates the figures and does not reproduce them.
- "`L_CLIP` is a lower bound of `L_CPI`" is true by construction (`min(x, y) <= x`), so the check confirms the implementation, while the informative part is where `L_CLIP` peaks.
