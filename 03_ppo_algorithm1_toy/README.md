# PPO Algorithm 1 on a toy gridworld (paper Section 5)

Paper reference: Section 5 of arXiv:1707.06347 - Algorithm 1 (actor-critic style PPO), the combined objective Eq. 9 (`L_CLIP - c1*L_VF + c2*S`), and truncated GAE, Eq. 10-12.

## What it does and why
It implements the loop of Algorithm 1 from scratch, so the algorithm can be shown running:

1. N = 8 actors each run `pi_theta_old` for T = 64 timesteps in a 5x5 gridworld (goal reward +1, step cost -0.01, 50-step episodes).
2. Advantages by truncated GAE (`gamma` = 0.99, `lambda` = 0.95; episode ends cut the bootstrap).
3. K = 4 epochs of minibatch ascent (minibatch 128 of the N*T = 512 samples) on `L_CLIP - c1*L_VF + c2*S` (eps = 0.2, c1 = 0.5, c2 = 0.01). Gradients are written by hand for a tabular softmax policy and a tabular value function.
4. `theta_old <- theta` and repeat.

It first runs a **GAE unit check**: the recursion equals the explicit sum of Eq. 11, and `lambda = 1` equals the Eq. 10 estimator (exponents written in the standard form). It then trains with the clipped objective and with no clipping, over 5 seeds, and sweeps the policy step size.

## How to run
```bash
cd ppo_proof_of_work_3/03_ppo_algorithm1_toy
pip install -r requirements.txt
python ppo_algorithm1_toy.py                  # about 10 minutes (includes the step-size sweep)
python ppo_algorithm1_toy.py --seeds 2 --iters 40 --out output/quick.png    # quick check
```

## Result (default settings, 5 seeds)
- The GAE unit check passes.
- With clipping, the mean return of finished episodes goes from -0.057 (iteration 1) to 0.909 (iteration 20) and 0.919 (final). The average clipped fraction is 0.9%, and the average exact KL per update (over visited states) is 0.0022, against 0.0047 without clipping (the 2 seeds that did not diverge).

| policy lr | clipped eps = 0.2: final return (diverged seeds) | no clipping: final return (diverged seeds) |
|---|---|---|
| 1 | 0.857 (0/5) | 0.857 (0/5) |
| 2.5 | 0.899 (0/5) | 0.899 (0/5) |
| 5 | 0.910 (0/5) | 0.912 (0/5) |
| 10 | 0.916 (0/5) | 0.917 (0/5) |
| 20 | 0.919 (0/5) | 0.918 (**3/5**) |
| 40 | 0.921 (0/5) | 0.920 (**3/5**) |

("Diverged" = the parameters became non-finite; final return = mean of the last 10 iterations, diverged seeds excluded.)

## What it supports and what it does not
- **Supported:** Algorithm 1 as written learns this task, and the clipped objective was stable at every step size tried, whereas the unclipped objective diverged in 3 of 5 seeds at the two largest step sizes. This matches the paper's point that clipping makes repeated minibatch epochs safe.
- **Not supported:** clipping does not give a better final return here. When both runs finish, the returns are the same (about 0.92), and without clipping the return rises slightly faster in the first iterations. The benefit in this toy is robustness to step size, not performance.
- **Early-iteration KL (default lr 20):** the unclipped runs have a KL spike of about 0.13 in the first few iterations, against a peak of about 0.02 with clipping (about 16% of samples clipped at the peak). This is the "destructively large update" effect the paper describes, read from the top-middle and top-right panels of the PNG.

## Caveats
- The "no clipping" curves in the PNG average only the 2 of 5 seeds that did not diverge (survivors), so they are not directly comparable with the 5-seed clipped curves.
- A toy gridworld with tabular functions, not the paper's MuJoCo or Atari benchmarks.
- Advantages are not normalised and `V_targ = A + V_old`, which are common choices but not specified in the paper.

Output: `output/ppo_algorithm1_toy.png`.
