"""PPO Algorithm 1 (paper Section 5) on a tiny gridworld, with GAE (Eq. 11-12) and the combined loss (Eq. 9).

Tabular softmax policy and tabular value function, gradients written by hand (NumPy only).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GRID = 5
N_STATES = GRID * GRID
N_ACTIONS = 4  # up, down, left, right
GOAL = N_STATES - 1
MAX_STEPS = 50
MOVES = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])


def softmax_rows(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class VecGrid:
    """N independent actors; an actor that finishes an episode restarts at state 0."""

    def __init__(self, n):
        self.s = np.zeros(n, dtype=int)
        self.t = np.zeros(n, dtype=int)
        self.ret = np.zeros(n)

    def step(self, a):
        row, col = self.s // GRID, self.s % GRID
        row = np.clip(row + MOVES[a, 0], 0, GRID - 1)
        col = np.clip(col + MOVES[a, 1], 0, GRID - 1)
        ns = row * GRID + col
        reward = np.where(ns == GOAL, 1.0, -0.01)
        self.t += 1
        self.ret += reward
        done = (ns == GOAL) | (self.t >= MAX_STEPS)
        finished = self.ret[done].tolist()
        ns = np.where(done, 0, ns)
        self.t[done] = 0
        self.ret[done] = 0.0
        self.s = ns
        return reward, done, finished


def gae(rewards, values, last_value, dones, gamma, lam):
    """Truncated GAE (Eq. 11-12) over a length-T segment; episode ends cut the bootstrap."""
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1])
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
    return adv


def gae_unit_check():
    rng = np.random.default_rng(0)
    T, gamma, lam = 8, 0.99, 0.95
    r = rng.normal(size=(T, 1))
    v = rng.normal(size=(T, 1))
    vT = rng.normal(size=1)
    d = np.zeros((T, 1))
    got = gae(r, v, vT, d, gamma, lam)[:, 0]
    vv = np.append(v[:, 0], vT[0])
    delta = r[:, 0] + gamma * vv[1:] - vv[:-1]
    explicit = np.array([sum((gamma * lam) ** l * delta[t + l] for l in range(T - t)) for t in range(T)])
    assert np.allclose(got, explicit), "recursion must equal the explicit sum of Eq. 11"
    lam1 = gae(r, v, vT, d, gamma, 1.0)[:, 0]
    eq10 = np.array(
        [-vv[t] + sum(gamma**l * r[t + l, 0] for l in range(T - t)) + gamma ** (T - t) * vv[T] for t in range(T)]
    )
    assert np.allclose(lam1, eq10), "lambda = 1 must reduce to the Eq. 10 estimator"
    print("GAE unit check passed: recursion == explicit Eq. 11 sum, and lambda=1 == Eq. 10 estimator.")


def train(seed, eps, hp):
    rng = np.random.default_rng(seed)
    theta = np.zeros((N_STATES, N_ACTIONS))
    V = np.zeros(N_STATES)
    env = VecGrid(hp.actors)
    T, N = hp.horizon, hp.actors
    hist = {k: np.full(hp.iters, np.nan) for k in ("ret", "kl", "clip", "ent")}
    last_ret = np.nan
    diverged = False
    with np.errstate(all="ignore"):
        for it in range(hp.iters):
            # Algorithm 1: every actor runs pi_theta_old for T timesteps
            S = np.zeros((T, N), dtype=int)
            A = np.zeros((T, N), dtype=int)
            R = np.zeros((T, N))
            D = np.zeros((T, N))
            finished_all = []
            pi_old_tab = softmax_rows(theta)
            for t in range(T):
                probs = pi_old_tab[env.s]
                a = np.minimum((rng.random(N)[:, None] > np.cumsum(probs, axis=1)).sum(axis=1), N_ACTIONS - 1)
                S[t], A[t] = env.s, a
                R[t], done, fin = env.step(a)
                D[t] = done
                finished_all += fin
            # advantage estimates with GAE
            v_old = V[S]
            adv = gae(R, v_old, V[env.s], D, hp.gamma, hp.lam)
            v_targ = adv + v_old
            s_f, a_f, adv_f, vt_f = S.ravel(), A.ravel(), adv.ravel(), v_targ.ravel()
            logp_old = np.log(pi_old_tab[s_f, a_f])
            # K epochs of minibatch ascent on L_CLIP - c1 L_VF + c2 S  (Eq. 9)
            M = len(s_f)
            clip_frac = []
            for _ in range(hp.epochs):
                order = rng.permutation(M)
                for k in range(0, M, hp.minibatch):
                    idx = order[k : k + hp.minibatch]
                    s, a, ad, vt = s_f[idx], a_f[idx], adv_f[idx], vt_f[idx]
                    probs = softmax_rows(theta[s])
                    logp = np.log(probs[np.arange(len(idx)), a])
                    r = np.exp(logp - logp_old[idx])
                    if eps is None:
                        active = np.ones(len(idx), bool)
                    else:
                        active = ~(((ad > 0) & (r > 1 + eps)) | ((ad < 0) & (r < 1 - eps)))
                        clip_frac.append(1 - active.mean())
                    g_logits = (active * r * ad)[:, None] * (np.eye(N_ACTIONS)[a] - probs)
                    ent = -(probs * np.log(probs)).sum(axis=1)
                    g_logits += hp.c2 * (-probs * (np.log(probs) + ent[:, None]))
                    g_theta = np.zeros_like(theta)
                    np.add.at(g_theta, s, g_logits)
                    g_v = np.zeros_like(V)
                    np.add.at(g_v, s, -2 * hp.c1 * (V[s] - vt))
                    theta += hp.lr_pi * g_theta / len(idx)
                    V += hp.lr_v * g_v / len(idx)
            if not (np.isfinite(theta).all() and np.isfinite(V).all()):
                diverged = True
                break
            new_tab = softmax_rows(theta)
            if not np.isfinite(new_tab).all():
                diverged = True
                break
            p_o, p_n = pi_old_tab[s_f], new_tab[s_f]
            kl = float(np.mean(np.sum(p_o * (np.log(p_o) - np.log(p_n)), axis=1)))  # exact KL(pi_old || pi_new) averaged over visited states
            if finished_all:
                last_ret = float(np.mean(finished_all))
            hist["ret"][it] = last_ret
            hist["kl"][it] = kl
            hist["clip"][it] = float(np.mean(clip_frac)) if clip_frac else 0.0
            hist["ent"][it] = float(-(new_tab * np.log(new_tab)).sum(axis=1).mean())
    return hist, diverged


def final_return(hist):
    return float(np.nanmean(hist["ret"][-10:]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--actors", type=int, default=8)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=128)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--lr-pi", type=float, default=20.0)
    p.add_argument("--lr-v", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--c1", type=float, default=0.5)
    p.add_argument("--c2", type=float, default=0.01)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default="output/ppo_algorithm1_toy.png")
    a = p.parse_args()

    gae_unit_check()
    settings = [(f"clipped eps={a.eps}", a.eps), ("no clipping", None)]
    print(f"\n{GRID}x{GRID} gridworld, goal reward +1, step cost -0.01, {a.actors} actors x T={a.horizon}, K={a.epochs} epochs, minibatch {a.minibatch}, {a.seeds} seeds, lr_pi={a.lr_pi}")
    print(f"{'setting':<18}{'diverged':>9}{'ret it.1':>10}{'ret it.20':>10}{'ret final':>10}{'mean KL':>9}{'mean clip':>10}")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x = np.arange(1, a.iters + 1)
    for name, eps in settings:
        res = [train(s, eps, a) for s in range(a.seeds)]
        ok = [h for h, d in res if not d]
        nd = sum(d for _, d in res)
        if ok:
            m = {k: np.mean([h[k] for h in ok], axis=0) for k in ok[0]}
            print(f"{name:<18}{f'{nd}/{a.seeds}':>9}{m['ret'][0]:>10.3f}{m['ret'][19]:>10.3f}{final_return(m):>10.3f}{np.mean(m['kl']):>9.4f}{np.mean(m['clip']):>10.3f}")
            for ax, key in zip(axes.ravel()[:4], ("ret", "kl", "clip", "ent")):
                ax.plot(x, m[key], label=f"{name} ({a.seeds - nd}/{a.seeds} seeds finished)")
        else:
            print(f"{name:<18}{f'{nd}/{a.seeds}':>9}   all seeds diverged (NaN)")
    titles = ["mean return of finished episodes", "KL(pi_old || pi_new) per iteration (visited states)", "fraction of samples clipped", "policy entropy"]
    for ax, t in zip(axes.ravel()[:4], titles):
        ax.set_title(t)
        ax.set_xlabel("iteration (Algorithm 1 outer loop)")
        ax.legend(fontsize=8)

    lrs = (1.0, 2.5, 5.0, 10.0, 20.0, 40.0)
    print("\nStep-size sweep (final return = mean of the last 10 iterations; diverged seeds excluded):")
    print(f"{'lr_pi':>6}" + "".join(f"{n:>26}" for n, _ in settings))
    ax = axes.ravel()[4]
    sweep = {n: ([], []) for n, _ in settings}
    for lr in lrs:
        hp = argparse.Namespace(**{**vars(a), "lr_pi": lr, "lr_v": a.lr_v})
        row = f"{lr:>6}"
        for name, eps in settings:
            res = [train(s, eps, hp) for s in range(a.seeds)]
            ok = [final_return(h) for h, d in res if not d]
            nd = sum(d for _, d in res)
            val = float(np.mean(ok)) if ok else np.nan
            sweep[name][0].append(val)
            sweep[name][1].append(nd)
            row += f"{('%.3f' % val if ok else 'diverged'):>16} ({nd}/{a.seeds} div.)"
        print(row)
    for name, _ in settings:
        ax.plot(lrs, sweep[name][0], "o-", label=name)
        for lr, v, nd in zip(lrs, *sweep[name]):
            if nd:
                ax.annotate(f"{nd}/{a.seeds} div.", (lr, v if np.isfinite(v) else 0), fontsize=7, textcoords="offset points", xytext=(0, 6))
    ax.set_xscale("log")
    ax.set_title("final return vs policy step size (finished seeds)")
    ax.set_xlabel("policy learning rate")
    ax.legend(fontsize=8)
    ax6 = axes.ravel()[5]
    for name, _ in settings:
        ax6.plot(lrs, np.array(sweep[name][1]) / a.seeds, "o-", label=name)
    ax6.set_xscale("log")
    ax6.set_ylim(-0.05, 1.05)
    ax6.set_title("fraction of seeds that diverged (non-finite parameters)")
    ax6.set_xlabel("policy learning rate")
    ax6.legend(fontsize=8)
    fig.suptitle("PPO Algorithm 1 on a toy gridworld (illustration, not the paper's benchmarks)")
    fig.tight_layout()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
