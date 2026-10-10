"""Adaptive KL penalty coefficient (PPO paper Section 4, Eq. 8): does beta track d_targ?

Toy softmax policy on a 5-armed bandit. Every update starts from the same (uniform) policy, so the only thing
that changes between updates is beta and the sampling noise. Half-way through, the reward scale changes by
`--jump` (the paper motivates adaptive beta because problem characteristics change during learning).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MEANS = np.linspace(0.0, 1.0, 5)
ARMS = len(MEANS)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def run(seed, beta0, adaptive, d_targ, iters, epochs, batch, lr, noise, jump):
    rng = np.random.default_rng(seed)
    pi_old = softmax(np.zeros(ARMS))
    beta = beta0
    d_hist, beta_hist = [], []
    for it in range(iters):
        scale = jump if it >= iters // 2 else 1.0
        actions = rng.choice(ARMS, size=batch, p=pi_old)
        rewards = scale * rng.normal(MEANS[actions], noise)
        adv = rewards - rewards.mean()
        th = np.zeros(ARMS)
        for _ in range(epochs):  # maximise E[r A] - beta * KL(pi_old || pi_theta)
            pi = softmax(th)
            r = pi[actions] / pi_old[actions]
            w = r * adv / batch
            grad = np.bincount(actions, weights=w, minlength=ARMS) - pi * w.sum() - beta * (pi - pi_old)
            th = th + lr * grad
        pi_new = softmax(th)
        d = float(np.sum(pi_old * np.log(pi_old / pi_new)))
        d_hist.append(d)
        beta_hist.append(beta)
        if adaptive:
            if d < d_targ / 1.5:  # the constants 1.5 and 2 are the paper's heuristic choices
                beta /= 2
            elif d > d_targ * 1.5:
                beta *= 2
    return np.array(d_hist), np.array(beta_hist)


def phase_stats(D, B, lo, hi, d_targ, start, stop):
    d, b = D[:, start:stop], B[:, start:stop]
    return (
        np.median(d),
        np.mean((d >= lo) & (d <= hi)),
        np.mean(d > 2 * d_targ),
        np.mean(d < d_targ / 2),
        np.median(b),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d-targ", type=float, default=0.01)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--noise", type=float, default=0.5)
    p.add_argument("--jump", type=float, default=3.0, help="reward scale after the half-way point")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--warmup", type=int, default=15, help="updates ignored at the start of each phase")
    p.add_argument("--out", default="output/adaptive_kl_penalty.png")
    a = p.parse_args()

    lo, hi = a.d_targ / 1.5, a.d_targ * 1.5
    half = a.iters // 2
    phases = {"phase 1 (scale 1)": (a.warmup, half), f"phase 2 (scale {a.jump:g})": (half + a.warmup, a.iters)}
    common = (a.d_targ, a.iters, a.epochs, a.batch, a.lr, a.noise, a.jump)

    configs = [(f"adaptive, beta0={b0}", b0, True) for b0 in (0.1, 1.0, 10.0)]
    results = {}
    for name, b0, adaptive in configs:
        runs = [run(s, b0, adaptive, *common) for s in range(a.seeds)]
        results[name] = (np.array([r[0] for r in runs]), np.array([r[1] for r in runs]))
    # Fixed beta: the value the adaptive rule settled on in phase 1, then frozen.
    beta_fixed = float(np.median(results["adaptive, beta0=1.0"][1][:, a.warmup:half]))
    runs = [run(s, beta_fixed, False, *common) for s in range(a.seeds)]
    fixed_name = f"fixed beta={beta_fixed:.3g}"
    results[fixed_name] = (np.array([r[0] for r in runs]), np.array([r[1] for r in runs]))

    print(f"d_targ={a.d_targ}, accepted band=[{lo:.4f}, {a.d_targ * 1.5:.4f}], {a.seeds} seeds, {a.iters} updates, reward scale x{a.jump:g} at update {half}")
    print(f"statistics skip the first {a.warmup} updates of each phase\n")
    print(f"{'setting':<24}{'phase':<20}{'median d':>9}{'in band':>9}{'d>2*targ':>9}{'d<targ/2':>9}{'median beta':>12}")
    for name, (D, B) in results.items():
        for ph, (s, e) in phases.items():
            md, inb, hi2, lo2, mb = phase_stats(D, B, lo, hi, a.d_targ, s, e)
            print(f"{name:<24}{ph:<20}{md:>9.4f}{inb:>9.2f}{hi2:>9.2f}{lo2:>9.2f}{mb:>12.3f}")

    x = np.arange(1, a.iters + 1)
    print("\nUpdates after the jump until d first falls inside the band (median over seeds; 0 = already inside):")
    for name, (D, _) in results.items():
        inside = (D[:, half:] >= lo) & (D[:, half:] <= hi)
        first = [int(np.argmax(row)) for row in inside if row.any()]
        label = f"{np.median(first):.0f}" if len(first) > a.seeds // 2 else "never"
        print(f"  {name:<24}{label:>6}  ({len(first)}/{a.seeds} seeds got inside)")
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8), sharex=True)
    for name, (D, B) in results.items():
        style = "--" if name.startswith("fixed") else "-"
        axes[0].plot(x, np.median(D, axis=0), style, label=name)
        if not name.startswith("fixed"):
            axes[1].plot(x, np.median(B, axis=0), label=name)
    axes[0].axhspan(lo, hi, color="green", alpha=0.12, label="d_targ/1.5 .. d_targ*1.5")
    axes[0].axhline(a.d_targ, color="green", lw=0.8)
    axes[1].axhline(beta_fixed, color="tab:red", ls="--", lw=0.8, label="fixed beta used for comparison")
    for ax in axes:
        ax.axvline(half + 0.5, color="black", lw=0.8, ls=":")
        ax.set_yscale("log")
    axes[0].set_ylabel("measured KL d per update (median)")
    axes[0].set_title(f"Section 4 adaptive KL penalty, toy policy, {a.seeds} seeds (dotted line: reward scale x{a.jump:g})")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("penalty coefficient beta (median)")
    axes[1].set_xlabel("policy update")
    axes[1].legend(fontsize=8)
    fig.tight_layout()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
