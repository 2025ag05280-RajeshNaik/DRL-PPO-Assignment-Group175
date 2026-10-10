"""Reproduce the idea of PPO paper Figures 1 and 2 (Section 3, Eq. 6-7) on a toy softmax policy."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRUE_MEANS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
ARMS = len(TRUE_MEANS)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def clipped_term(r, adv, eps):
    return np.minimum(r * adv, np.clip(r, 1 - eps, 1 + eps) * adv)


def figure1_checks(eps):
    lo, hi = 1 - eps, 1 + eps
    assert clipped_term(1.0, 1.0, eps) == 1.0  # equals r*A at r = 1
    assert np.isclose(clipped_term(1.5, 1.0, eps), hi)  # A > 0: gain capped at 1+eps
    assert np.isclose(clipped_term(0.5, -1.0, eps), -lo)  # A < 0: no extra reward below 1-eps
    assert np.isclose(clipped_term(1.5, -1.0, eps), -1.5)  # A < 0, r > 1+eps: unclipped term kept
    assert np.isclose(clipped_term(0.5, 1.0, eps), 0.5)  # A > 0, r < 1-eps: unclipped term kept


def ppo_update(seed, eps, batch, epochs, lr):
    rng = np.random.default_rng(seed)
    theta_old = np.zeros(ARMS)
    pi_old = softmax(theta_old)
    actions = rng.choice(ARMS, size=batch, p=pi_old)
    rewards = rng.normal(TRUE_MEANS[actions], 0.5)
    adv = rewards - rewards.mean()
    theta = theta_old.copy()
    for _ in range(epochs):  # gradient ascent on L_CLIP; clipped samples have zero gradient
        pi = softmax(theta)
        r = pi[actions] / pi_old[actions]
        active = ~(((adv > 0) & (r > 1 + eps)) | ((adv < 0) & (r < 1 - eps)))
        w = active * r * adv / batch
        theta = theta + lr * (np.bincount(actions, weights=w, minlength=ARMS) - pi * w.sum())
    return theta_old, theta, actions, adv


def sweep(theta_old, theta_new, actions, adv, eps, alphas):
    pi_old = softmax(theta_old)
    out = {k: [] for k in ("cpi", "clip_only", "lclip", "kl")}
    for a in alphas:
        pi = softmax(theta_old + a * (theta_new - theta_old))
        r = pi[actions] / pi_old[actions]
        out["cpi"].append(np.mean(r * adv))
        out["clip_only"].append(np.mean(np.clip(r, 1 - eps, 1 + eps) * adv))
        out["lclip"].append(np.mean(clipped_term(r, adv, eps)))
        out["kl"].append(float(np.sum(pi_old * np.log(pi_old / pi))))
    return {k: np.array(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="output/clipped_surrogate_fig1_fig2.png")
    a = p.parse_args()

    figure1_checks(a.eps)
    print("Figure 1 property checks passed (flat beyond 1+eps for A>0, below 1-eps for A<0, unclipped term kept in the wrong direction).")

    theta_old, theta_new, actions, adv = ppo_update(a.seed, a.eps, a.batch, a.epochs, a.lr)
    alphas = np.linspace(0, 1.5, 151)
    s = sweep(theta_old, theta_new, actions, adv, a.eps, alphas)

    gap = float(np.max(s["lclip"] - s["cpi"]))
    assert gap <= 1e-12, "L_CLIP must never exceed L_CPI"
    i1 = int(np.argmin(np.abs(alphas - 1.0)))
    i_small = int(np.argmin(np.abs(alphas - 0.02)))
    peak = alphas[int(np.argmax(s["lclip"]))]
    print(f"Toy policy: {ARMS}-armed softmax, batch {a.batch}, {a.epochs} epochs on L_CLIP (eps={a.eps})")
    print(f"max(L_CLIP - L_CPI) over the sweep = {gap:.2e}  (<= 0: L_CLIP is a lower bound of L_CPI)")
    print(f"at alpha=0.02: L_CPI={s['cpi'][i_small]:.5f}, L_CLIP={s['lclip'][i_small]:.5f}  (equal to first order near theta_old)")
    print(f"PPO update point (alpha=1): KL={s['kl'][i1]:.4f}, L_CPI={s['cpi'][i1]:.4f}, L_CLIP={s['lclip'][i1]:.4f}")
    print(f"alpha that maximises L_CLIP: {peak:.2f}; L_CPI keeps growing to {s['cpi'][-1]:.4f} at alpha=1.5, L_CLIP={s['lclip'][-1]:.4f}")

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2)
    r = np.linspace(0, 2, 401)
    for col, adv_val in enumerate((1.0, -1.0)):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(r, r * adv_val, "--", color="gray", label="r*A (L_CPI term)")
        ax.plot(r, clipped_term(r, adv_val, a.eps), color="tab:blue", label="L_CLIP term")
        ax.axvline(1 - a.eps, color="tab:red", lw=0.8, ls=":")
        ax.axvline(1 + a.eps, color="tab:red", lw=0.8, ls=":")
        ax.plot([1], [adv_val], "ro")
        ax.set_title(f"Figure 1 idea: A = {adv_val:+.0f}, eps = {a.eps}")
        ax.set_xlabel("probability ratio r")
        ax.legend()
    ax = fig.add_subplot(gs[1, :])
    ax.plot(alphas, s["cpi"], label="L_CPI = E[r A]")
    ax.plot(alphas, s["clip_only"], label="E[clip(r, 1-eps, 1+eps) A]")
    ax.plot(alphas, s["lclip"], lw=2.5, label="L_CLIP = E[min(r A, clip(r) A)]")
    ax.plot(alphas, s["kl"], label="E[KL(pi_old || pi)]")
    ax.axvline(1.0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("interpolation factor alpha  (0 = theta_old, 1 = PPO update)")
    ax.set_title("Figure 2 idea: surrogates along the update direction (toy policy, not Hopper-v1)")
    ax.legend()
    fig.tight_layout()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
