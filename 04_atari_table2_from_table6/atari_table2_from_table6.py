"""Tally per-game winners (A2C / ACER / PPO) from the PPO paper's Appendix Table 6."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_TABLE2_LAST100 = {"A2C": 1, "ACER": 28, "PPO": 19, "Tie": 1}
NAMES = ("A2C", "ACER", "PPO")


def tally(rows):
    wins = {"A2C": 0, "ACER": 0, "PPO": 0, "Tie": 0}
    ties = []
    for row in rows:
        scores = (float(row["a2c"]), float(row["acer"]), float(row["ppo"]))
        best = max(scores)
        top = [n for n, s in zip(NAMES, scores) if s == best]
        if len(top) == 1:
            wins[top[0]] += 1
        else:
            wins["Tie"] += 1
            ties.append((row["game"], "/".join(top), best))
    return wins, ties


def main():
    default_csv = Path(__file__).with_name("table6_scores.csv")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(default_csv), help="path to table6_scores.csv")
    parser.add_argument("--out", default="output/atari_table2_from_table6.png", help="output image path")
    args = parser.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 49, f"expected 49 games, found {len(rows)}"

    wins, ties = tally(rows)
    total = sum(wins.values())

    print(f"Games: {len(rows)}  (Table 6 = mean final score, last 100 episodes)\n")
    print(f"{'':28}{'A2C':>6}{'ACER':>6}{'PPO':>6}{'Tie':>6}{'Total':>7}")
    p = PAPER_TABLE2_LAST100
    print(f"{'Paper, Table 2 (metric 2)':28}{p['A2C']:>6}{p['ACER']:>6}{p['PPO']:>6}{p['Tie']:>6}{sum(p.values()):>7}")
    print(f"{'Our tally from Table 6':28}{wins['A2C']:>6}{wins['ACER']:>6}{wins['PPO']:>6}{wins['Tie']:>6}{total:>7}")

    print("\nTied games in our tally (equal rounded scores):")
    for game, who, score in ties:
        print(f"  {game}: {who} at {score}")

    print(f"\nPPO win share: {wins['PPO']}/{total} = {100 * wins['PPO'] / total:.1f}%")
    print(f"ACER win share: {wins['ACER']}/{total} = {100 * wins['ACER'] / total:.1f}%")
    print("Metric 1 (all-training average, PPO 30/ACER 18/A2C 1) is not derivable from Table 6; see paper Table 2.")

    cats = ["A2C", "ACER", "PPO", "Tie"]
    xs = range(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    b1 = ax.bar([x - w / 2 for x in xs], [p[c] for c in cats], w, label="Paper, Table 2")
    b2 = ax.bar([x + w / 2 for x in xs], [wins[c] for c in cats], w, label="Our tally from Table 6")
    ax.bar_label(b1)
    ax.bar_label(b2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(cats)
    ax.set_ylabel("games won (out of 49)")
    ax.set_title("Atari wins, last 100 episodes: paper vs our tally")
    ax.legend()
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
