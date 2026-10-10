# Atari Table 2 re-derived from Table 6 (paper Section 6.4)

Paper reference: Section 6.4, Table 2 (games "won" by each algorithm) and Appendix Table 6 (mean final score of A2C, ACER and PPO per game) of arXiv:1707.06347.

## What it does and why
Table 2 reports win counts for two metrics: (1) average reward over all of training and (2) average reward over the last 100 episodes. This script shows how the second set of counts is obtained and checks it:

- Input: `table6_scores.csv` - the 49 rows of Appendix Table 6 (final scores, transcribed from the paper).
- Rule per game: winner = `argmax(A2C, ACER, PPO)`; equal top scores count as a tie.
- Output: our tally next to the paper's Table 2, the tied games, and a bar chart.

## How to run
```bash
cd ppo_proof_of_work_3/04_atari_table2_from_table6
pip install -r requirements.txt
python atari_table2_from_table6.py
python atari_table2_from_table6.py --csv table6_scores.csv --out output/other.png
```

## Result
| Metric 2 (last 100 episodes) | A2C | ACER | PPO | Tie |
|---|---|---|---|---|
| Paper, Table 2 | 1 | 28 | 19 | 1 |
| Our tally from Table 6 | 1 | 27 | 19 | 2 |

PPO (19) and A2C (1) match exactly. The only difference is Pong: ACER and PPO both show 20.7 in the rounded table, so we count a tie, while the paper counts it for ACER, presumably from unrounded scores. Venture (0 for all three) is the other tie.

Output: `output/atari_table2_from_table6.png`.

## Limits
- Metric 1 (all-training average: PPO 30, ACER 18, A2C 1) cannot be re-derived from Table 6; the paper only shows it through learning curves (Appendix Figure 6). It is quoted from Table 2.
- The CSV was checked row by row (49 of 49 games, all three scores) against Table 6 in the text version of the paper (`Proximal_Policy_Optimization_Algorithms.md`); re-check it against the PDF if you edit it.
