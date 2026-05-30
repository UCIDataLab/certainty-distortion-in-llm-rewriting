"""
Rao-Kupper Model — Bradley-Terry with Ties
===========================================
Load the csv file containing the following columns:
- "idx_i": i,
- "idx_j": j,
- "text_i": text_i,
- "text_j": text_j,
- "wins_i": wins_i,
- "wins_j": wins_j,
- "ties": ties,
- "total_votes": wins_i + wins_j + ties

With this information create the `wins` and `ties` matrices, 
where wins[i,j] corresponds to how many times text_i was preferred over text_j
and ties is a symmetric matrix, where ties[i,j] corresponds to how many times
text_i and text_j had a tie.

"""
import math
from pathlib import Path
from typing import List

import click
import numpy as np
np.seterr(invalid='raise', divide='raise')
import pandas as pd
import trackio
import tqdm

# ── Helpers (already implemented) ──────────────────────────────
def total_wins(wins, i):
    """Total wins by player i."""
    return wins[i,:].sum()

def total_ties(ties, i):
    """Total ties involving player i."""
    return ties[i].sum()


# ── TODO 1: Rao-Kupper strength update ─────────────────────────
def update_strengths(s, nu, wins, ties):
    """
    One iteration of the Rao-Kupper strength update.

    For each player i:

        numerator   = W_i + T_i / 2
        denominator = sum over j != i of:
                        (w_ij + w_ji + t_ij) / (s[i] + nu * sqrt(s[i]*s[j]))

        s_new[i] = numerator / denominator

    where:
        W_i    = total wins by player i          (use total_wins(i))
        T_i    = total ties involving player i   (use total_ties(i))
        w_ij   = wins[i][j]
        w_ji   = wins[j][i]
        t_ij   = ties[i][j]

    Args:
        s:  list of current strength estimates
        nu: current tie-threshold estimate

    Returns:
        s_new: list of updated strength estimates
    """
    n = len(s)
    s_new = [0.0] * n

    for i in range(n):
        num = total_wins(wins, i) + total_ties(ties, i) / 2

        denom = 0.0
        for j in range(n):
            if j == i:
                continue
            denom += (wins[i][j] + wins[j][i] + ties[i][j]) / (s[i] + nu * np.sqrt(s[i]*s[j]))

        s_new[i] = float(num / denom)
        # Debug (players should never become 0)
        if s_new[i] == 0:
            pass

    return s_new


# ── 2: Rao-Kupper nu update ───────────────────────────────
def update_nu(s, nu, ties):
    """
    One iteration of the Rao-Kupper nu update.

        T_total = total number of tied matches across all PAIRS i < j
                  (count each pair once — ties[i][j] already equals ties[j][i])

        denominator = sum over pairs i < j of:
                        ties[i][j] * sqrt(s[i]*s[j])
                        / (s[i] + nu*sqrt(s[i]*s[j]) + s[j])

        nu_new = T_total / denominator

    Args:
        s:  current strength estimates
        nu: current nu estimate

    Returns:
        nu_new: updated nu
    """
    n = len(s)
    T_total = np.triu(ties, k=1).sum()
    assert T_total != 0, "We do not expect the model to have 0 ties."

    denom = 0.0
    for i in range(n):
        for j in range(i + 1, n):   # i < j ensures we count each pair once
            geo = math.sqrt(s[i] * s[j])
            if s[i] + nu * geo + s[j] != 0:
                denom += ties[i,j] * geo / (s[i] + nu * geo + s[j])

    # return T_total / denom  (guard against denom == 0)
    if denom == 0:
        return nu
    return T_total / denom


# ─── 3: rank_players (same as before) ──────────────────────
def rank_players(s, ids):
    """
    Return list of (player_name, strength) sorted strongest first.
    """
    ranked = [(p, s) for p, s in zip(ids, s)]
    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    return ranked


# ── 4: run_btt loop ───────────────────────────────────────
def run_btt(wins, ties, max_iter=300, tol=1e-8):
    """
    Alternating iterative MLE for Rao-Kupper.

    Each iteration should:
      1. Call update_strengths(s, nu)  → s_new
      2. Normalise s_new so the values sum to n  (same trick as before)
      3. Call update_nu(s_new, nu)     → nu_new
      4. Compute delta = max change across all s values AND nu
      5. Update s, nu
      6. Break when delta < tol

    Start with:
        s  = [1.0] * n
        nu = 1.0          ← nu also needs an initial value!

    Return (s, nu).
    """
    n  = len(wins)
    s  = [1.0] * n
    nu = 1.0

    for iteration in tqdm.tqdm(range(1, max_iter + 1)):
        # step 1 — update strengths
        s_new = update_strengths(s, nu, wins, ties)

        # step 2 — normalise s_new so it sums to n
        total = sum(s_new)
        s_new = [s_i * n / total for s_i in s_new]

        # step 3 — update nu
        nu_new = update_nu(s_new, nu, ties)

        # step 4 — compute delta (max change in s AND nu)
        delta_s = max(abs(s_new[i] - s[i]) for i in range(n))
        delta_nu = abs(nu - nu_new)
        delta = max(delta_s, delta_nu)

        # step 5 — update s and nu
        s  = s_new
        nu = nu_new

        trackio.log({
            "delta_s": float(delta_s), 
            "delta_nu": float(delta_nu),
            "nu": float(nu),
            "delta": float(delta),
        }, step=iteration)

        # Step 6 — convergence check (already written)
        if delta_s < tol:
            trackio.alert(f"Strengths converged after {iteration} iterations.  nu = {nu:.4f}\n")
            break

    return s, nu


def load_matrices(csv_paths: List[str], smooth=1e-4):
    """
    Load one or more pairwise comparison CSVs produced by collect_data.py,
    concatenate them, and build the wins and ties matrices.

    Returns:
        wins : np.ndarray (n, n) — wins[i, j] = times i beat j
        ties : np.ndarray (n, n) — ties[i, j] = ties between i and j (symmetric)
        ids  : list of length n  — original player indices (sorted)
    """
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    all_ids = sorted(set(df["idx_i"].tolist() + df["idx_j"].tolist()))
    id_to_pos = {pid: pos for pos, pid in enumerate(all_ids)}
    n = len(all_ids)

    # Build idx → text lookup from both columns
    id_to_text = {**dict(zip(df["idx_i"], df["text_i"])),
                  **dict(zip(df["idx_j"], df["text_j"]))}

    wins = np.full((n, n), smooth, dtype=float)
    ties = np.full((n, n), smooth, dtype=float)
    for _, row in df.iterrows():
        # Note this indexing seem redundant, but allows for non-integer indexing in the future.
        ii = id_to_pos[row["idx_i"]]
        jj = id_to_pos[row["idx_j"]]

        wins[ii, jj] += max(row["wins_i"], 0)
        wins[jj, ii] += max(row["wins_j"], 0)

        ties[ii,jj] += max(row["ties"], 0)
        ties[jj,ii] += max(row["ties"], 0)

    return wins, ties, all_ids, id_to_text


@click.command()
@click.option("--input_path", required=True, multiple=True, type=str, help="Path to CSV file produced by collect_data.py. Can be repeated to combine multiple rounds.")
@click.option("--output_path", required=True, type=str, help="Path to write the ranked results CSV.")
@click.option("--max_iter", default=300, show_default=True, type=int, help="Maximum number of EM iterations.")
@click.option("--tol", default=1e-4, show_default=True, type=float, help="Convergence tolerance.")
def main(input_path: List[str], output_path: str, max_iter: int, tol: float):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    wins, ties, ids, id_to_text = load_matrices(input_path)
    print(f"Loaded {len(ids)} players from {len(input_path)} file(s).")

    trackio.init(project="btt-fitting", config={"input_path": input_path, "max_iter": max_iter, "tol": tol})
    s, nu = run_btt(wins, ties, max_iter=max_iter, tol=tol)
    trackio.finish()

    ranked = rank_players(s, ids)
    print(f"nu = {nu:.6f}\n")
    print(f"{'Rank':<6} {'Strength':>10}  Text")
    print("-" * 80)
    results = []
    for rank, (player, strength) in enumerate(ranked, start=1):
        text = id_to_text.get(player, "")
        print(f"{rank:<6} {strength:>10.6f}  {text}")
        results.append(dict(rank=rank, strength_btt=strength, text=id_to_text.get(player, "")))
    print()

    results = pd.DataFrame(results)
    results.to_csv(output_path, index=None)


if __name__ == "__main__":
    main()