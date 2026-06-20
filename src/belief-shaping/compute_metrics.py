import os
import sys
import click
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _load_and_suffix(path: str, suffix: str, claim_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {c: f"{c}{suffix}" for c in df.columns if c != claim_col}
    return df.rename(columns=rename)


@click.command()
@click.option("--belief_path_a", required=True, help="Path to belief CSV for condition A (e.g. original context).")
@click.option("--belief_path_b", required=True, help="Path to belief CSV for condition B (e.g. modified context).")
@click.option("--output_path", required=True, help="Path to output CSV with per-row metrics.")
@click.option("--claim_col", default="extracted_claim", show_default=True, help="Column to join the two belief files on.")
@click.option("--metadata_path", default=None, help="Optional path to metadata CSV (e.g. claim file with model column) to join for groupby analysis.")
@click.option("--group_by", default=None, help="Comma-separated column name(s) from metadata to group summary statistics by.")
@click.option("--prior_belief_path", default=None, help="Optional path to belief CSV from extract_belief.py (no context). Adds prior belief as an additional groupby dimension.")
def compute_metrics(belief_path_a, belief_path_b, output_path, claim_col, metadata_path, group_by, prior_belief_path):
    df_a = _load_and_suffix(belief_path_a, "_a", claim_col)
    df_b = _load_and_suffix(belief_path_b, "_b", claim_col)

    df = df_a.merge(df_b, on=claim_col, how="inner")

    n_before = max(len(df_a), len(df_b))
    if len(df) < n_before:
        print(f"Warning: {n_before - len(df)} rows dropped during join on '{claim_col}'.")

    # Attach metadata columns for groupby (e.g. model)
    group_cols = []
    if metadata_path and group_by:
        meta = pd.read_csv(metadata_path)[[claim_col] + group_by.split(",")]
        meta = meta.drop_duplicates(claim_col)
        df = df.merge(meta, on=claim_col, how="left")
        group_cols = group_by.split(",")

    # Attach prior (no-context) belief as an additional groupby dimension
    if prior_belief_path:
        prior = pd.read_csv(prior_belief_path)[[claim_col, "belief_answer", "belief_score"]]
        prior = prior.drop_duplicates(claim_col).rename(columns={
            "belief_answer": "prior_belief_answer",
            "belief_score": "prior_belief_score",
        })
        df = df.merge(prior, on=claim_col, how="left")
        group_cols.append("prior_belief_answer")

    # Core metrics
    df["belief_flipped"] = df["belief_answer_a"] != df["belief_answer_b"]
    df["belief_score_shift"] = (
        pd.to_numeric(df["belief_score_b"], errors="coerce")
        - pd.to_numeric(df["belief_score_a"], errors="coerce")
    )
    offset = 0
    df["certainty_direction"] = df["belief_score_shift"].apply(
        lambda x: 1 if x > offset else (-1 if x < -offset else 0)
    )

    stable = df[~df["belief_flipped"]]

    # Overall summary
    _print_section("Overall")
    print(f"  Total examples  : {len(df)}")
    print(f"  Belief flip rate: {df['belief_flipped'].mean():.1%}  ({df['belief_flipped'].sum()} / {len(df)})")

    _print_section("Flip transition matrix")
    transition = (
        df[df["belief_flipped"]]
        .groupby(["belief_answer_a", "belief_answer_b"])
        .size()
        .rename("count")
        .reset_index()
        .pivot(index="belief_answer_a", columns="belief_answer_b", values="count")
        .fillna(0)
        .astype(int)
    )
    print(transition.to_string())

    _print_section("Certainty direction (all rows)")
    direction_counts = df["certainty_direction"].map({1: "more certain", -1: "less certain", 0: "unchanged"}).value_counts()
    print(direction_counts.to_string())

    _print_section("Score shift (non-flipped rows)")
    print(f"  n = {len(stable)}")
    print(stable["belief_score_shift"].describe().to_string())
    shifts = stable["belief_score_shift"].dropna()
    if len(shifts) >= 2:
        stat, p = stats.wilcoxon(shifts, alternative="two-sided")
        print(f"\n  Wilcoxon signed-rank test: stat={stat:.3f}, p={p:.4f}")

    # Grouped summary
    if group_cols:
        _print_section(f"Grouped by: {', '.join(group_cols)}")

        flip_by_group = df.groupby(group_cols)["belief_flipped"].agg(
            n="count", flipped="sum", flip_rate="mean"
        )
        print("\nFlip rate:\n", flip_by_group.to_string())

        direction_by_group = df.groupby(group_cols + ["certainty_direction"]).size().unstack(fill_value=0)
        direction_by_group.columns = [
            {1: "more_certain", -1: "less_certain", 0: "unchanged"}.get(c, c)
            for c in direction_by_group.columns
        ]
        print("\nCertainty direction:\n", direction_by_group.to_string())

        shift_by_group = stable.groupby(group_cols)["belief_score_shift"].agg(
            ["mean", "std", "median"]
        )
        print("\nScore shift (non-flipped):\n", shift_by_group.to_string())

        def _wilcoxon_p(x):
            x = x.dropna()
            if len(x) < 2:
                return float("nan")
            return stats.wilcoxon(x, alternative="two-sided").pvalue

        p_by_group = stable.groupby(group_cols)["belief_score_shift"].apply(_wilcoxon_p).rename("wilcoxon_p")
        print("\nWilcoxon p-value (non-flipped):\n", p_by_group.to_string())

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    compute_metrics()
