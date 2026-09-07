"""Calculate single-sample gene-set enrichment scores and group comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from io_utils import load_config, require_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--signatures", type=Path, required=True, help="GMT file with one gene set per line.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_gmt(path: Path) -> dict[str, list[str]]:
    signatures: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split("\t")
        if len(fields) < 3:
            continue
        signatures[fields[0]] = list(dict.fromkeys(fields[2:]))
    return signatures


def ss_gsea(values: pd.Series, genes: list[str]) -> float:
    expression = values.dropna().sort_values(ascending=False)
    genes = [gene for gene in genes if gene in expression.index]
    if len(genes) < 1 or len(expression) <= len(genes):
        return np.nan
    gene_set = set(genes)
    ordered_genes = expression.index.to_list()
    weights = np.abs(expression.to_numpy(dtype=float))
    hit_weights = np.asarray([weights[index] if gene in gene_set else 0.0 for index, gene in enumerate(ordered_genes)])
    hit_total = float(hit_weights.sum())
    if hit_total <= 0:
        hit_weights = np.asarray([1.0 if gene in gene_set else 0.0 for gene in ordered_genes])
        hit_total = float(hit_weights.sum())
    miss_total = float(len(ordered_genes) - len(genes))
    running_hit = np.cumsum(hit_weights / hit_total)
    running_miss = np.cumsum(np.asarray([0.0 if gene in gene_set else 1.0 / miss_total for gene in ordered_genes]))
    running = running_hit - running_miss
    positive = float(running.max())
    negative = float(running.min())
    return positive if abs(positive) >= abs(negative) else negative


def adjust_bh(values: list[float]) -> list[float]:
    pvalues = np.asarray(values, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 1.0
    for position in range(len(order) - 1, -1, -1):
        rank = position + 1
        running = min(running, pvalues[order[position]] * len(order) / rank)
        adjusted[order[position]] = running
    return adjusted.tolist()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    bulk_cfg = config["bulk_ssgsea"]
    expression = pd.read_csv(args.expression)
    metadata = pd.read_csv(args.metadata)
    sample_col = bulk_cfg["sample_id_column"]
    group_col = bulk_cfg["group_column"]
    if expression.shape[1] < 2:
        raise ValueError("The expression matrix must contain a gene column and at least one sample column.")
    gene_col = expression.columns[0]
    expression = expression.set_index(gene_col)
    expression.index = expression.index.astype(str)
    require_columns(metadata, [sample_col, group_col], "The bulk metadata")
    sample_ids = [str(value) for value in metadata[sample_col]]
    missing_samples = sorted(set(sample_ids).difference(expression.columns.astype(str)))
    if missing_samples:
        raise ValueError(f"The expression matrix is missing metadata samples: {', '.join(missing_samples[:5])}")
    expression.columns = expression.columns.astype(str)
    expression = expression.loc[:, sample_ids]
    signatures = read_gmt(args.signatures)
    minimum = int(bulk_cfg["min_genes_per_signature"])
    scores = []
    for sample_id in sample_ids:
        row = {sample_col: sample_id}
        values = expression[sample_id]
        for name, genes in signatures.items():
            overlap = [gene for gene in genes if gene in expression.index]
            row[name] = ss_gsea(values, overlap) if len(overlap) >= minimum else np.nan
        scores.append(row)
    score_frame = pd.DataFrame(scores).merge(metadata[[sample_col, group_col]], on=sample_col, how="left")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_frame.to_csv(args.output_dir / "ssgsea_scores.csv", index=False)
    groups = [value for value in score_frame[group_col].dropna().astype(str).unique()]
    if len(groups) < 2:
        raise ValueError("At least two risk groups are required for ssGSEA comparison.")
    comparison_rows = []
    for signature in signatures:
        first = score_frame.loc[score_frame[group_col].astype(str) == groups[0], signature].dropna()
        second = score_frame.loc[score_frame[group_col].astype(str) == groups[1], signature].dropna()
        if len(first) < 1 or len(second) < 1:
            continue
        statistic, pvalue = mannwhitneyu(first, second, alternative="two-sided")
        comparison_rows.append(
            {
                "signature": signature,
                "group_1": groups[0],
                "group_2": groups[1],
                "n_group_1": len(first),
                "n_group_2": len(second),
                "u_statistic": float(statistic),
                "p_value": float(pvalue),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    if not comparison.empty:
        comparison["fdr_bh"] = adjust_bh(comparison["p_value"].tolist())
    comparison.to_csv(args.output_dir / "ssgsea_group_comparison.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
