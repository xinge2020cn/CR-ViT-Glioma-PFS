"""Calculate rank-based ssGSEA and patient-level risk-group comparisons."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, false_discovery_control

from io_utils import load_config, require_columns
from omics_utils import attach_risk


def read_gmt(path: Path) -> dict[str, list[str]]:
    signatures = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.rstrip().split("\t")
        if len(fields) < 3 or fields[0] in signatures:
            raise ValueError("GMT signatures must be nonempty and uniquely named.")
        signatures[fields[0]] = list(dict.fromkeys(fields[2:]))
    if not signatures:
        raise ValueError("No signatures were supplied.")
    return signatures


def calculate_scores(expression: pd.DataFrame, signatures: dict, config: dict) -> pd.DataFrame:
    import gseapy as gp
    if expression.index.has_duplicates or expression.columns.has_duplicates:
        raise ValueError("Expression genes and sample identifiers must be unique.")
    if not np.isfinite(expression.to_numpy(dtype=float)).all():
        raise ValueError("Expression contains missing or nonfinite values.")
    cfg = config["bulk_ssgsea"]
    result = gp.ssgsea(
        data=expression, gene_sets=signatures, outdir=None,
        sample_norm_method=cfg["sample_normalization"], correl_norm_type="rank",
        weight=float(cfg["weight"]), min_size=int(cfg["min_genes_per_signature"]),
        max_size=int(cfg["max_genes_per_signature"]), permutation_num=0,
        no_plot=True, threads=1, seed=int(config["project"]["random_seed"]),
    ).res2d
    scores = result.pivot(index="Name", columns="Term", values="NES").astype(float)
    if set(scores.columns) != set(signatures):
        missing = sorted(set(signatures).difference(scores.columns))
        raise ValueError(f"Signatures failed the overlap-size filter: {missing}")
    return scores.loc[expression.columns]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "expression", "metadata", "signatures", "risk-manifest", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    sample_col = config["bulk_ssgsea"]["sample_id_column"]
    expression = pd.read_csv(args.expression, index_col=0)
    expression.index = expression.index.astype(str)
    expression.columns = expression.columns.astype(str)
    metadata = pd.read_csv(args.metadata, dtype={sample_col: str, "patient_id": str})
    require_columns(metadata, [sample_col, "patient_id"], "The bulk metadata")
    if metadata[[sample_col, "patient_id"]].isna().any().any() or any(
            metadata[c].duplicated().any() for c in (sample_col, "patient_id")):
        raise ValueError("Bulk analysis requires one unique expression sample per tumor.")
    risk = pd.read_csv(args.risk_manifest, dtype={"patient_id": str})
    metadata = attach_risk(metadata, risk, "Bulk transcriptomics")
    if set(metadata.vit_risk_group) != {"Low risk", "High risk"}:
        raise ValueError("Both final ViT risk groups are required.")
    missing = set(metadata[sample_col]).difference(expression.columns)
    if missing:
        raise ValueError(f"Missing expression samples: {sorted(missing)[:5]}")
    expression = expression.loc[:, metadata[sample_col]]
    scores = calculate_scores(expression, read_gmt(args.signatures), config)
    scores.index.name = sample_col
    result = scores.reset_index().merge(metadata, on=sample_col, validate="one_to_one")
    comparisons = []
    for signature in scores:
        low = result.loc[result.vit_risk_group.eq("Low risk"), signature]
        high = result.loc[result.vit_risk_group.eq("High risk"), signature]
        test = mannwhitneyu(low, high, alternative="two-sided", method="asymptotic")
        comparisons.append({"signature": signature, "n_low": len(low), "n_high": len(high),
                            "median_low": low.median(), "median_high": high.median(),
                            "u_statistic": test.statistic, "p_value": test.pvalue})
    comparisons = pd.DataFrame(comparisons)
    comparisons["fdr_bh"] = false_discovery_control(comparisons.p_value, method="bh")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "ssgsea_scores.csv", index=False)
    comparisons.to_csv(args.output_dir / "ssgsea_group_comparison.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
