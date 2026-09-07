"""Run patient-aware single-cell QC, clustering, state scoring, and optional communication analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import load_config, require_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--marker-sets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ligand-receptor", type=Path)
    parser.add_argument("--developmental-scores", type=Path)
    return parser.parse_args()


def load_marker_sets(path: Path) -> dict[str, list[str]]:
    value = load_config(path)
    if not isinstance(value, dict) or not value:
        raise ValueError("Marker sets must be a non-empty YAML mapping.")
    return {str(name): [str(gene) for gene in genes] for name, genes in value.items()}


def expression_matrix(adata):
    from scipy import sparse

    matrix = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    return np.asarray(matrix, dtype=float)


def communication_scores(adata, pair_path: Path, cluster_key: str, output_path: Path) -> None:
    pairs = pd.read_csv(pair_path)
    require_columns(pairs, ["ligand", "receptor"], "The ligand-receptor table")
    source = adata.raw if adata.raw is not None else adata
    matrix = expression_matrix(source)
    genes = {str(gene): index for index, gene in enumerate(source.var_names)}
    clusters = adata.obs[cluster_key].astype(str)
    cluster_names = sorted(clusters.unique())
    rows = []
    for _, pair in pairs.iterrows():
        ligand = str(pair["ligand"])
        receptor = str(pair["receptor"])
        if ligand not in genes or receptor not in genes:
            continue
        for sender in cluster_names:
            sender_mean = float(matrix[clusters.to_numpy() == sender, genes[ligand]].mean())
            if sender_mean <= 0:
                continue
            for receiver in cluster_names:
                receiver_mean = float(matrix[clusters.to_numpy() == receiver, genes[receptor]].mean())
                if receiver_mean <= 0:
                    continue
                rows.append(
                    {
                        "ligand": ligand,
                        "receptor": receptor,
                        "sender_cluster": sender,
                        "receiver_cluster": receiver,
                        "sender_mean_expression": sender_mean,
                        "receiver_mean_expression": receiver_mean,
                        "communication_score": float(np.sqrt(sender_mean * receiver_mean)),
                    }
                )
    pd.DataFrame(rows).sort_values("communication_score", ascending=False).to_csv(output_path, index=False)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    sc_cfg = config["single_cell"]
    try:
        import scanpy as sc
    except ImportError as exc:
        raise SystemExit("scanpy and anndata are required for single-cell analysis.") from exc
    adata = sc.read_h5ad(args.input)
    patient_col = sc_cfg["patient_id_column"]
    risk_col = sc_cfg["risk_group_column"]
    require_columns(adata.obs, [patient_col, risk_col], "The single-cell observation metadata")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adata.var["mt"] = [str(gene).upper().startswith("MT-") for gene in adata.var_names]
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[adata.obs["n_genes_by_counts"] >= int(sc_cfg["min_genes_per_cell"])].copy()
    adata = adata[adata.obs["pct_counts_mt"] <= float(sc_cfg["max_mitochondrial_percent"])].copy()
    sc.pp.filter_genes(adata, min_cells=int(sc_cfg["min_cells_per_gene"]))
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=int(sc_cfg["highly_variable_genes"]), flavor="cell_ranger")
    if adata.var["highly_variable"].sum() >= 20:
        adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    n_pcs = min(int(sc_cfg["pca_components"]), max(2, adata.n_obs - 1), max(2, adata.n_vars - 1))
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    n_neighbors = min(int(sc_cfg["neighbors"]), max(2, adata.n_obs - 1))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.leiden(adata, resolution=float(sc_cfg["leiden_resolution"]), key_added="cell_cluster")
    sc.tl.rank_genes_groups(adata, groupby="cell_cluster", method=str(sc_cfg["marker_test"]))
    marker_result = adata.uns["rank_genes_groups"]
    marker_rows = []
    for group in marker_result["names"].dtype.names:
        for rank, gene in enumerate(marker_result["names"][group]):
            marker_rows.append(
                {
                    "cluster": str(group),
                    "rank": rank + 1,
                    "gene": str(gene),
                    "score": float(marker_result["scores"][group][rank]),
                    "p_value": float(marker_result["pvals"][group][rank]),
                    "p_value_adj": float(marker_result["pvals_adj"][group][rank]),
                }
            )
    pd.DataFrame(marker_rows).to_csv(args.output_dir / "cluster_markers.csv", index=False)
    marker_sets = load_marker_sets(args.marker_sets)
    score_columns = []
    for state, genes in marker_sets.items():
        available = [gene for gene in genes if gene in adata.var_names]
        if not available:
            continue
        column = f"state_score_{state}"
        sc.tl.score_genes(adata, available, score_name=column, use_raw=False)
        score_columns.append(column)
    if score_columns:
        adata.obs["neural_lineage_state"] = adata.obs[score_columns].idxmax(axis=1).str.replace("state_score_", "", regex=False)
    else:
        adata.obs["neural_lineage_state"] = "unassigned"
    state_counts = (
        adata.obs.groupby([patient_col, risk_col, "cell_cluster", "neural_lineage_state"], observed=True)
        .size()
        .reset_index(name="cell_count")
    )
    patient_totals = state_counts.groupby([patient_col, risk_col], observed=True)["cell_count"].transform("sum")
    state_counts["cell_fraction"] = state_counts["cell_count"] / patient_totals
    state_counts.to_csv(args.output_dir / "patient_state_abundance.csv", index=False)
    if args.developmental_scores:
        developmental = pd.read_csv(args.developmental_scores)
        require_columns(developmental, ["cell_id", "developmental_potential"], "The developmental-potential table")
        developmental.to_csv(args.output_dir / "developmental_potential.csv", index=False)
    if args.ligand_receptor:
        communication_scores(adata, args.ligand_receptor, "cell_cluster", args.output_dir / "ligand_receptor_scores.csv")
    if bool(sc_cfg["run_cnv_if_available"]):
        try:
            import infercnvpy as cnv

            if hasattr(cnv, "tl") and hasattr(cnv.tl, "infercnv"):
                reference_group = str(adata.obs["cell_cluster"].value_counts().index[0])
                cnv.tl.infercnv(adata, reference_key="cell_cluster", reference_cat=[reference_group])
                if "cnv_score" in adata.obs:
                    adata.obs[[patient_col, risk_col, "cell_cluster", "cnv_score"]].to_csv(args.output_dir / "cnv_scores.csv", index=False)
        except Exception as exc:
            (args.output_dir / "cnv_status.txt").write_text(f"CNV analysis was not run: {exc}\n", encoding="utf-8")
    adata.write_h5ad(args.output_dir / "processed_single_cell.h5ad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
