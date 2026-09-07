"""Summarize supplied patient-level segmentation metrics without generating data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COHORTS = ["Training cohort", "Temporal validation cohort", "Spatial validation cohort"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def percentile_ci(values: np.ndarray, statistic: str, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        point = float(values[0]) if values.size else np.nan
        return point, point
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    sampled = values[indices]
    if statistic == "mean":
        estimates = sampled.mean(axis=1)
    elif statistic == "median":
        estimates = np.median(sampled, axis=1)
    else:
        raise ValueError(f"Unsupported statistic: {statistic}")
    return tuple(np.percentile(estimates, [2.5, 97.5]).astype(float))


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    data_path = root / "Data" / "segmentation_metrics_patient_level.csv"
    results_dir = root / "Results"
    qa_dir = root / "QA"
    results_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)
    required = {"patient_id", "cohort", "dice_similarity", "hd95_mm"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing segmentation columns: {', '.join(missing)}")

    data["dice_similarity"] = pd.to_numeric(data["dice_similarity"], errors="raise")
    data["hd95_mm"] = pd.to_numeric(data["hd95_mm"], errors="raise")
    if data["patient_id"].duplicated().any():
        raise ValueError("Segmentation input must contain one row per patient.")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | str]] = []
    for cohort in COHORTS:
        part = data.loc[data["cohort"].eq(cohort)].copy()
        if part.empty:
            raise ValueError(f"No segmentation records found for {cohort}.")
        dice = part["dice_similarity"].to_numpy(float)
        hd95 = part["hd95_mm"].to_numpy(float)
        dice_ci = percentile_ci(dice, "mean", args.bootstrap, rng)
        hd_ci = percentile_ci(hd95, "median", args.bootstrap, rng)
        rows.append(
            {
                "cohort": cohort,
                "n": int(len(part)),
                "dice_mean": float(np.mean(dice)),
                "dice_sd": float(np.std(dice, ddof=1)),
                "dice_mean_ci_low": dice_ci[0],
                "dice_mean_ci_high": dice_ci[1],
                "dice_median": float(np.median(dice)),
                "dice_q1": float(np.percentile(dice, 25)),
                "dice_q3": float(np.percentile(dice, 75)),
                "hd95_median_mm": float(np.median(hd95)),
                "hd95_q1_mm": float(np.percentile(hd95, 25)),
                "hd95_q3_mm": float(np.percentile(hd95, 75)),
                "hd95_median_ci_low_mm": hd_ci[0],
                "hd95_median_ci_high_mm": hd_ci[1],
                "dice_below_0_80_n": int(np.sum(dice < 0.80)),
                "dice_below_0_80_pct": float(np.mean(dice < 0.80) * 100),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "segmentation_performance.csv", index=False)
    audit = pd.DataFrame(
        [
            {"item": "input_rows", "value": len(data)},
            {"item": "unique_patient_ids", "value": data["patient_id"].nunique()},
            {"item": "bootstrap_resamples", "value": args.bootstrap},
            {"item": "seed", "value": args.seed},
        ]
    )
    audit.to_csv(qa_dir / "segmentation_summary_audit.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
