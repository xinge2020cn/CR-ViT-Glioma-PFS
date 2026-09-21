"""Compute two-reader agreement from supplied patient-level ratings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


ORDINAL_LEVELS = [
    "None/minimal (<=5%)",
    "Mild (>5-33%)",
    "Moderate (>33-67%)",
    "Extensive (>67%)",
]
FEATURE_ORDER = [
    "tumor_location", "enhancing_proportion", "necrotic_proportion",
    "peritumoral_flair_extent", "ependymal_involvement",
    "deep_white_matter_invasion", "enhancing_tumor_crossing_midline",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260931)
    return parser.parse_args()


def encode_ratings(values: pd.Series, scale: str, levels: list[str] | None = None) -> np.ndarray:
    if values.isna().any():
        raise ValueError("Reader ratings must be complete.")
    values = values.astype(str)
    if scale == "Ordinal":
        level_map = {value: index for index, value in enumerate(ORDINAL_LEVELS)}
        unknown = sorted(set(values).difference(level_map))
        if unknown:
            raise ValueError(f"Unknown ordinal levels: {', '.join(unknown)}")
        return values.map(level_map).to_numpy(int)
    if scale != "Nominal":
        raise ValueError(f"Unknown measurement scale: {scale}")
    if levels is None:
        raise ValueError("Nominal ratings require a shared category vocabulary for both readers.")
    level_map = {value: index for index, value in enumerate(levels)}
    if set(values).difference(level_map):
        raise ValueError("Reader ratings contain categories outside the shared vocabulary.")
    return values.map(level_map).to_numpy(int)


def agreement(a: np.ndarray, b: np.ndarray, scale: str) -> tuple[float, float]:
    observed = float(np.mean(a == b))
    weights = "quadratic" if scale == "Ordinal" else None
    labels = np.arange(len(ORDINAL_LEVELS)) if scale == "Ordinal" else None
    kappa = float(cohen_kappa_score(a, b, weights=weights, labels=labels))
    return observed, kappa


def bootstrap_interval(a: np.ndarray, b: np.ndarray, scale: str, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(a) < 2:
        point = agreement(a, b, scale)[1]
        return point, point
    estimates = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(a), size=len(a))
        estimates[index] = agreement(a[sample], b[sample], scale)[1]
    estimates = estimates[np.isfinite(estimates)]
    if not estimates.size:
        raise ValueError("Kappa is undefined in all bootstrap samples.")
    return tuple(np.percentile(estimates, [2.5, 97.5]).astype(float))


def main() -> int:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("Bootstrap resamples must be positive.")
    root = args.root.resolve()
    data_path = root / "Data" / "reader_ratings_long.csv"
    results_dir = root / "Results"
    qa_dir = root / "QA"
    results_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)
    required = {
        "patient_id",
        "feature",
        "feature_label",
        "scale",
        "reader_1_rating",
        "reader_2_rating",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing reader-rating columns: {', '.join(missing)}")
    if data.duplicated(subset=["patient_id", "feature"]).any():
        raise ValueError("Reader-rating input must contain one row per patient and feature.")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | str]] = []
    features = [feature for feature in FEATURE_ORDER if feature in set(data.feature)]
    features += sorted(set(data.feature).difference(features))
    for feature in features:
        part = data.loc[data.feature.eq(feature)]
        part = part.reset_index(drop=True)
        if part["scale"].nunique() != 1:
            raise ValueError(f"Inconsistent measurement scales for {feature}.")
        scale = str(part["scale"].iloc[0])
        levels = sorted(set(part.reader_1_rating.dropna().astype(str)) |
                        set(part.reader_2_rating.dropna().astype(str)))
        a = encode_ratings(part["reader_1_rating"], scale, levels)
        b = encode_ratings(part["reader_2_rating"], scale, levels)
        observed, kappa = agreement(a, b, scale)
        ci_low, ci_high = bootstrap_interval(a, b, scale, args.bootstrap, rng)
        ordinal = scale == "Ordinal"
        rows.append(
            {
                "feature": feature,
                "feature_label": str(part["feature_label"].iloc[0]),
                "scale": scale,
                "agreement_metric": "Quadratic weighted Cohen kappa" if ordinal else "Cohen kappa",
                "metric_symbol": "κw" if ordinal else "κ",
                "n": int(len(part)),
                "observed_agreement": observed,
                "kappa": kappa,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bootstrap_resamples": args.bootstrap,
            }
        )

    output = pd.DataFrame(rows)
    columns = ["observed_agreement", "kappa", "ci_low", "ci_high"]
    output[columns] = output[columns].round(4)
    output.to_csv(results_dir / "interobserver_agreement.csv", index=False)
    pd.DataFrame(
        [
            {"item": "input_rows", "value": len(data)},
            {"item": "unique_patients", "value": data["patient_id"].nunique()},
            {"item": "features", "value": data["feature"].nunique()},
            {"item": "bootstrap_resamples", "value": args.bootstrap},
            {"item": "seed", "value": args.seed},
        ]
    ).to_csv(qa_dir / "reader_agreement_audit.csv", index=False)
    print(output.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
