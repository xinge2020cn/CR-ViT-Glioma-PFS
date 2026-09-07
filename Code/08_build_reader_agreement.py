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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def encode_ratings(values: pd.Series, scale: str) -> np.ndarray:
    values = values.astype(str)
    if scale == "Ordinal":
        level_map = {value: index for index, value in enumerate(ORDINAL_LEVELS)}
        unknown = sorted(set(values).difference(level_map))
        if unknown:
            raise ValueError(f"Unknown ordinal levels: {', '.join(unknown)}")
        return values.map(level_map).to_numpy(int)
    levels = {value: index for index, value in enumerate(sorted(values.unique()))}
    return values.map(levels).to_numpy(int)


def agreement(a: np.ndarray, b: np.ndarray, scale: str) -> tuple[float, float]:
    observed = float(np.mean(a == b))
    weights = "quadratic" if scale == "Ordinal" else None
    kappa = float(cohen_kappa_score(a, b, weights=weights))
    return observed, kappa


def bootstrap_interval(a: np.ndarray, b: np.ndarray, scale: str, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(a) < 2:
        point = agreement(a, b, scale)[1]
        return point, point
    estimates = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(a), size=len(a))
        estimates[index] = agreement(a[sample], b[sample], scale)[1]
    return tuple(np.percentile(estimates, [2.5, 97.5]).astype(float))


def main() -> int:
    args = parse_args()
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
    for feature, part in data.groupby("feature", sort=False):
        part = part.reset_index(drop=True)
        scale = str(part["scale"].iloc[0])
        a = encode_ratings(part["reader_1_rating"], scale)
        b = encode_ratings(part["reader_2_rating"], scale)
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
