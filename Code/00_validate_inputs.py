"""Validate the patient-level inputs required by the public analysis workflow."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np

ORDINAL_LEVELS = ["None/minimal (<=5%)", "Mild (>5-33%)", "Moderate (>33-67%)", "Extensive (>67%)"]


def composition_checks(frame: pd.DataFrame) -> dict[str, bool]:
    """Check disjoint enhancing/necrotic fractions with a shared core denominator."""
    e = frame.enhancing_proportion.map(dict(zip(ORDINAL_LEVELS, range(4))))
    n = frame.necrotic_proportion.map(dict(zip(ORDINAL_LEVELS, range(4))))
    lower = {0: 0.0, 1: 0.05, 2: 0.33, 3: 0.67}
    result = {"Known composition categories": bool(e.notna().all() and n.notna().all()),
              "Joint composition categories are feasible": bool((e.map(lower) + n.map(lower) < 1).all())}
    columns = ["enhancing_fraction_core", "necrotic_fraction_core", "other_fraction_core"]
    if any(c in frame for c in columns):
        result["Complete fraction triplet"] = all(c in frame for c in columns)
        if not result["Complete fraction triplet"]: return result
        values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
        result["Core fractions are finite and sum to one"] = bool(np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all() and np.allclose(values.sum(axis=1), 1, atol=1e-8, rtol=0))
        for col, cat in [(columns[0], e), (columns[1], n)]:
            coded = np.searchsorted([0.05, 0.33, 0.67], frame[col], side="left")
            result[f"Category matches {col}"] = bool((coded == cat).all())
        for fraction, volume in zip(columns, ["enhancing_volume_cm3", "necrotic_volume_cm3", "other_core_volume_cm3"]):
            if volume in frame:
                result[f"Volume matches {fraction}"] = bool(np.allclose(frame[volume], frame[fraction] * frame.tumor_volume_cm3, atol=1e-7, rtol=1e-7))
    return result


MAIN_REQUIRED = {
    "patient_id",
    "center",
    "institution",
    "subtype",
    "age",
    "sex",
    "preoperative_kps",
    "who_grade",
    "mgmt_promoter_methylation",
    "extent_of_resection",
    "tumor_location",
    "tumor_volume_cm3",
    "enhancing_proportion",
    "necrotic_proportion",
    "peritumoral_flair_extent",
    "ependymal_involvement",
    "deep_white_matter_invasion",
    "enhancing_tumor_crossing_midline",
    "follow_up_time_months",
    "pfs_time_months",
    "event",
    "progression_status",
    "risk_score_3dvit",
    "risk_score_3dcnn",
    "cohort",
}

SUBSET_REQUIRED = {"patient_id", "cohort", "subtype", "omics_type"}
SEGMENTATION_REQUIRED = {"patient_id", "cohort", "dice_similarity", "hd95_mm"}
RATING_REQUIRED = {
    "patient_id",
    "cohort",
    "feature",
    "feature_label",
    "scale",
    "reader_1_rating",
    "reader_2_rating",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Package root containing Data, Results, QA, and Figures.",
    )
    return parser.parse_args()


def check_columns(frame: pd.DataFrame, required: set[str], label: str) -> tuple[bool, str]:
    missing = sorted(required.difference(frame.columns))
    if missing:
        return False, f"{label}: missing columns: {', '.join(missing)}"
    return True, f"{label}: required columns present"


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    data_dir = root / "Data"
    qa_dir = root / "QA"
    qa_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, str]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    paths = {
        "main": data_dir / "imaging_3center_patient_level.csv",
        "subset": data_dir / "center1_gbm_multiomics_subset_patient_level.csv",
        "segmentation": data_dir / "segmentation_metrics_patient_level.csv",
        "ratings": data_dir / "reader_ratings_long.csv",
    }
    for label, path in paths.items():
        record(f"File exists: {label}", path.exists(), str(path.relative_to(root)))

    if not all(path.exists() for path in paths.values()):
        output = pd.DataFrame(checks)
        output.to_csv(qa_dir / "input_validation.csv", index=False)
        print(output.to_string(index=False))
        return 1

    main_data = pd.read_csv(paths["main"])
    subset_data = pd.read_csv(paths["subset"])
    segmentation = pd.read_csv(paths["segmentation"])
    ratings = pd.read_csv(paths["ratings"])

    for label, frame, required in (
        ("main patient table", main_data, MAIN_REQUIRED),
        ("multi-omics subset table", subset_data, SUBSET_REQUIRED),
        ("segmentation table", segmentation, SEGMENTATION_REQUIRED),
        ("reader-rating table", ratings, RATING_REQUIRED),
    ):
        passed, detail = check_columns(frame, required, label)
        record(f"Columns: {label}", passed, detail)

    main_ids = set(main_data.get("patient_id", pd.Series(dtype=str)).astype(str))
    subset_ids = set(subset_data.get("patient_id", pd.Series(dtype=str)).astype(str))

    record(
        "Main patient identifiers are unique",
        "patient_id" in main_data and not main_data["patient_id"].duplicated().any(),
        f"rows={len(main_data)}",
    )
    record(
        "Subset identifiers are unique",
        "patient_id" in subset_data and not subset_data["patient_id"].duplicated().any(),
        f"rows={len(subset_data)}",
    )
    record(
        "Multi-omics subset is nested in the main table",
        subset_ids.issubset(main_ids),
        f"subset={len(subset_ids)}, overlap={len(subset_ids.intersection(main_ids))}",
    )

    if MAIN_REQUIRED.issubset(main_data.columns):
        finite_numeric = main_data[["age", "preoperative_kps", "tumor_volume_cm3", "pfs_time_months"]].apply(
            pd.to_numeric, errors="coerce"
        ).notna().all().all()
        record("Required numeric fields are numeric", bool(finite_numeric), "age, KPS, volume, and PFS time")
        valid_event = set(pd.to_numeric(main_data["event"], errors="coerce").dropna().unique()).issubset({0, 1})
        record("Event indicator is binary", valid_event, "allowed values: 0 and 1")
        positive_time = (pd.to_numeric(main_data["pfs_time_months"], errors="coerce") > 0).all()
        record("PFS times are positive", bool(positive_time), "pfs_time_months > 0")
        no_missing = not main_data[sorted(MAIN_REQUIRED)].isna().any().any()
        record("Required main-table fields are complete", no_missing, "no missing values in required columns")
        for name, passed in composition_checks(main_data).items():
            record(name, passed, "Enhancing and necrotic tissue share the tumor-core denominator")

        if subset_ids.issubset(main_ids):
            shared = sorted(set(subset_data.columns).intersection(main_data.columns) - {"patient_id", "cohort", "cohort_label", "cohort_period", "omics_type"})
            master = main_data.set_index("patient_id").loc[subset_data.patient_id]
            for column in shared:
                left, right = master[column].reset_index(drop=True), subset_data[column].reset_index(drop=True)
                equal = np.allclose(left, right, atol=1e-10, rtol=1e-10, equal_nan=True) if pd.api.types.is_numeric_dtype(left) else left.fillna("").astype(str).equals(right.fillna("").astype(str))
                record(f"Subset agrees with master: {column}", bool(equal), "patient-ID matched")
            record("Subset is from Institution I temporal GBM", bool(master.institution.eq("Institution I").all() and master.cohort.eq("Temporal validation cohort").all() and master.subtype.eq("GBM").all()), "Nested subset, not an independent validation cohort")

    if SUBSET_REQUIRED.issubset(subset_data.columns):
        subset_is_gbm = subset_data["subtype"].astype(str).eq("GBM").all()
        record("Multi-omics subset is GBM-only", bool(subset_is_gbm), "subtype == GBM")

    if SEGMENTATION_REQUIRED.issubset(segmentation.columns):
        record(
            "Segmentation identifiers are unique",
            not segmentation["patient_id"].duplicated().any(),
            f"rows={len(segmentation)}",
        )
        numeric = segmentation[["dice_similarity", "hd95_mm"]].apply(pd.to_numeric, errors="coerce")
        record("Segmentation metrics are numeric", bool(numeric.notna().all().all()), "Dice and HD95")

    if RATING_REQUIRED.issubset(ratings.columns):
        duplicate_rating_keys = ratings.duplicated(subset=["patient_id", "feature"]).any()
        record("Reader-rating keys are unique", not duplicate_rating_keys, "one row per patient and feature")
        if "consensus_label" in ratings and not main_data.patient_id.duplicated().any():
            master=main_data.set_index("patient_id")
            for feature, group in ratings.groupby("feature"):
                valid=feature in master and set(group.patient_id).issubset(master.index)
                equal=valid and master.loc[group.patient_id,feature].reset_index(drop=True).astype(str).equals(group.consensus_label.reset_index(drop=True).astype(str))
                record(f"Reader consensus matches master: {feature}", bool(equal), "patient-ID matched")
        if not duplicate_rating_keys:
            for reader in ["reader_1_rating", "reader_2_rating"]:
                pair=ratings.pivot(index="patient_id",columns="feature",values=reader)
                if {"enhancing_proportion", "necrotic_proportion"}.issubset(pair):
                    for name, passed in composition_checks(pair).items(): record(f"{reader}: {name}", passed, "nonoverlapping compartments")

    han_pattern = re.compile(r"[\u3400-\u9fff]")
    code_dir = Path(__file__).resolve().parent
    code_files = list(code_dir.rglob("*.py")) + list(code_dir.rglob("*.R"))
    code_with_han = [str(path.relative_to(code_dir)) for path in code_files if han_pattern.search(path.read_text(encoding="utf-8"))]
    record("Analysis code contains no Han characters", not code_with_han, ", ".join(code_with_han) or "none")

    output = pd.DataFrame(checks)
    output.to_csv(qa_dir / "input_validation.csv", index=False)
    print(output.to_string(index=False))
    return 0 if (output["status"] == "PASS").all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
