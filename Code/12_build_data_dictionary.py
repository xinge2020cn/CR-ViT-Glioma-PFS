"""Build a machine-readable data dictionary from the supplied input headers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DESCRIPTION = {
    "patient_id": "De-identified patient-level key used for joins.",
    "center": "Participating center label.",
    "institution": "Institution label.",
    "imaging_cohort": "Imaging cohort used for nested-subset provenance.",
    "cohort": "Analysis cohort.",
    "subtype": "Molecular tumor subtype.",
    "omics_type": "Available transcriptomic modality in the nested subset.",
    "age": "Age at imaging, in years.",
    "sex": "Recorded sex category.",
    "preoperative_kps": "Preoperative Karnofsky Performance Status.",
    "who_grade": "Integrated WHO tumor grade category.",
    "mgmt_promoter_methylation": "MGMT promoter methylation category.",
    "extent_of_resection": "Extent of surgical resection.",
    "postoperative_radiotherapy": "Recorded postoperative radiotherapy status.",
    "postoperative_chemotherapy": "Recorded postoperative chemotherapy status.",
    "tumor_location": "Conventional MRI tumor-location category.",
    "tumor_volume_cm3": "Tumor volume in cubic centimeters.",
    "enhancing_proportion": "Ordinal fraction of the tumor core occupied by enhancing tissue; nonoverlapping with necrosis.",
    "necrotic_proportion": "Ordinal fraction of the same tumor core occupied by necrotic tissue; nonoverlapping with enhancement.",
    "enhancing_fraction_core": "Enhancing tissue fraction of tumor-core volume, from 0 to 1.",
    "necrotic_fraction_core": "Necrotic tissue fraction of tumor-core volume, from 0 to 1.",
    "other_fraction_core": "Remaining core fraction; all three fractions sum to 1.",
    "enhancing_volume_cm3": "Enhancing core volume in cubic centimeters.",
    "necrotic_volume_cm3": "Necrotic core volume in cubic centimeters.",
    "other_core_volume_cm3": "Remaining tumor-core volume in cubic centimeters.",
    "peritumoral_flair_extent": "Ordinal peritumoral T2/FLAIR abnormality category.",
    "ependymal_involvement": "Ependymal involvement category.",
    "deep_white_matter_invasion": "Deep-white-matter invasion category.",
    "enhancing_tumor_crossing_midline": "Enhancing tumor crossing the midline category.",
    "follow_up_time_months": "Follow-up duration in months.",
    "pfs_time_months": "Progression-free survival time in months.",
    "event": "PFS event indicator, where 1 denotes an event and 0 denotes censoring.",
    "progression_status": "Observed PFS status label.",
    "risk_score_3dvit": "Supplied 3D-ViT risk score used as a fixed analysis input.",
    "risk_score_3dcnn": "Supplied 3D-CNN risk score used as a fixed analysis input.",
    "risk_group_3dvit": "Supplied 3D-ViT risk-group label.",
    "risk_group_3dcnn": "Supplied 3D-CNN risk-group label.",
    "training_cutoff_3dvit": "Training-derived 3D-ViT score cutoff.",
    "training_cutoff_3dcnn": "Training-derived 3D-CNN score cutoff.",
    "vit_z": "Training-standardized 3D-ViT score.",
    "clinical_signal_z": "Supplied standardized clinicoradiologic signal.",
    "dice_similarity": "Patient-level Dice similarity coefficient.",
    "hd95_mm": "Patient-level 95th-percentile Hausdorff distance in millimeters.",
    "segmentation_qc_category": "Segmentation quality-control category.",
    "feature": "Reader-assessed conventional MRI feature key.",
    "feature_label": "Reader-assessed conventional MRI feature label.",
    "scale": "Measurement scale used for reader agreement.",
    "consensus_label": "Consensus label for the assessed feature.",
    "reader_1_rating": "Rating from reader 1.",
    "reader_2_rating": "Rating from reader 2.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    data_dir = root / "Data"
    output_path = data_dir / "data_dictionary.csv"
    rows: list[dict[str, str]] = []
    for path in sorted(data_dir.glob("*.csv")):
        if path.name == output_path.name:
            continue
        frame = pd.read_csv(path, nrows=50)
        for column in frame.columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                dtype = "numeric"
            elif pd.api.types.is_bool_dtype(series):
                dtype = "boolean"
            else:
                dtype = "categorical_or_text"
            rows.append(
                {
                    "source_table": path.name,
                    "field": column,
                    "data_type": dtype,
                    "analysis_role": "identifier" if column == "patient_id" else "input_or_derived_analysis_field",
                    "description": DESCRIPTION.get(column, "Field retained from the supplied analysis table."),
                }
            )
    dictionary = pd.DataFrame(rows).drop_duplicates(subset=["source_table", "field"])
    dictionary.to_csv(output_path, index=False)
    print(f"Wrote data dictionary: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
