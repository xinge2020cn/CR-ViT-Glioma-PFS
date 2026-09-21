"""Check input-to-output linkage, frozen thresholds and analysis consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def same_values(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Compare aligned records without requiring identical CSV-inferred dtypes."""
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, rtol=1e-10, atol=1e-10)
        return True
    except AssertionError:
        return False


def audit(root: Path) -> pd.DataFrame:
    """Validate computed values without imposing target clinical results."""
    checks = []

    def check(name: str, passed: bool) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL"})

    master = pd.read_csv(root / "Data/imaging_3center_patient_level.csv").set_index("patient_id")
    pred = pd.read_csv(root / "Results/clinicoradiologic_crvit_patient_predictions.csv").set_index("patient_id")
    ids_ok = master.index.is_unique and pred.index.is_unique and set(master.index) == set(pred.index)
    check("One prediction per input patient", ids_ok)
    if ids_ok:
        columns = [c for c in pred if c in master]
        check("Predictions use current patient records",
              same_values(master.loc[pred.index, columns], pred[columns]))
    for model in ("clinic", "cnn", "vit", "crvit"):
        values = pred[[f"{model}_surv_{t}" for t in (6, 12, 18, 24, 30, 36)]].to_numpy()
        check(f"{model}: bounded and nonincreasing survival",
              np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
              and (np.diff(values, axis=1) <= 1e-12).all())
    for subtype, group in pred.groupby("subtype"):
        train = group.loc[group.cohort.eq("Training cohort")]
        check(f"{subtype}: nonempty training set", len(train) > 0)
        for model in ("clinic", "vit", "crvit"):
            score = f"{model}_km_score"
            cutoff = float(train[score].median())
            check(f"{subtype} {model}: frozen training median",
                  np.allclose(group[f"{model}_cutoff"], cutoff, atol=1e-10, rtol=1e-10))
            expected = np.where(group[score] >= group[f"{model}_cutoff"], "High risk", "Low risk")
            check(f"{subtype} {model}: groups match stored scores",
                  np.array_equal(expected, group[f"{model}_risk_group"].to_numpy()))
    selection = pd.read_csv(root / "Results/clinicoradiologic_selected_predictors.csv")
    rules = pd.read_csv(root / "QA/lasso_lambda_rule_audit.csv")
    for subtype, selected in selection.groupby("subtype"):
        rule = rules.loc[rules.subtype.eq(subtype) & rules.type_measure.eq("deviance")
                         & rules.lambda_rule.eq("lambda.min")]
        check(f"{subtype}: selection agrees with independent LASSO audit",
              len(rule) == 1 and set(selected.model_variable) ==
              set(str(rule.iloc[0].selected_predictors).split("; ")))
    for name in ("input_validation.csv",):
        frame = pd.read_csv(root / "QA" / name)
        check(f"Prerequisite {name}", frame.status.eq("PASS").all())
    return pd.DataFrame(checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    checks = audit(root)
    qa = root / "QA"
    qa.mkdir(parents=True, exist_ok=True)
    checks.to_csv(qa / "analysis_output_integrity.csv", index=False)
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for folder in ("Data", "Results") for p in sorted((root / folder).glob("*.csv"))}
    (qa / "analysis_file_manifest.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
    print(checks.to_string(index=False))
    return 0 if checks.status.eq("PASS").all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
