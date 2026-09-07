"""Run public-release checks for code, inputs, and reproducibility metadata."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


HAN = re.compile(r"[\u3400-\u9fff]")
FORBIDDEN_GENERATOR_FIELDS = {"outcome_generation_seed", "outcome_generation_scale", "outcome_generation_shape"}
REQUIRED_CODE = {
    "00_validate_inputs.py",
    "01_run_primary_analysis.R",
    "02_build_pfs_summary.R",
    "03_audit_model_scores.R",
    "04_audit_lasso_selection.R",
    "05_build_multivariable_cox_results.R",
    "06_build_km_statistics.R",
    "07_build_segmentation_summary.py",
    "08_build_reader_agreement.py",
    "09_build_figure_s1.py",
    "10_build_figure_components.py",
    "11_audit_public_release.py",
    "12_build_data_dictionary.py",
}
REQUIRED_UPSTREAM = {
    "01_preprocess_mri.py",
    "02_prepare_nnunet_dataset.py",
    "03_train_nnunet.py",
    "04_train_survival_models.py",
    "05_fit_crvit_fusion.R",
    "06_generate_attributions.py",
    "07_run_bulk_ssgsea.py",
    "08_run_single_cell_analysis.py",
    "09_validate_upstream_configuration.py",
    "io_utils.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    qa_dir = root / "QA"
    qa_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, str]] = []

    def record(check: str, passed: bool, detail: str) -> None:
        checks.append({"check": check, "status": "PASS" if passed else "FAIL", "detail": detail})

    code_dir = root / "Code"
    code_names = {path.name for path in code_dir.iterdir() if path.is_file()}
    missing = sorted(REQUIRED_CODE.difference(code_names))
    record("Required analysis scripts are present", not missing, ", ".join(missing) or "all present")

    upstream_dir = root / "Upstream"
    upstream_names = {path.name for path in upstream_dir.iterdir() if path.is_file()}
    missing_upstream = sorted(REQUIRED_UPSTREAM.difference(upstream_names))
    record("Required upstream scripts are present", not missing_upstream, ", ".join(missing_upstream) or "all present")

    code_files = list(code_dir.rglob("*.py")) + list(code_dir.rglob("*.R"))
    code_files += list(upstream_dir.rglob("*.py")) + list(upstream_dir.rglob("*.R"))
    han_files = [str(path.relative_to(root)) for path in code_files if HAN.search(path.read_text(encoding="utf-8"))]
    record("R and Python source files contain no Han characters", not han_files, ", ".join(han_files) or "none")

    data_dir = root / "Data"
    data_frames = []
    for path in sorted(data_dir.glob("*.csv")):
        frame = pd.read_csv(path, nrows=0)
        data_frames.append((path.name, set(frame.columns)))
    forbidden = sorted({field for _, columns in data_frames for field in columns if field in FORBIDDEN_GENERATOR_FIELDS})
    record("Bundled data omit generator metadata fields", not forbidden, ", ".join(forbidden) or "none")

    required_docs = [root / "README.md", root / "LICENSE_PENDING.md", root / "requirements.txt", root / "r-packages.txt"]
    record("Public-release metadata files are present", all(path.exists() for path in required_docs), "README, license status, and dependency lists")

    for directory in (root / "Results", root / "QA", root / "Figures"):
        directory.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(checks)
    output.to_csv(qa_dir / "public_release_audit.csv", index=False)
    print(output.to_string(index=False))
    return 0 if (output["status"] == "PASS").all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
