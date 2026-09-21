"""Check standalone modules without creating patient data in the repository."""
import importlib.util
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name[:-3], ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseTests(unittest.TestCase):
    def test_import_figure_code_does_not_load_study_data(self):
        load("10_build_figure_components.py")

    def test_obsolete_grouping_detected(self):
        module = load("00_validate_inputs.py")
        self.assertEqual(module.obsolete_group_fields(pd.DataFrame(columns=["risk_group_3dvit"])),
                         ["risk_group_3dvit"])

    def test_nominal_readers_share_vocabulary(self):
        module = load("08_build_reader_agreement.py")
        one = pd.Series(["A", "B", "A"])
        two = pd.Series(["B", "C", "B"])
        levels = sorted(set(one) | set(two))
        self.assertFalse((module.encode_ratings(one, "Nominal", levels) ==
                          module.encode_ratings(two, "Nominal", levels)).any())

    def test_omics_manifest_uses_only_final_groups(self):
        module = load("15_build_omics_risk_manifest.py")
        pred = pd.DataFrame(dict(patient_id=["a", "b"], subtype=["GBM"] * 2,
            vit_km_score=[-1., 1.], vit_cutoff=[0., 0.], vit_risk_group=["Low risk", "High risk"]))
        subset = pd.DataFrame(dict(patient_id=["a", "b"],
            omics_type=["Single-cell RNA-seq", "Bulk transcriptomics"], risk_group_3dvit=["wrong"] * 2))
        result = module.build_manifest(pred, subset)
        self.assertNotIn("risk_group_3dvit", result)
        self.assertEqual(result.vit_risk_group.tolist(), ["Low risk", "High risk"])


if __name__ == "__main__":
    unittest.main()
