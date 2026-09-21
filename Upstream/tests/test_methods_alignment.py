"""Generated-fixture checks of segmentation and post-model omics invariants."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import load_config
from omics_utils import attach_risk, validate_risk_manifest


def load_module(name):
    spec = importlib.util.spec_from_file_location(name[:-3], ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SegmentationTests(unittest.TestCase):
    def test_every_patient_held_out_once(self):
        module = load_module("03_train_nnunet.py")
        names = [f"case_{i}" for i in range(17)]
        splits = module.make_splits(names, 7)
        self.assertEqual(sorted(x for fold in splits for x in fold["val"]), sorted(names))
        for fold in splits:
            self.assertFalse(set(fold["train"]) & set(fold["val"]))
        self.assertEqual(splits, module.make_splits(list(reversed(names)), 7))

    def test_five_training_commands_and_frozen_ensemble(self):
        module = load_module("03_train_nnunet.py")
        config = load_config(ROOT / "config/article_defaults.yml")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "imagesTs").mkdir()
            commands = module.build_commands(config, path, path / "results")
        self.assertEqual([c[3] for c in commands if c[0] == "nnUNetv2_train"], list("01234"))
        self.assertEqual(commands[-1][-2:], ["-chk", "checkpoint_final.pth"])
        self.assertNotIn("all", [token for command in commands for token in command])

    def test_physical_mask_metrics(self):
        module = load_module("11_evaluate_segmentation.py")
        a = np.zeros((5, 5, 5), bool)
        b = a.copy()
        a[1, 2, 2], b[2, 2, 2] = True, True
        self.assertEqual(module.mask_metrics(a, a, (5, 1, 1)), (1., 0.))
        self.assertEqual(module.mask_metrics(a, b, (5, 1, 1)), (0., 5.))
        self.assertTrue(np.isinf(module.mask_metrics(np.zeros_like(a), b, (5, 1, 1))[1]))


class OmicsTests(unittest.TestCase):
    def risk(self):
        return pd.DataFrame(dict(patient_id=["001", "002", "003", "004"], subtype=["GBM"] * 4,
            omics_type=["Single-cell RNA-seq"] * 2 + ["Bulk transcriptomics"] * 2,
            vit_km_score=[-1., 1., -2., 2.], vit_cutoff=[0.] * 4,
            vit_risk_group=["Low risk", "High risk"] * 2))

    def test_risk_linkage_preserves_identifier_and_final_group(self):
        risk = self.risk()
        result = attach_risk(pd.DataFrame({"patient_id": ["003", "004"]}), risk, "Bulk transcriptomics")
        self.assertEqual(result.vit_risk_group.tolist(), ["Low risk", "High risk"])

    def test_stale_group_rejected(self):
        risk = self.risk()
        risk.loc[0, "vit_risk_group"] = "High risk"
        with self.assertRaises(ValueError):
            validate_risk_manifest(risk)

    def test_assay_overlap_rejected(self):
        with self.assertRaises(ValueError):
            attach_risk(pd.DataFrame({"patient_id": ["001", "004"]}), self.risk(), "Bulk transcriptomics")

    def test_supplied_legacy_groups_rejected(self):
        metadata = pd.DataFrame({"patient_id": ["003", "004"], "risk_group": ["High risk", "Low risk"]})
        with self.assertRaises(ValueError):
            attach_risk(metadata, self.risk(), "Bulk transcriptomics")

    @unittest.skipUnless(importlib.util.find_spec("gseapy"), "GSEApy is not installed")
    def test_rank_ssgsea_monotone_transform_invariance(self):
        module = load_module("07_run_bulk_ssgsea.py")
        config = load_config(ROOT / "config/article_defaults.yml")
        rng = np.random.default_rng(17)
        expression = pd.DataFrame(rng.uniform(1, 3, (80, 4)),
            index=[f"G{i}" for i in range(80)], columns=["a", "b", "c", "d"])
        sets = {"one": list(expression.index[:12]), "two": list(expression.index[40:55])}
        first = module.calculate_scores(expression, sets, config)
        second = module.calculate_scores(expression ** 3, sets, config)
        np.testing.assert_allclose(first, second, atol=1e-10)
        self.assertEqual(first.shape, (4, 2))
        self.assertTrue(np.isfinite(first).all().all())


if __name__ == "__main__":
    unittest.main()
