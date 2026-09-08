"""Deterministic arithmetic tests; toy tensors are not study MRI or study results."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn

UPSTREAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UPSTREAM))
spec = importlib.util.spec_from_file_location("occlusion", UPSTREAM / "06_generate_attributions.py")
occ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(occ)


def config():
    return {
        "project": {"device": "cpu"},
        "data": {"image_columns": {name: name for name in ("T1WI", "T2WI", "T2_FLAIR", "CE_T1WI")},
                 "cohort_column": "cohort", "training_cohort": "training",
                 "patient_id_column": "patient_id", "subtype_column": "subtype"},
        "mri_preprocessing": {"target_size_voxels_xyz": [4, 3, 2]},
        "survival_model": {"input_array_order": "CZYX", "patch_size_xyz": [2, 2, 1], "embed_dim": 8},
        "attribution": {"methods": ["occlusion"], "occlusion_fill_strategy": "training_channel_mean",
                        "occlusion_block_size_xyz": [2, 2, 1], "occlusion_stride_xyz": [1, 1, 1]},
    }


class SumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, x):
        self.calls.append((self.training, torch.is_grad_enabled(), x.detach().clone()))
        return self.weight * x.flatten(1).sum(1)


class OcclusionTests(unittest.TestCase):
    def test_full_windows_and_article_count(self):
        self.assertEqual(occ.window_starts(8, 3, 2), [0, 2, 4, 5])
        self.assertEqual(occ.window_starts(3, 3, 3), [0])
        counts = [len(occ.window_starts(a, b, s)) for a, b, s in zip((192, 192, 24), (16, 16, 4), (8, 8, 2))]
        self.assertEqual(np.prod(counts), 5819)
        for args in ((4, 5, 2), (5, 2, 3), (5, 2, 0)):
            with self.assertRaises(ValueError):
                occ.window_starts(*args)

    def test_joint_means_arithmetic_and_frozen_model(self):
        model = SumModel().train()
        before = model.weight.detach().clone()
        image = torch.arange(4 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 4, 2, 3, 4)
        unchanged = image.clone()
        fill = np.array([1, 2, 3, 4], dtype=np.float32)
        result = occ.occlusion_map(model, image, (1, 2, 2), (1, 1, 1), fill, "cpu")
        expected_sum = np.zeros((2, 3, 4))
        expected_count = np.zeros((2, 3, 4))
        for bounds, delta in zip(result["window_bounds_zyx"], result["window_score_differences"]):
            z, y, x, ze, ye, xe = bounds
            region = image[0, :, z:ze, y:ye, x:xe]
            self.assertEqual(tuple(region.shape), (4, 1, 2, 2))
            expected_delta = float((region - torch.tensor(fill).reshape(4, 1, 1, 1)).sum())
            self.assertAlmostEqual(delta, expected_delta)
            expected_sum[z:ze, y:ye, x:xe] += expected_delta
            expected_count[z:ze, y:ye, x:xe] += 1
        np.testing.assert_allclose(result["occlusion_signed_zyx"], expected_sum / expected_count)
        np.testing.assert_array_equal(result["occlusion_overlap_count_zyx"], expected_count)
        self.assertEqual(len(model.calls), len(result["window_score_differences"]) + 1)
        self.assertTrue(all(not training and not grad for training, grad, _ in model.calls))
        self.assertFalse(model.weight.requires_grad)
        self.assertIsNone(model.weight.grad)
        self.assertTrue(torch.equal(before, model.weight))
        self.assertTrue(torch.equal(image, unchanged))

    def test_absolute_after_signed_overlap_average_not_before(self):
        image = torch.zeros(1, 4, 1, 1, 3)
        image[0, 0, 0, 0] = torch.tensor([2.0, 0.0, -2.0])
        result = occ.occlusion_map(SumModel(), image, (1, 1, 2), (1, 1, 1), [0, 0, 0, 0], "cpu")
        np.testing.assert_array_equal(result["window_score_differences"], [2, -2])
        np.testing.assert_array_equal(result["occlusion_signed_zyx"].flatten(), [2, 0, -2])
        np.testing.assert_array_equal(result["occlusion_absolute_zyx"].flatten(), [2, 0, 2])

    def write_volume(self, path, cfg, values, **extra):
        array = np.broadcast_to(np.asarray(values, dtype=np.float32).reshape(4, 1, 1, 1), occ.input_shape(cfg)).copy()
        np.savez(path, image=array, sequence_names=np.asarray(list(cfg["data"]["image_columns"])),
                 preprocessing_config_json=np.asarray(json.dumps(cfg["mri_preprocessing"])), **extra)

    def test_all_training_channel_means_without_validation_or_subtype_leakage(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"toy_{i}.npz" for i in range(3)]
            for path, offset in zip(paths, (0, 2, 100)):
                self.write_volume(path, cfg, np.arange(4) + offset)
            frame = pd.DataFrame({"patient_id": ["a", "b"], "subtype": ["GBM"] * 2,
                                  "cohort": ["training"] * 2, "processed_npz": paths[:2]})
            means, metadata = occ.compute_channel_means(frame, cfg)
            np.testing.assert_allclose(means, [1, 2, 3, 4])
            self.assertEqual(metadata["training_patient_count"], 2)
            self.assertEqual(metadata["voxels_per_channel"], 48)
            self.assertTrue(metadata["preprocessing_metadata_verified"])
            frame.loc[1, "cohort"] = "temporal"
            with self.assertRaisesRegex(ValueError, "validation"):
                occ.compute_channel_means(frame, cfg)
            frame.loc[1, "cohort"] = "training"
            frame.loc[1, "subtype"] = "IDHmut-intact"
            with self.assertRaisesRegex(ValueError, "subtype"):
                occ.compute_channel_means(frame, cfg)

    def test_input_order_shape_finiteness_and_preprocessing_checks(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.npz"
            self.write_volume(path, cfg, [0, 1, 2, 3])
            self.assertEqual(tuple(occ.load_image(path, cfg).shape), (1, 4, 2, 3, 4))
            altered = copy.deepcopy(cfg)
            altered["mri_preprocessing"]["target_size_voxels_xyz"] = [3, 3, 2]
            with self.assertRaisesRegex(ValueError, "preprocessing"):
                occ.load_image(path, altered)
            np.savez(path, image=np.zeros((4, 2, 3, 4)), sequence_names=np.asarray(["wrong"] * 4))
            with self.assertRaisesRegex(ValueError, "channel order"):
                occ.load_image(path, cfg)
            np.savez(path, image=np.zeros((4, 2, 3, 5)), sequence_names=np.asarray(list(cfg["data"]["image_columns"])))
            with self.assertRaisesRegex(ValueError, "shape"):
                occ.load_image(path, cfg)
            self.write_volume(path, cfg, [0, float("nan"), 2, 3])
            with self.assertRaisesRegex(ValueError, "non-finite"):
                occ.load_image(path, cfg)

    def test_checkpoint_mismatch_rejected_before_instantiation_and_safe_load(self):
        cfg = config()
        toy = SumModel()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.pt"
            torch.save({"config": cfg, "model_name": "3d_vit", "subtype": "GBM", "state_dict": toy.state_dict()}, path)
            module = mock.Mock()
            module.make_model.side_effect = lambda name, loaded_config: SumModel()
            with mock.patch.object(occ.torch, "load", wraps=torch.load) as loader:
                locked, _ = occ.load_locked_model(path, cfg, module, "cpu")
                self.assertTrue(loader.call_args.kwargs["weights_only"])
            self.assertFalse(locked.training)
            self.assertFalse(locked.weight.requires_grad)
            module.reset_mock()
            altered = copy.deepcopy(cfg)
            altered["survival_model"]["embed_dim"] = 16
            with self.assertRaisesRegex(ValueError, "architecture"):
                occ.load_locked_model(path, altered, module, "cpu")
            module.make_model.assert_not_called()
            torch.save({"model_name": "3d_vit", "subtype": "GBM", "state_dict": toy.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "original training config"):
                occ.load_locked_model(path, cfg, module, "cpu")

    def test_expanded_mask_not_used_as_core_and_true_max_area_slice(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.npz"
            expanded = np.ones((2, 3, 4))
            self.write_volume(path, cfg, [0, 1, 2, 3], tumor_mask=expanded)
            arrays, metadata = occ.display_geometry(path, (2, 3, 4))
            self.assertNotIn("axial_slice_index", arrays)
            self.assertIsNone(metadata["axial_slice_index"])
            core = np.zeros((2, 3, 4))
            core[0, 0, 0] = 1
            core[1, 1, :3] = 1
            self.write_volume(path, cfg, [0, 1, 2, 3], tumor_core_mask=core, tumor_mask=expanded)
            arrays, metadata = occ.display_geometry(path, (2, 3, 4))
            self.assertEqual(int(arrays["axial_slice_index"]), 1)
            self.assertEqual(metadata["axial_slice_index"], 1)

    def test_no_unsupported_method_and_no_shap_import(self):
        cfg = config()
        self.assertEqual(occ.validate_attribution_config(cfg)["methods"], ["occlusion"])
        for methods in (["deep_shap"], ["occlusion", "deep_shap"], []):
            cfg["attribution"]["methods"] = methods
            with self.assertRaisesRegex(ValueError, "unsupported methods"):
                occ.validate_attribution_config(cfg)
        source = (UPSTREAM / "06_generate_attributions.py").read_text(encoding="utf-8")
        self.assertNotIn("import shap", source)
        self.assertNotIn("DeepExplainer", source)


if __name__ == "__main__":
    unittest.main()
