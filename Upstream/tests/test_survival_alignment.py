"""Small deterministic unit checks; these are not patient/model validation."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import load_config


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


models = load_module("04_train_survival_models.py", "survival_models_test")
validation = load_module("09_validate_upstream_configuration.py", "config_validation_test")


class ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class SurvivalAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = load_config(ROOT / "config/article_defaults.yml")

    def test_current_profile_and_retired_settings(self):
        validation.validate_config(self.config)
        for section, key, value in [
            ("survival_model", "embed_dim", 128),
            ("attribution", "methods", ["occlusion", "deep_shap"]),
            ("attribution", "occlusion_fill_value", 0.0),
        ]:
            config = copy.deepcopy(self.config)
            config[section][key] = value
            with self.assertRaises(SystemExit):
                validation.validate_config(config)

    def test_architecture_matches_manuscript(self):
        vit = models.ViTCox3D(self.config)
        self.assertEqual(tuple(vit.position.shape), (1, 865, 256))
        self.assertEqual(len(vit.encoder.layers), 6)
        self.assertEqual(vit.encoder.layers[0].self_attn.num_heads, 8)
        self.assertTrue(vit.encoder.layers[0].norm_first)
        self.assertEqual(vit.cox_head[0].out_features, 64)
        self.assertEqual(vit.cox_head[2].p, 0.3)
        # Small spatial input only for shape/finite smoke testing, not study inference.
        small = copy.deepcopy(self.config)
        small["mri_preprocessing"]["target_size_voxels_xyz"] = [32, 32, 8]
        smoke = models.ViTCox3D(small).eval()
        with torch.no_grad():
            result = smoke(torch.zeros(2, 4, 8, 32, 32))
        self.assertEqual(tuple(result.shape), (2,))
        self.assertTrue(bool(torch.isfinite(result).all()))
        cnn = models.ResNet3D18Cox(self.config)
        self.assertEqual(cnn.dropout.p, 0.3)

    def test_breslow_ties_and_constant_shift(self):
        risk = torch.tensor([0.4, -0.2, 0.7, 0.1], dtype=torch.float64, requires_grad=True)
        time = torch.tensor([2.0, 2.0, 4.0, 5.0], dtype=torch.float64)
        event = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64)
        expected = -(risk[:2].sum() - 2 * torch.logsumexp(risk, 0)
                     + risk[2] - torch.logsumexp(risk[2:], 0)) / 3
        actual = models.cox_breslow_loss(risk, time, event)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, models.cox_breslow_loss(risk + 1000, time, event))
        actual.backward()
        self.assertTrue(bool(torch.isfinite(risk.grad).all()))

    def test_full_risk_set_chunked_forward_same_loss_and_gradient(self):
        x = torch.tensor([[1.0, 0.2], [0.3, 1.1], [-0.2, 0.7], [0.8, -0.2], [1.4, 0.4]])
        time = torch.tensor([1.0, 2.0, 2.0, 4.0, 5.0])
        event = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
        model = ScalarModel()
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor([[0.2, -0.1]]))
        full = models.cox_breslow_loss(model(x), time, event)
        full.backward()
        gradient = model.linear.weight.grad.clone()
        model.zero_grad()
        loader = DataLoader(TensorDataset(x, time, event, torch.arange(5)), batch_size=2)
        risk, gathered_time, gathered_event = models.collect_risk_set(model, loader, torch.device("cpu"))
        chunked = models.cox_breslow_loss(risk, gathered_time, gathered_event)
        chunked.backward()
        torch.testing.assert_close(full, chunked)
        torch.testing.assert_close(gradient, model.linear.weight.grad)
        self.assertAlmostEqual(models.evaluate_loss(model, loader, torch.device("cpu")), full.item())

    def test_invalid_cox_inputs_rejected(self):
        risk = torch.tensor([0.2, 0.3])
        time = torch.tensor([1.0, 2.0])
        for bad_risk, bad_time, bad_event in [
            (risk, time, torch.zeros(2)),
            (risk, time, torch.tensor([0.9, 1.0])),
            (risk, torch.tensor([1.0, float("inf")]), torch.ones(2)),
            (torch.tensor([float("nan"), 0.2]), time, torch.ones(2)),
        ]:
            with self.assertRaises(ValueError):
                models.cox_breslow_loss(bad_risk, bad_time, bad_event)

    def test_stratified_split_deterministic_disjoint(self):
        frame = pd.DataFrame({"patient_id": np.arange(160), "pfs_time_months": np.arange(1, 161),
                              "event": np.tile([0, 1], 80)})
        train, tune = models.split_frame(frame, 0.2, 2026)
        train2, tune2 = models.split_frame(frame, 0.2, 2026)
        self.assertEqual(len(tune), 32)
        self.assertEqual(set(tune.patient_id), set(tune2.patient_id))
        self.assertFalse(set(train.patient_id) & set(tune.patient_id))
        frame["stratum"] = frame.event.astype(str) + "_" + pd.qcut(
            frame.pfs_time_months, q=4, labels=False).astype(str)
        self.assertEqual(len(frame.loc[tune.index, "stratum"].unique()), 8)
        self.assertEqual(len(frame.loc[train.index, "stratum"].unique()), 8)

    def test_warmup_and_cosine(self):
        cfg = self.config["survival_model"]
        self.assertAlmostEqual(models.learning_rate_at_epoch(cfg, 1), 1e-5)
        self.assertAlmostEqual(models.learning_rate_at_epoch(cfg, 10), 1e-4)
        self.assertAlmostEqual(models.learning_rate_at_epoch(cfg, 105), 5e-5)
        self.assertAlmostEqual(models.learning_rate_at_epoch(cfg, 200), 0.0)

    def test_no_flip_identity_augmentation(self):
        aug = copy.deepcopy(self.config["survival_model"]["augmentation"])
        aug.update(rotation_degrees_xyz=[0, 0, 0], translation_voxels_xyz=[0, 0, 0],
                   spatial_scale_range=[1, 1], intensity_scale_range=[1, 1],
                   intensity_shift_max=0, noise_std_max=0)
        image = np.arange(4 * 4 * 8 * 8, dtype=np.float32).reshape(4, 4, 8, 8)
        np.testing.assert_array_equal(models.augment_image(image, aug), image)


if __name__ == "__main__":
    unittest.main(verbosity=2)
