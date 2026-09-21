"""Small generated-array tests; these are not clinical validation experiments."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from io_utils import read_manifest


class ManifestIntegrity(unittest.TestCase):
    def check_row(self, time, event, patient_id="case"):
        cfg = {"data": {"patient_id_column": "id", "subtype_column": "subtype",
            "cohort_column": "cohort", "time_column": "time", "event_column": "event",
            "image_columns": {"T1": "T1"}, "mask_column": "mask", "brain_mask_column": "brain"}}
        row = dict(id=patient_id, subtype="GBM", cohort="training", time=time,
                   event=event, T1="x", mask="m", brain="b")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            return read_manifest(path, cfg)

    def test_reject_fractional_event_before_integer_cast(self):
        for value in (0.1, 0.9, 1.9, -0.1, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check_row(12, value)

    def test_reject_nonfinite_or_nonpositive_time(self):
        for value in (float("inf"), float("nan"), 0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check_row(value, 1)

    def test_valid_event(self):
        self.assertEqual(self.check_row(12, 1)["event"].iloc[0], 1)

    def test_leading_zero_identifier_is_preserved(self):
        self.assertEqual(self.check_row(12, 1, "003")["id"].iloc[0], "003")


try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


@unittest.skipIf(sitk is None, "SimpleITK is not installed")
class CoreMaskGeometry(unittest.TestCase):
    def test_segmentation_exports_full_volume_independent_reference(self):
        spec = importlib.util.spec_from_file_location("preprocess_export", ROOT / "01_preprocess_mri.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from io_utils import load_config
        config = load_config(ROOT / "config/article_defaults.yml")
        config["mri_preprocessing"]["n4_bias_correction"]["enabled"] = False
        shape = (8, 24, 24)
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        reference = np.zeros(shape, dtype=np.uint8)
        reference[2:5, 8:15, 7:14] = 1
        final = reference.copy()
        final[2] = 0
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            def save(name, array):
                image = sitk.GetImageFromArray(array)
                image.SetSpacing((1., 1., 5.))
                path = directory / f"{name}.nii.gz"
                sitk.WriteImage(image, str(path))
                return str(path)
            image_path = save("image", values)
            row = pd.Series({"patient_id": "fixture",
                **{column: image_path for column in config["data"]["image_columns"].values()},
                "tumor_mask": save("final", final), "reference_tumor_mask": save("reference", reference),
                "brain_mask": save("brain", np.ones(shape, dtype=np.uint8))})
            output = directory / "out"
            output.mkdir()
            with patch.object(module, "register_rigid", side_effect=lambda fixed, moving, *args: moving):
                result = module.process_case(row, config, output, sitk, directory, "segmentation")
            mask = sitk.GetArrayFromImage(sitk.ReadImage(result["reference_tumor_mask"]))
            np.testing.assert_array_equal(mask, reference)
            self.assertEqual(result["preprocessing_stage"], "registered_normalized_full_volume")
            for column in config["data"]["image_columns"].values():
                normalized = sitk.GetArrayFromImage(sitk.ReadImage(result[column]))
                self.assertEqual(normalized.shape, shape)
                self.assertAlmostEqual(float(normalized.mean()), 0, places=5)
                self.assertAlmostEqual(float(normalized.std()), 1, places=5)

    def test_expanded_crop_is_not_used_as_core(self):
        spec = importlib.util.spec_from_file_location("preprocess", ROOT / "01_preprocess_mri.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        core = np.zeros((8, 24, 24), dtype=np.uint8)
        core[3, 8:14, 8:14] = 1
        core_image = sitk.GetImageFromArray(core)
        core_image.SetSpacing((1., 1., 5.))
        image = sitk.Cast(core_image, sitk.sitkFloat32)
        expanded = module.expand_mask(core_image, 5., sitk)
        images, crop_mask, spacing, index, size = module.crop_and_resize(
            [image] * 4, expanded, [24, 24, 8], sitk)
        resized_core = module.resample_to_reference(core_image, images[0], sitk.sitkNearestNeighbor, sitk)
        core_array = sitk.GetArrayFromImage(resized_core)
        expanded_array = sitk.GetArrayFromImage(crop_mask)
        self.assertTrue(core_array.any())
        self.assertLess(core_array.sum(), expanded_array.sum())
        self.assertEqual(resized_core.GetOrigin(), images[0].GetOrigin())
        self.assertEqual(resized_core.GetSpacing(), tuple(spacing))
        self.assertTrue(np.all(core_array <= expanded_array))


if __name__ == "__main__":
    unittest.main()
