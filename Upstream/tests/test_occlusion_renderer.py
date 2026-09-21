"""Temporary toy-array rendering tests, never study attribution maps."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

UPSTREAM = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("render_occlusion", UPSTREAM / "10_render_occlusion_panels.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class RendererTests(unittest.TestCase):
    def toy_arrays(self, scale=1.0, checkpoint="a" * 64):
        image = (np.arange(4 * 2 * 4 * 4).reshape(4, 2, 4, 4) / 80.0 - 0.5).astype(np.float32)
        signed = np.linspace(-scale, scale, 32, dtype=np.float32).reshape(2, 4, 4)
        core = np.zeros((2, 4, 4), dtype=np.uint8)
        core[0, 0, 0] = 1
        core[1, 1:3, 1:3] = 1
        metadata = {"method": "3D occlusion sensitivity", "model": "3d_vit",
                    "checkpoint_sha256": checkpoint, "processed_npz_sha256": "b" * 64,
                    "axial_slice_index": 1, "purpose": "temporary toy test only"}
        return dict(image_czyx=image, occlusion_signed_zyx=signed,
                    occlusion_absolute_zyx=np.abs(signed), tumor_core_mask_zyx=core,
                    axial_slice_index=np.asarray(1, dtype=np.int32),
                    sequence_names=np.asarray(["T1WI", "T2WI", "T2_FLAIR", "CE_T1WI"]),
                    metadata_json=np.asarray(json.dumps(metadata)))

    def test_valid_panel_preserves_matching_slice_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.npz"
            arrays = self.toy_arrays()
            np.savez(path, **arrays)
            panel = renderer.read_panel(path, "T2_FLAIR")
            self.assertEqual(panel["slice"], 1)
            np.testing.assert_array_equal(panel["mri"], arrays["image_czyx"][2, 1])
            np.testing.assert_array_equal(panel["sensitivity"], arrays["occlusion_absolute_zyx"][1])

    def test_rejects_geometry_absolute_and_core_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.npz"
            mutations = [
                ("occlusion_absolute_zyx", np.full((2, 4, 4), 99.0), "Absolute map"),
                ("tumor_core_mask_zyx", np.ones((1, 4, 4)), "geometry"),
                ("tumor_core_mask_zyx", np.zeros((2, 4, 4)), "nonempty"),
                ("axial_slice_index", np.asarray(0), "display slice"),
                ("occlusion_signed_zyx", np.full((2, 4, 4), np.nan), "finite"),
            ]
            for key, value, error in mutations:
                with self.subTest(key=key, error=error):
                    arrays = self.toy_arrays()
                    arrays[key] = value
                    np.savez(path, **arrays)
                    with self.assertRaisesRegex(ValueError, error):
                        renderer.read_panel(path, "T2_FLAIR")
            arrays = self.toy_arrays()
            del arrays["tumor_core_mask_zyx"]
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "missing"):
                renderer.read_panel(path, "T2_FLAIR")

    def test_same_checkpoint_shared_scale_and_png_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            maps = [root / "toy_a.npz", root / "toy_b.npz"]
            for path, scale in zip(maps, [1.0, 2.0]):
                np.savez(path, **self.toy_arrays(scale=scale))
            output = root / "toy_render_only"
            argv = ["renderer", "--maps", *map(str, maps), "--output-dir", str(output)]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(renderer.main(), 0)
            metadata = json.loads((output / "render_metadata.json").read_text())
            self.assertEqual(metadata["color_scale"], [0, 2.0])
            self.assertEqual(len(metadata["panels"]), 2)
            self.assertTrue(all(p["clipped_voxel_fraction"] == 0 for p in metadata["panels"]))
            self.assertEqual(metadata["smoothing"], "none")
            pngs = list(output.glob("*.png"))
            self.assertEqual(len(pngs), 5)
            for path in pngs:
                with Image.open(path) as image:
                    image.verify()
                self.assertGreater(path.stat().st_size, 100)

    def test_refuses_common_raw_scale_for_different_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            maps = [root / "toy_a.npz", root / "toy_b.npz"]
            np.savez(maps[0], **self.toy_arrays(checkpoint="a" * 64))
            np.savez(maps[1], **self.toy_arrays(checkpoint="c" * 64))
            argv = ["renderer", "--maps", *map(str, maps), "--output-dir", str(root / "output")]
            with mock.patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "matched checkpoint"):
                renderer.main()
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
