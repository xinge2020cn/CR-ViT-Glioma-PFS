"""Render independent MRI/occlusion PNG panels from verified 06 output arrays.

No checkpoint is trained, no attribution is invented, and no smoothing is used.
The PPT layout remains a separate figure-production step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_panel(path: Path, channel: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = {"image_czyx", "occlusion_signed_zyx", "occlusion_absolute_zyx",
                    "tumor_core_mask_zyx", "axial_slice_index", "sequence_names", "metadata_json"}
        if missing := required.difference(data.files):
            raise ValueError(f"{path.name}: missing verified rendering inputs: {sorted(missing)}")
        meta = json.loads(str(data["metadata_json"].item()))
        if meta.get("method") != "3D occlusion sensitivity" or meta.get("model") != "3d_vit":
            raise ValueError("Only fixed-ViT occlusion arrays from stage 06 are supported.")
        if not meta.get("checkpoint_sha256") or not meta.get("processed_npz_sha256"):
            raise ValueError("Checkpoint and input provenance must be retained.")
        image = np.asarray(data["image_czyx"])
        signed = np.asarray(data["occlusion_signed_zyx"])
        absolute = np.asarray(data["occlusion_absolute_zyx"])
        core = np.asarray(data["tumor_core_mask_zyx"])
        names = data["sequence_names"].tolist()
        if (image.ndim != 4 or image.shape[0] != 4 or channel not in names
                or not isinstance(names, list) or len(names) != 4 or len(set(names)) != 4):
            raise ValueError("Four-channel CZYX MRI and the requested sequence are required.")
        if any(a.shape != image.shape[1:] for a in (signed, absolute, core)):
            raise ValueError("Attribution and core-mask geometry differ from the MRI input.")
        if any(not np.isfinite(a).all() for a in (image, signed, absolute, core)):
            raise ValueError("Rendering inputs must be finite.")
        if not np.allclose(absolute, np.abs(signed), rtol=1e-6, atol=1e-7):
            raise ValueError("Absolute map is not abs(overlap-averaged signed map).")
        if not (core > 0).any():
            raise ValueError("An unexpanded, nonempty tumor-core mask is required.")
        raw_index = np.asarray(data["axial_slice_index"]).item()
        if (isinstance(raw_index, bool) or not np.isfinite(raw_index)
                or float(raw_index) != int(raw_index)):
            raise ValueError("Slice index must be a finite integer.")
        z = int(raw_index)
        if meta.get("axial_slice_index") != z:
            raise ValueError("Metadata and saved array disagree on the display slice.")
        expected = int((core > 0).sum(axis=(1, 2)).argmax())
        if z != expected:
            raise ValueError("Recorded slice does not have the largest unexpanded tumor-core area.")
        return {"source": str(path.resolve()), "metadata": meta, "slice": z,
                "mri": image[names.index(channel), z].astype(float),
                "sensitivity": absolute[z].astype(float)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel", default="T2_FLAIR")
    parser.add_argument("--vmax", type=float, help="Fixed positive upper color limit; default: maximum across supplied slices.")
    parser.add_argument("--alpha", type=float, default=0.55)
    args = parser.parse_args()
    if not 0 < args.alpha <= 1:
        raise ValueError("Overlay opacity must lie in (0,1].")
    panels = [read_panel(path, args.channel) for path in args.maps]
    if len({p["metadata"]["checkpoint_sha256"] for p in panels}) != 1:
        raise ValueError("Render one matched checkpoint per run; raw scores from separate subtype models are not directly comparable.")
    maximum = max(float(p["sensitivity"].max()) for p in panels)
    vmax = maximum if args.vmax is None else args.vmax
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("Color limit must be finite and positive; all-zero maps have no nonzero sensitivity to display.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    plt.rcParams.update({"font.family": "Times New Roman", "font.weight": "bold",
                         "axes.labelweight": "bold", "font.size": 11})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    norm, cmap = Normalize(0, vmax, clip=True), plt.get_cmap("inferno")
    records = []
    for index, panel in enumerate(panels, 1):
        brain = panel["mri"][panel["mri"] != 0]
        if brain.size < 2:
            raise ValueError("MRI slice lacks sufficient nonzero voxels for display.")
        lo, hi = np.percentile(brain, [2, 98])
        if hi <= lo:
            raise ValueError("MRI slice has no usable contrast.")
        stem = f"panel_{index:02d}"
        for overlay, suffix in [(False, "MRI"), (True, "occlusion")]:
            fig = plt.figure(figsize=(3, 3), facecolor="black")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(panel["mri"], cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
            if overlay:
                alpha = args.alpha * (panel["sensitivity"] > 0)
                ax.imshow(panel["sensitivity"], cmap=cmap, norm=norm, alpha=alpha,
                          origin="lower", interpolation="nearest")
            ax.set_axis_off()
            fig.savefig(args.output_dir / f"{stem}_{suffix}.png", dpi=300, pad_inches=0)
            plt.close(fig)
        records.append({"panel": stem, "source": panel["source"], "slice": panel["slice"],
                        "metadata": panel["metadata"], "mri_display_percentiles": [2, 98],
                        "mri_display_limits": [float(lo), float(hi)],
                        "clipped_voxel_fraction": float((panel["sensitivity"] > vmax).mean())})
    fig = plt.figure(figsize=(3.8, 0.7), facecolor="white")
    ax = fig.add_axes([0.1, 0.56, 0.8, 0.22])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation="horizontal")
    colorbar.set_ticks([0, vmax / 2, vmax])
    colorbar.set_ticklabels(["0", f"{vmax / 2:.3g}", f"{vmax:.3g}"])
    colorbar.set_label("Absolute mean risk-score change", labelpad=1)
    fig.savefig(args.output_dir / "Occlusion color bar.png", dpi=300)
    plt.close(fig)
    (args.output_dir / "render_metadata.json").write_text(json.dumps({
        "channel": args.channel, "color_scale": [0, vmax], "colormap": "inferno",
        "opacity": args.alpha, "smoothing": "none", "resampling_for_display": "nearest",
        "input_orientation": "CZYX input slice; origin lower, no additional flip or rotation",
        "selection": "first maximum unexpanded tumor-core area slice",
        "shared_scale_scope": "supplied slices from one matched checkpoint", "panels": records,
    }, indent=2), encoding="utf-8")
    print(f"Rendered {len(panels)} independent panels and one shared numeric color bar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
