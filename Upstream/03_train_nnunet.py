"""Train five nnU-Net folds and infer frozen validation-ensemble predictions."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess

from io_utils import load_config


def build_commands(config: dict, dataset_root: Path, results_root: Path) -> list[list[str]]:
    nn = config["nnunet"]
    folds = list(nn["folds"])
    if folds != [0, 1, 2, 3, 4] or nn["trainer"] != "nnUNetTrainer":
        raise ValueError("The segmentation profile requires the standard five-fold trainer.")
    dataset = str(int(nn["dataset_id"]))
    commands = [["nnUNetv2_plan_and_preprocess", "-d", dataset, "--verify_dataset_integrity"]]
    commands += [["nnUNetv2_train", dataset, nn["configuration"], str(fold), "-tr", nn["trainer"]]
                 for fold in folds]
    if (dataset_root / "imagesTs").is_dir():
        commands.append([
            "nnUNetv2_predict", "-i", str(dataset_root / "imagesTs"),
            "-o", str(results_root / dataset_root.name / "validation_ensemble"),
            "-d", dataset, "-c", nn["configuration"], "-tr", nn["trainer"],
            "-f", *map(str, folds), "-chk", "checkpoint_final.pth",
        ])
    return commands


def make_splits(case_names: list[str], seed: int) -> list[dict]:
    from sklearn.model_selection import KFold
    if len(case_names) < 5 or len(case_names) != len(set(case_names)):
        raise ValueError("Five-fold splitting requires at least five unique training patients.")
    names = sorted(case_names)
    return [{"train": [names[i] for i in train], "val": [names[i] for i in val]}
            for train, val in KFold(5, shuffle=True, random_state=seed).split(names)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = args.dataset_root.resolve()
    env = os.environ.copy()
    env["nnUNet_raw"] = str(dataset.parent)
    work_root = dataset.parent.parent
    preprocessed = Path(env.setdefault("nnUNet_preprocessed", str(work_root / "nnUNet_preprocessed")))
    results = Path(env.setdefault("nnUNet_results", str(work_root / "nnUNet_results")))
    commands = build_commands(config, dataset, results)
    for command in commands:
        print(subprocess.list2cmdline(command))
    if not args.run:
        return 0
    if version("nnunetv2") != str(config["nnunet"]["version"]):
        raise RuntimeError("Install the configured nnU-Net version before executing training.")
    records = json.loads((dataset / "case_manifest.json").read_text(encoding="utf-8"))
    splits = make_splits([r["case"] for r in records if r["training"]],
                        int(config["project"]["random_seed"]))
    split_path = preprocessed / dataset.name / "splits_final.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    if split_path.exists() and json.loads(split_path.read_text(encoding="utf-8")) != splits:
        raise ValueError("Existing folds differ; use a separate preprocessing workspace.")
    split_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    for command in commands:
        subprocess.run(command, env=env, check=True)
    model_dir = results / dataset.name / f"{config['nnunet']['trainer']}__nnUNetPlans__{config['nnunet']['configuration']}"
    rows = []
    held_out = {name: fold for fold, split in enumerate(splits) for name in split["val"]}
    for row in records:
        folder = (model_dir / f"fold_{held_out[row['case']]}" / "validation" if row["training"]
                  else results / dataset.name / "validation_ensemble")
        prediction = folder / f"{row['case']}.nii.gz"
        if not prediction.is_file():
            raise FileNotFoundError(f"Missing held-out prediction: {prediction}")
        rows.append({**row, "prediction_mask": str(prediction),
                     "prediction_type": "out_of_fold" if row["training"] else "frozen_ensemble"})
    (results / dataset.name / "evaluation_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
