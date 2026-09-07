"""Run the configured nnU-Net v2 planning and full-resolution training commands."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from io_utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Execute the commands after printing them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dataset_id = int(config["nnunet"]["dataset_id"])
    configuration = str(config["nnunet"]["configuration"])
    trainer = str(config["nnunet"]["trainer"])
    dataset = f"Dataset{dataset_id:03d}"
    plan = ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "--verify_dataset_integrity"]
    train = ["nnUNetv2_train", dataset, configuration, "all", "-tr", trainer]
    print(" ".join(plan))
    print(" ".join(train))
    if not args.run:
        return 0
    env = os.environ.copy()
    env.setdefault("nnUNet_raw", str(args.dataset_root.parent.resolve()))
    env.setdefault("nnUNet_preprocessed", str(args.dataset_root.parent.resolve() / "nnUNet_preprocessed"))
    env.setdefault("nnUNet_results", str(args.dataset_root.parent.resolve() / "nnUNet_results"))
    subprocess.run(plan, cwd=args.dataset_root.parent, env=env, check=True)
    subprocess.run(train, cwd=args.dataset_root.parent, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
