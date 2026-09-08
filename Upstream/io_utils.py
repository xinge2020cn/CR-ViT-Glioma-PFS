"""Shared utilities for the configurable upstream workflow."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    import pandas as pd


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration must contain a YAML mapping.")
    return config


def set_seed(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def require_columns(frame: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {', '.join(missing)}")


def read_manifest(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    data_cfg = config["data"]
    # Preserve identifiers such as 003; these are labels, never measurements.
    frame = pd.read_csv(path, dtype={data_cfg["patient_id_column"]: "string"})
    required = [
        data_cfg["patient_id_column"],
        data_cfg["subtype_column"],
        data_cfg["cohort_column"],
        data_cfg["time_column"],
        data_cfg["event_column"],
        *data_cfg["image_columns"].values(),
        data_cfg["mask_column"],
        data_cfg["brain_mask_column"],
    ]
    require_columns(frame, required, "The input manifest")
    patient_col = data_cfg["patient_id_column"]
    if (frame[patient_col].isna().any()
            or frame[patient_col].str.strip().eq("").any()
            or frame[patient_col].duplicated().any()):
        raise ValueError("The manifest must contain one non-missing row per patient.")
    event_col = data_cfg["event_column"]
    time_col = data_cfg["time_column"]
    frame[time_col] = pd.to_numeric(frame[time_col], errors="raise")
    frame[event_col] = pd.to_numeric(frame[event_col], errors="raise")
    if (not np.isfinite(frame[time_col]).all()
            or not frame[time_col].gt(0).all()
            or not frame[event_col].isin([0, 1]).all()):
        raise ValueError("Survival time must be finite and positive and event must be exactly 0/1.")
    frame[event_col] = frame[event_col].astype(int)
    return frame


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
