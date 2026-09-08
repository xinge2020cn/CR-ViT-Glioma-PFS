"""Train subtype-specific 3D-ViT and 3D-ResNet-18 Cox models."""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from io_utils import load_config, read_manifest, set_seed, write_json


def augment_image(image: np.ndarray, aug: dict) -> np.ndarray:
    """Apply one common 3D transform to all MRI channels; never reflect an axis."""
    from scipy.ndimage import affine_transform
    from scipy.spatial.transform import Rotation

    angles = np.random.uniform(-1.0, 1.0, size=3) * np.asarray(aug["rotation_degrees_xyz"])
    rotation_xyz = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    scale = float(np.random.uniform(*aug["spatial_scale_range"]))
    forward_zyx = (rotation_xyz * scale)[::-1, ::-1]
    inverse = np.linalg.inv(forward_zyx)
    translation_zyx = (
        np.random.uniform(-1.0, 1.0, size=3) * np.asarray(aug["translation_voxels_xyz"])
    )[::-1]
    center = (np.asarray(image.shape[1:], dtype=float) - 1.0) / 2.0
    offset = center - inverse @ (center + translation_zyx)
    transformed = np.stack([
        affine_transform(channel, inverse, offset=offset, order=1, mode="constant", cval=0.0,
                         prefilter=False)
        for channel in image
    ])
    intensity_scale = np.random.uniform(*aug["intensity_scale_range"], size=(image.shape[0], 1, 1, 1))
    shift = np.random.uniform(-aug["intensity_shift_max"], aug["intensity_shift_max"],
                              size=(image.shape[0], 1, 1, 1))
    std = np.random.uniform(0.0, aug["noise_std_max"], size=(image.shape[0], 1, 1, 1))
    transformed = transformed * intensity_scale + shift + np.random.normal(size=image.shape) * std
    return np.ascontiguousarray(transformed, dtype=np.float32)


class NPZSurvivalDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, config: dict, training: bool) -> None:
        self.frame = frame.reset_index(drop=True)
        self.config = config
        self.training = training
        self.data_cfg = config["data"]
        self.model_cfg = config["survival_model"]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, str]:
        row = self.frame.iloc[index]
        with np.load(row["processed_npz"], allow_pickle=False) as bundle:
            image = np.asarray(bundle["image"], dtype=np.float32)
        if image.ndim != 4:
            raise ValueError("Each processed image must have shape CZYX.")
        expected = (len(self.data_cfg["image_columns"]), *reversed(
            self.config["mri_preprocessing"]["target_size_voxels_xyz"]))
        if image.shape != expected or not np.isfinite(image).all():
            raise ValueError(f"Image must be finite with shape {expected}; got {image.shape}.")
        if self.training and self.model_cfg["augmentation"]["enabled"]:
            image = augment_image(image, self.model_cfg["augmentation"])
        time = float(row[self.data_cfg["time_column"]])
        event = float(row[self.data_cfg["event_column"]])
        if not np.isfinite(time) or time <= 0 or event not in (0.0, 1.0):
            raise ValueError("Survival times must be finite and positive; events must be 0/1.")
        return (
            torch.from_numpy(image),
            torch.tensor(time, dtype=torch.float32),
            torch.tensor(event, dtype=torch.float32),
            str(row[self.data_cfg["patient_id_column"]]),
        )


class PatchEmbed3D(nn.Module):
    def __init__(self, channels: int, embed_dim: int, patch_size_zyx: tuple[int, int, int]) -> None:
        super().__init__()
        self.projection = nn.Conv3d(channels, embed_dim, kernel_size=patch_size_zyx, stride=patch_size_zyx)

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)


class ViTCox3D(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config["survival_model"]
        target_xyz = [int(value) for value in config["mri_preprocessing"]["target_size_voxels_xyz"]]
        patch_xyz = [int(value) for value in model_cfg["patch_size_xyz"]]
        target_zyx = tuple(reversed(target_xyz))
        patch_zyx = tuple(reversed(patch_xyz))
        if any(target % patch for target, patch in zip(target_zyx, patch_zyx)):
            raise ValueError("The target size must be divisible by the 3D patch size.")
        channels = len(config["data"]["image_columns"])
        embed_dim = int(model_cfg["embed_dim"])
        token_count = int(np.prod([target // patch for target, patch in zip(target_zyx, patch_zyx)]))
        self.patch_embed = PatchEmbed3D(channels, embed_dim, patch_zyx)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, token_count + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=int(model_cfg["attention_heads"]),
            dim_feedforward=int(embed_dim * float(model_cfg["mlp_ratio"])),
            dropout=float(model_cfg["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(model_cfg["transformer_depth"]))
        self.norm = nn.LayerNorm(embed_dim)
        self.cox_head = nn.Sequential(
            nn.Linear(embed_dim, int(model_cfg["survival_head_hidden_dim"])),
            nn.GELU(),
            nn.Dropout(float(model_cfg["survival_head_dropout"])),
            nn.Linear(int(model_cfg["survival_head_hidden_dim"]), 1),
        )
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.patch_embed(x)
        class_tokens = self.class_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([class_tokens, tokens], dim=1)
        tokens = self.encoder(tokens + self.position)
        return self.cox_head(self.norm(tokens[:, 0])).squeeze(-1)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.downsample(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class ResNet3D18Cox(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        channels = len(config["data"]["image_columns"])
        self.stem = nn.Sequential(
            nn.Conv3d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(float(config["survival_model"]["cnn_head_dropout"]))
        self.cox_head = nn.Linear(512, 1)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = [BasicBlock3D(in_channels, out_channels, stride)]
        layers.extend(BasicBlock3D(out_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.cox_head(self.dropout(self.pool(x).flatten(1))).squeeze(-1)


def make_model(name: str, config: dict) -> nn.Module:
    if name == "3d_vit":
        return ViTCox3D(config)
    if name == "3d_resnet18":
        return ResNet3D18Cox(config)
    raise ValueError(f"Unsupported survival model: {name}")


def cox_breslow_loss(risk: Tensor, time: Tensor, event: Tensor) -> Tensor:
    """Mean negative Breslow partial log likelihood on one complete risk set."""
    if risk.ndim != 1 or time.shape != risk.shape or event.shape != risk.shape:
        raise ValueError("Risk, time, and event must be equally sized one-dimensional tensors.")
    if not bool(torch.isfinite(risk).all() & torch.isfinite(time).all() & torch.isfinite(event).all()):
        raise ValueError("Cox inputs must be finite.")
    if not bool((time > 0).all()) or not bool(((event == 0) | (event == 1)).all()):
        raise ValueError("Survival times must be positive and events binary.")
    event_mask = event > 0.5
    if not bool(event_mask.any()):
        raise ValueError("The complete Cox fitting/evaluation risk set has no observed events.")
    event_times = torch.unique(time[event_mask])
    terms = []
    for event_time in event_times:
        deaths = (event_mask & (time == event_time))
        at_risk = time >= event_time
        death_count = deaths.sum().to(dtype=risk.dtype)
        terms.append(risk[deaths].sum() - death_count * torch.logsumexp(risk[at_risk], dim=0))
    return -torch.stack(terms).sum() / event_mask.sum().to(dtype=risk.dtype)


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        risk, time, event = collect_risk_set(model, loader, device)
        return float(cox_breslow_loss(risk, time, event).cpu())


def collect_risk_set(model: nn.Module, loader: DataLoader, device: torch.device):
    """Chunk only the forward pass; never truncate a patient's Cox risk set.

    During fitting, all chunk graphs remain alive until the cohort loss is
    differentiated. This is mathematically exact but memory demanding. Decreasing
    batch_size does not remove the memory needed for all retained graphs.
    """
    risks, times, events = [], [], []
    for image, time, event, _ in loader:
        risks.append(model(image.to(device)))
        times.append(time.to(device))
        events.append(event.to(device))
    if not risks:
        raise ValueError("Cannot evaluate an empty Cox risk set.")
    return torch.cat(risks), torch.cat(times), torch.cat(events)


def split_frame(frame: pd.DataFrame, fraction: float, seed: int,
                time_col: str = "pfs_time_months", event_col: str = "event"):
    """Patient-level event-by-time-quartile stratification; fail on sparse strata."""
    from sklearn.model_selection import train_test_split

    if len(frame) < 10 or not 0.0 < fraction < 1.0:
        raise ValueError("A stratified tuning split requires at least 10 patients and fraction in (0,1).")
    times = pd.to_numeric(frame[time_col], errors="raise")
    events = pd.to_numeric(frame[event_col], errors="raise")
    if not np.isfinite(times).all() or not times.gt(0).all() or not events.isin([0, 1]).all():
        raise ValueError("Invalid PFS times/events in the training cohort.")
    quartiles = pd.qcut(times, q=4, labels=False, duplicates="drop")
    if quartiles.isna().any():
        raise ValueError("Survival-time quartiles cannot be formed from the supplied training times.")
    strata = events.astype(int).astype(str) + "_q" + quartiles.astype(int).astype(str)
    counts = strata.value_counts()
    if counts.min() < 2:
        raise ValueError("Event-by-time-quartile stratum has fewer than two patients; "
                         "do not silently substitute an unstratified split.")
    train_idx, tuning_idx = train_test_split(
        np.arange(len(frame)), test_size=fraction, random_state=seed, stratify=strata)
    result = frame.iloc[train_idx].copy(), frame.iloc[tuning_idx].copy()
    if any(not part[event_col].eq(1).any() for part in result):
        raise ValueError("Optimization and tuning subsets must each contain PFS events.")
    return result


def learning_rate_at_epoch(model_cfg: dict, epoch: int) -> float:
    warmup = int(model_cfg["warmup_epochs"])
    maximum = int(model_cfg["epochs"])
    base = float(model_cfg["learning_rate"])
    if epoch <= warmup:
        return base * epoch / max(1, warmup)
    progress = min(1.0, (epoch - warmup) / max(1, maximum - warmup))
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def fit_cohort_epoch(model, loader, optimizer, config, device, epoch):
    model_cfg = config["survival_model"]
    if model_cfg["cox_risk_set"] != "full_cohort":
        raise ValueError("Only complete-cohort Cox risk sets are supported.")
    lr = learning_rate_at_epoch(model_cfg, epoch)
    for group in optimizer.param_groups:
        group["lr"] = lr
    model.train()
    optimizer.zero_grad(set_to_none=True)
    risk, time, event = collect_risk_set(model, loader, device)
    loss = cox_breslow_loss(risk, time, event)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite full-risk-set training loss.")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(model_cfg["gradient_clip_norm"]),
                                   error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach().cpu()), lr


def train_one_model(model: nn.Module, train_frame: pd.DataFrame, validation_frame: pd.DataFrame, config: dict, device: torch.device):
    model_cfg = config["survival_model"]
    train_loader = DataLoader(
        NPZSurvivalDataset(train_frame, config, training=True),
        batch_size=int(model_cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        NPZSurvivalDataset(validation_frame, config, training=False),
        batch_size=int(model_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg["weight_decay"]),
    )
    model.to(device)
    best_state = None
    best_loss = float("inf")
    best_epoch = 1
    stale = 0
    history = []
    epochs = int(model_cfg["epochs"])
    patience = int(model_cfg["early_stopping_patience"])
    for epoch in range(1, epochs + 1):
        train_loss, lr = fit_cohort_epoch(model, train_loader, optimizer, config, device, epoch)
        validation_loss = evaluate_loss(model, validation_loader, device)
        if not np.isfinite(validation_loss):
            raise FloatingPointError("Nonfinite tuning loss; no checkpoint will be silently accepted.")
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": validation_loss, "learning_rate": lr})
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No valid tuning checkpoint was selected.")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def train_full_model(model: nn.Module, frame: pd.DataFrame, config: dict, device: torch.device, epochs: int):
    model_cfg = config["survival_model"]
    loader = DataLoader(
        NPZSurvivalDataset(frame, config, training=True),
        batch_size=int(model_cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg["weight_decay"]),
    )
    model.to(device)
    history = []
    for epoch in range(1, max(1, int(epochs)) + 1):
        loss, lr = fit_cohort_epoch(model, loader, optimizer, config, device, epoch)
        history.append({"epoch": epoch, "train_loss": loss, "learning_rate": lr})
    return model, history


def predict(model: nn.Module, frame: pd.DataFrame, config: dict, device: torch.device) -> np.ndarray:
    loader = DataLoader(NPZSurvivalDataset(frame, config, training=False), batch_size=1, shuffle=False, num_workers=0)
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for image, _, _, _ in loader:
            values.append(float(model(image.to(device)).cpu().item()))
    scores = np.asarray(values, dtype=float)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Frozen-model predictions contain nonfinite risk scores.")
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["random_seed"]))
    frame = read_manifest(args.manifest, config)
    if "processed_npz" not in frame.columns:
        raise ValueError("The survival manifest must contain the processed_npz column.")
    if not frame["processed_npz"].map(lambda value: Path(value).exists()).all():
        raise FileNotFoundError("At least one processed_npz path does not exist.")
    output_dir = args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    prediction_dir = output_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    device_cfg = str(config["project"]["device"]).lower()
    device = torch.device("cuda" if device_cfg == "auto" and torch.cuda.is_available() else "cpu")
    if device_cfg in {"cpu", "cuda"}:
        device = torch.device(device_cfg)
    patient_col = config["data"]["patient_id_column"]
    subtype_col = config["data"]["subtype_column"]
    cohort_col = config["data"]["cohort_column"]
    training_value = str(config["data"]["training_cohort"]).lower()
    training_all = frame[frame[cohort_col].astype(str).str.lower() == training_value].copy()
    if training_all.empty:
        raise ValueError("No training-cohort rows were found in the manifest.")
    # Precompute partitions once so comparator architectures see identical patients.
    # A stable subtype-derived seed cannot change when training histories change.
    partition_rows = []
    partitions = {}
    for subtype in sorted(training_all[subtype_col].astype(str).unique()):
        subtype_training = training_all[training_all[subtype_col].astype(str) == subtype].copy()
        subtype_seed = (int(config["project"]["random_seed"]) +
                        int(hashlib.sha256(subtype.encode("utf-8")).hexdigest()[:8], 16)) % (2**32)
        train_frame, validation_frame = split_frame(
            subtype_training, float(config["survival_model"]["validation_fraction"]), subtype_seed,
            config["data"]["time_column"], config["data"]["event_column"])
        partitions[subtype] = (train_frame, validation_frame, subtype_seed)
        for part_name, part in (("optimization", train_frame), ("tuning", validation_frame)):
            for patient_id in part[patient_col]:
                partition_rows.append({"patient_id": patient_id, "subtype": subtype,
                                       "partition": part_name, "seed": subtype_seed})
    pd.DataFrame(partition_rows).to_csv(output_dir / "model_development_partitions.csv", index=False)
    history_rows = []
    for model_name in config["survival_model"]["models"]:
        all_scores = []
        for subtype in sorted(frame[subtype_col].dropna().astype(str).unique()):
            subtype_training = training_all[training_all[subtype_col].astype(str) == subtype].copy()
            subtype_all = frame[frame[subtype_col].astype(str) == subtype].copy()
            if subtype_training.empty or subtype_all.empty:
                continue
            train_frame, validation_frame, subtype_seed = partitions[subtype]
            set_seed(subtype_seed)
            model = make_model(model_name, config)
            _, history, best_epoch = train_one_model(model, train_frame, validation_frame, config, device)
            for row in history:
                history_rows.append({"model": model_name, "subtype": subtype, "phase": "internal_validation", **row})
            set_seed(subtype_seed)
            model = make_model(model_name, config)
            model, final_history = train_full_model(model, subtype_training, config, device, best_epoch)
            for row in final_history:
                history_rows.append({"model": model_name, "subtype": subtype, "phase": "final_training", **row})
            scores = predict(model, subtype_all, config, device)
            scored = subtype_all[[patient_col, subtype_col, cohort_col]].copy()
            scored["risk_score"] = scores
            scored["model"] = model_name
            all_scores.append(scored)
            checkpoint = checkpoint_dir / f"{model_name}_{subtype.replace(' ', '_')}.pt"
            torch.save(
                {
                    "model_name": model_name,
                    "subtype": subtype,
                    "state_dict": model.state_dict(),
                    "config": config,
                    "device": str(device),
                    "architecture_version": "manuscript_20260908_occlusion_only",
                    "selected_epoch": best_epoch,
                    "cox_risk_set": "full_cohort",
                },
                checkpoint,
            )
        if all_scores:
            pd.concat(all_scores, ignore_index=True).to_csv(prediction_dir / f"{model_name}_predictions.csv", index=False)
    pd.DataFrame(history_rows).to_csv(output_dir / "training_history.csv", index=False)
    write_json(output_dir / "run_metadata.json", {"device": str(device), "config": config})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
