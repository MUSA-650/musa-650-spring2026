#!/usr/bin/env python3
"""Evaluate a trained U-Net/DeepLab model on a manifest split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from rasterio.windows import Window
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--model", choices=["unet", "deeplabv3plus"], default="unet")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--in-channels", type=int, default=24)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument(
        "--normalization",
        choices=["patch", "global"],
        default="patch",
        help="Use per-patch normalization or fixed train-set statistics.",
    )
    parser.add_argument("--norm-mean", type=Path, default=None, help="Path to .npy per-band mean.")
    parser.add_argument("--norm-std", type=Path, default=None, help="Path to .npy per-band std.")
    return parser.parse_args()


def load_ids(path: Path, max_patches: int | None = None) -> set[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_patches is not None:
        ids = ids[:max_patches]
    return set(ids)


def load_norm_array(path: Path, channels: int) -> np.ndarray:
    values = np.load(path).astype(np.float32).reshape(-1)
    if values.size != channels:
        raise ValueError(f"{path} has {values.size} values, expected {channels}")
    return values.reshape(channels, 1, 1)


class WindowDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        max_patches: int | None = None,
        normalization: str = "patch",
        norm_mean: np.ndarray | None = None,
        norm_std: np.ndarray | None = None,
    ) -> None:
        self.normalization = normalization
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.ids = load_ids(root / "splits" / f"{split}.txt", max_patches)
        self.rows = []
        with (root / "manifest.csv").open("r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["tile_id"] in self.ids:
                    self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        tile_size = int(row["tile_size"])
        win = Window(int(row["col"]), int(row["row"]), tile_size, tile_size)

        with rasterio.open(row["image_path"]) as img_ds:
            image = img_ds.read(window=win).astype(np.float32)
        with rasterio.open(row["label_path"]) as lbl_ds:
            label = lbl_ds.read(1, window=win).astype(np.int64)

        if self.normalization == "global":
            if self.norm_mean is None or self.norm_std is None:
                raise ValueError("Global normalization requires norm_mean and norm_std")
            mean = self.norm_mean
            std = self.norm_std
        else:
            mean = image.mean(axis=(1, 2), keepdims=True)
            std = image.std(axis=(1, 2), keepdims=True)
        image = (image - mean) / np.clip(std, 1e-6, None)
        label = np.where(label == 0, 255, label - 1).astype(np.int64)

        return torch.from_numpy(image), torch.from_numpy(label)


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    encoder_weights = "imagenet" if args.in_channels == 3 else None
    if args.model == "unet":
        return smp.Unet(
            encoder_name=args.encoder,
            encoder_weights=encoder_weights,
            in_channels=args.in_channels,
            classes=args.num_classes,
        )
    return smp.DeepLabV3Plus(
        encoder_name=args.encoder,
        encoder_weights=encoder_weights,
        in_channels=args.in_channels,
        classes=args.num_classes,
    )


def compute_iou(cm: np.ndarray) -> tuple[list[float | None], float]:
    ious: list[float | None] = []
    for cls in range(cm.shape[0]):
        intersection = cm[cls, cls]
        union = cm[cls, :].sum() + cm[:, cls].sum() - intersection
        ious.append(None if union == 0 else float(intersection / union))
    valid = [iou for iou in ious if iou is not None]
    return ious, float(np.mean(valid)) if valid else 0.0


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    norm_mean = norm_std = None
    if args.normalization == "global":
        if args.norm_mean is None or args.norm_std is None:
            raise ValueError("--normalization global requires --norm-mean and --norm-std")
        norm_mean = load_norm_array(args.norm_mean, args.in_channels)
        norm_std = load_norm_array(args.norm_std, args.in_channels)

    dataset = WindowDataset(
        args.data_dir,
        args.split,
        args.max_patches,
        normalization=args.normalization,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Evaluating {args.split}"):
            images = images.to(device)
            logits = model(images)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            target = labels.numpy()
            valid = target != 255
            y_true_chunks.append(target[valid].reshape(-1))
            y_pred_chunks.append(pred[valid].reshape(-1))

    y_true = np.concatenate(y_true_chunks)
    y_pred = np.concatenate(y_pred_chunks)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    per_class_iou, miou = compute_iou(cm)

    metrics = {
        "model": args.model,
        "split": args.split,
        "patches": len(dataset),
        "pixels": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2, 3, 4])),
        "per_class_iou": per_class_iou,
        "miou": miou,
        "confusion_matrix": cm.tolist(),
        "class_names": ["Forest", "Plantation", "Cropland", "Human-use", "Water/Wetland"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
