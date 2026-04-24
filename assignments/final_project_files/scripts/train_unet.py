#!/usr/bin/env python3
"""Train a simple semantic segmentation baseline on exported .npy patches."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory created by tile_training_data.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["unet", "deeplabv3plus"], default="unet")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalization",
        choices=["patch", "global"],
        default="patch",
        help="Use per-patch normalization or fixed train-set statistics.",
    )
    parser.add_argument("--norm-mean", type=Path, default=None, help="Path to .npy per-band mean.")
    parser.add_argument("--norm-std", type=Path, default=None, help="Path to .npy per-band std.")
    parser.add_argument(
        "--class-weights",
        default=None,
        help="Comma-separated class weights for classes 0..num_classes-1.",
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=0.0,
        help="Optional Dice loss multiplier added to cross entropy.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ids(split_file: Path) -> list[str]:
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_norm_array(path: Path, channels: int) -> np.ndarray:
    values = np.load(path).astype(np.float32).reshape(-1)
    if values.size != channels:
        raise ValueError(f"{path} has {values.size} values, expected {channels}")
    return values.reshape(channels, 1, 1)


class PatchDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        train: bool = False,
        normalization: str = "patch",
        norm_mean: np.ndarray | None = None,
        norm_std: np.ndarray | None = None,
    ) -> None:
        self.root = root
        self.train = train
        self.normalization = normalization
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.ids = load_ids(root / "splits" / f"{split}.txt")
        self.image_dir = root / "images"
        self.label_dir = root / "labels"

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile_id = self.ids[index]
        image = np.load(self.image_dir / f"{tile_id}.npy").astype(np.float32)
        label = np.load(self.label_dir / f"{tile_id}.npy").astype(np.int64)

        if image.ndim == 2:
            image = image[None, :, :]
        elif image.ndim == 3 and image.shape[0] > image.shape[-1]:
            pass
        elif image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))

        if self.train and np.random.rand() < 0.5:
            image = image[:, :, ::-1].copy()
            label = label[:, ::-1].copy()
        if self.train and np.random.rand() < 0.5:
            image = image[:, ::-1, :].copy()
            label = label[::-1, :].copy()

        if self.normalization == "global":
            if self.norm_mean is None or self.norm_std is None:
                raise ValueError("Global normalization requires norm_mean and norm_std")
            mean = self.norm_mean
            std = self.norm_std
        else:
            mean = image.mean(axis=(1, 2), keepdims=True)
            std = image.std(axis=(1, 2), keepdims=True)
        image = (image - mean) / np.clip(std, 1e-6, None)

        # Labels come in as: 0 ignore, 1..5 valid classes.
        label = np.where(label == 0, 255, label - 1).astype(np.int64)

        return torch.from_numpy(image), torch.from_numpy(label)


class WindowDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        train: bool = False,
        normalization: str = "patch",
        norm_mean: np.ndarray | None = None,
        norm_std: np.ndarray | None = None,
    ) -> None:
        self.root = root
        self.train = train
        self.normalization = normalization
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.ids = set(load_ids(root / "splits" / f"{split}.txt"))
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
        win_row = int(row["row"])
        win_col = int(row["col"])

        with rasterio.open(row["image_path"]) as img_ds:
            image = img_ds.read(
                window=rasterio.windows.Window(win_col, win_row, tile_size, tile_size)
            ).astype(np.float32)
        with rasterio.open(row["label_path"]) as lbl_ds:
            label = lbl_ds.read(
                1,
                window=rasterio.windows.Window(win_col, win_row, tile_size, tile_size),
            ).astype(np.int64)

        if self.train and np.random.rand() < 0.5:
            image = image[:, :, ::-1].copy()
            label = label[:, ::-1].copy()
        if self.train and np.random.rand() < 0.5:
            image = image[:, ::-1, :].copy()
            label = label[::-1, :].copy()

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


def build_model(args: argparse.Namespace) -> nn.Module:
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


def compute_mean_iou(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    pred = torch.argmax(logits, dim=1)
    valid = target != 255
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred == cls) & valid
        target_mask = target == cls
        union = (pred_mask | target_mask).sum().item()
        if union == 0:
            continue
        intersection = (pred_mask & target_mask).sum().item()
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def parse_class_weights(weights: str | None, num_classes: int, device: torch.device) -> torch.Tensor | None:
    if weights is None:
        return None
    values = [float(value.strip()) for value in weights.split(",") if value.strip()]
    if len(values) != num_classes:
        raise ValueError(f"Expected {num_classes} class weights, got {len(values)}")
    return torch.tensor(values, dtype=torch.float32, device=device)


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
        dice_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.ce(logits, target)
        if self.dice_weight <= 0:
            return loss

        valid = target != 255
        safe_target = target.masked_fill(~valid, 0)
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(safe_target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1)
        probs = probs * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        denominator = probs.sum(dims) + one_hot.sum(dims)
        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        return loss + self.dice_weight * (1.0 - dice.mean())


def update_confusion_matrix(
    cm: np.ndarray,
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> None:
    pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
    labels = target.detach().cpu().numpy()
    valid = labels != 255
    if not np.any(valid):
        return
    encoded = labels[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    counts = np.bincount(encoded, minlength=num_classes * num_classes)
    cm += counts.reshape(num_classes, num_classes)


def iou_from_confusion(cm: np.ndarray) -> tuple[list[float], float]:
    per_class = []
    for cls in range(cm.shape[0]):
        intersection = cm[cls, cls]
        union = cm[cls, :].sum() + cm[:, cls].sum() - intersection
        per_class.append(float(intersection / union) if union > 0 else 0.0)
    return per_class, float(np.mean(per_class)) if per_class else 0.0


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, list[float]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_batches = 0
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)

    autocast_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_enabled and train_mode)

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        if train_mode:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        update_confusion_matrix(cm, logits, labels, num_classes)
        total_batches += 1

    per_class_iou, miou = iou_from_confusion(cm)
    return total_loss / total_batches, miou, per_class_iou


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    if (args.data_dir / "manifest.csv").exists():
        train_ds = WindowDataset(
            args.data_dir,
            split="train",
            train=True,
            normalization=args.normalization,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
        val_ds = WindowDataset(
            args.data_dir,
            split="val",
            train=False,
            normalization=args.normalization,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
    else:
        train_ds = PatchDataset(
            args.data_dir,
            split="train",
            train=True,
            normalization=args.normalization,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
        val_ds = PatchDataset(
            args.data_dir,
            split="val",
            train=False,
            normalization=args.normalization,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args).to(device)
    class_weights = parse_class_weights(args.class_weights, args.num_classes, device)
    criterion = SegmentationLoss(
        num_classes=args.num_classes,
        class_weights=class_weights,
        dice_weight=args.dice_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    metrics_path = args.output_dir / "metrics.csv"
    class_names = ["Forest", "Plantation", "Cropland", "Human-use", "Water/Wetland"]
    best_iou = -1.0

    with metrics_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_miou",
                "val_loss",
                "val_miou",
                *[f"val_iou_{name}" for name in class_names],
            ]
        )

        for epoch in range(1, args.epochs + 1):
            train_loss, train_iou, _ = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                num_classes=args.num_classes,
            )
            val_loss, val_iou, val_per_class_iou = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                criterion=criterion,
                device=device,
                num_classes=args.num_classes,
            )

            writer.writerow([epoch, train_loss, train_iou, val_loss, val_iou, *val_per_class_iou])
            csvfile.flush()

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} train_mIoU={train_iou:.4f} | "
                f"val_loss={val_loss:.4f} val_mIoU={val_iou:.4f}"
            )

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), args.output_dir / "best_model.pt")

    print(f"Best val mIoU: {best_iou:.4f}")
    print(f"Saved model to: {args.output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
