#!/usr/bin/env python3
"""Train a pixel-level Random Forest baseline from manifest windows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with manifest.csv and splits.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-patches", type=int, default=1200)
    parser.add_argument("--val-patches", type=int, default=400)
    parser.add_argument("--pixels-per-patch", type=int, default=512)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_ids(path: Path, limit: int | None = None) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids if limit is None else ids[:limit]


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csvfile:
        return {row["tile_id"]: row for row in csv.DictReader(csvfile)}


def sample_pixels_from_window(
    row: dict[str, str],
    pixels_per_patch: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    tile_size = int(row["tile_size"])
    win = Window(int(row["col"]), int(row["row"]), tile_size, tile_size)

    with rasterio.open(row["image_path"]) as img_ds:
        image = img_ds.read(window=win).astype(np.float32)
    with rasterio.open(row["label_path"]) as lbl_ds:
        label = lbl_ds.read(1, window=win).astype(np.uint8)

    valid = np.flatnonzero(label.reshape(-1) > 0)
    if valid.size == 0:
        return np.empty((0, image.shape[0]), dtype=np.float32), np.empty((0,), dtype=np.uint8)

    n = min(pixels_per_patch, valid.size)
    chosen = rng.choice(valid, size=n, replace=False)

    flat_image = image.reshape(image.shape[0], -1).T
    x = flat_image[chosen]
    y = label.reshape(-1)[chosen] - 1
    return x, y


def build_samples(
    manifest: dict[str, dict[str, str]],
    ids: list[str],
    pixels_per_patch: int,
    seed: int,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for tile_id in tqdm(ids, desc=desc):
        x, y = sample_pixels_from_window(manifest[tile_id], pixels_per_patch, rng)
        if x.size == 0:
            continue
        xs.append(x)
        ys.append(y)

    if not xs:
        raise RuntimeError(f"No valid pixels found for {desc}")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def compute_iou(cm: np.ndarray) -> tuple[list[float | None], float]:
    ious: list[float | None] = []
    for cls in range(cm.shape[0]):
        intersection = cm[cls, cls]
        union = cm[cls, :].sum() + cm[:, cls].sum() - intersection
        ious.append(None if union == 0 else float(intersection / union))
    valid_ious = [iou for iou in ious if iou is not None]
    return ious, float(np.mean(valid_ious)) if valid_ious else 0.0


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.data_dir / "manifest.csv")
    train_ids = read_ids(args.data_dir / "splits" / "train.txt", args.train_patches)
    val_ids = read_ids(args.data_dir / "splits" / "val.txt", args.val_patches)

    x_train, y_train = build_samples(
        manifest,
        train_ids,
        pixels_per_patch=args.pixels_per_patch,
        seed=args.seed,
        desc="Sampling train pixels",
    )
    x_val, y_val = build_samples(
        manifest,
        val_ids,
        pixels_per_patch=args.pixels_per_patch,
        seed=args.seed + 1,
        desc="Sampling val pixels",
    )

    # Per-pixel baseline: normalize each feature globally for numerical stability.
    mean = x_train.mean(axis=0, keepdims=True)
    std = np.clip(x_train.std(axis=0, keepdims=True), 1e-6, None)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
        verbose=1,
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_val)

    cm = confusion_matrix(y_val, pred, labels=[0, 1, 2, 3, 4])
    per_class_iou, miou = compute_iou(cm)
    metrics = {
        "model": "RandomForestClassifier",
        "train_pixels": int(x_train.shape[0]),
        "val_pixels": int(x_val.shape[0]),
        "n_features": int(x_train.shape[1]),
        "accuracy": float(accuracy_score(y_val, pred)),
        "macro_f1": float(f1_score(y_val, pred, average="macro", labels=[0, 1, 2, 3, 4])),
        "per_class_iou": per_class_iou,
        "miou": miou,
        "confusion_matrix": cm.tolist(),
        "class_names": ["Forest", "Plantation", "Cropland", "Human-use", "Water/Wetland"],
    }

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(clf, args.output_dir / "random_forest.joblib")
    np.save(args.output_dir / "feature_mean.npy", mean)
    np.save(args.output_dir / "feature_std.npy", std)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
