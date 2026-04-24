#!/usr/bin/env python3
"""Export aligned imagery and label rasters into training patches."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import fiona
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_geom
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from shapely.geometry import box, shape
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True, help="Aligned multi-band imagery GeoTIFFs.")
    parser.add_argument("--label-dir", type=Path, required=True, help="Aligned single-band label GeoTIFFs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for exported .npy tiles.")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-labeled-ratio",
        type=float,
        default=0.05,
        help="Minimum ratio of non-zero label pixels required to keep a tile.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["random", "block"],
        default="block",
        help="Use block split to reduce spatial leakage between train and validation tiles.",
    )
    parser.add_argument(
        "--block-size-multiplier",
        type=int,
        default=4,
        help="When split-mode=block, neighboring tiles are grouped into blocks of this many tile sizes.",
    )
    parser.add_argument(
        "--exclude-aoi",
        type=Path,
        default=None,
        help="Optional AOI vector. Tiles intersecting this geometry will be excluded.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Do not save .npy tiles. Save a CSV manifest of window locations instead.",
    )
    return parser.parse_args()


def list_pairs(image_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    images = sorted(image_dir.rglob("*.tif"))
    labels = {path.stem: path for path in label_dir.rglob("*.tif")}
    pairs = []
    for image_path in images:
        label_path = labels.get(image_path.stem)
        if label_path is not None:
            pairs.append((image_path, label_path))
    return pairs


def write_split(ids: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def split_tile_ids(
    tile_ids: list[str],
    groups: list[str],
    val_ratio: float,
    seed: int,
    split_mode: str,
) -> tuple[list[str], list[str]]:
    if split_mode == "random":
        return train_test_split(
            tile_ids,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    indices = np.arange(len(tile_ids))
    train_idx, val_idx = next(splitter.split(indices, groups=groups))
    train_ids = [tile_ids[i] for i in train_idx]
    val_ids = [tile_ids[i] for i in val_idx]
    return train_ids, val_ids


def load_exclude_geometries(path: Path, dst_crs) -> list:
    with fiona.open(path) as src:
        src_crs = src.crs_wkt or src.crs
        geoms = [feature["geometry"] for feature in src if feature.get("geometry")]
    if not geoms:
        return []
    return [shape(transform_geom(src_crs, dst_crs, geom)) for geom in geoms]


def main() -> None:
    args = parse_args()
    pairs = list_pairs(args.image_dir, args.label_dir)
    if not pairs:
        raise RuntimeError("No image/label tif pairs found with matching file stems.")

    images_out = args.output_dir / "images"
    labels_out = args.output_dir / "labels"
    if not args.manifest_only:
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

    tile_ids: list[str] = []
    groups: list[str] = []
    manifest_rows: list[list[str | int]] = []

    for image_path, label_path in tqdm(pairs, desc="Exporting patches"):
        with rasterio.open(image_path) as img_ds, rasterio.open(label_path) as lbl_ds:
            if img_ds.width != lbl_ds.width or img_ds.height != lbl_ds.height:
                raise ValueError(f"Shape mismatch between {image_path.name} and {label_path.name}")
            if img_ds.transform != lbl_ds.transform:
                raise ValueError(f"Transform mismatch between {image_path.name} and {label_path.name}")
            if img_ds.crs != lbl_ds.crs:
                raise ValueError(f"CRS mismatch between {image_path.name} and {label_path.name}")

            exclude_geoms = []
            if args.exclude_aoi is not None:
                exclude_geoms = load_exclude_geometries(args.exclude_aoi, img_ds.crs)

            for row in range(0, img_ds.height - args.tile_size + 1, args.stride):
                for col in range(0, img_ds.width - args.tile_size + 1, args.stride):
                    window = Window(col, row, args.tile_size, args.tile_size)
                    if exclude_geoms:
                        bounds = rasterio.windows.bounds(window, img_ds.transform)
                        window_geom = box(*bounds)
                        if any(window_geom.intersects(geom) for geom in exclude_geoms):
                            continue

                    image = img_ds.read(window=window).astype(np.float32)
                    label = lbl_ds.read(1, window=window).astype(np.uint8)

                    labeled_ratio = float(np.count_nonzero(label)) / float(label.size)
                    if labeled_ratio < args.min_labeled_ratio:
                        continue

                    tile_id = f"{image_path.stem}_r{row}_c{col}"
                    if args.split_mode == "block":
                        block_span = args.tile_size * args.block_size_multiplier
                        group_id = (
                            f"{image_path.stem}_br{row // block_span}_bc{col // block_span}"
                        )
                    else:
                        group_id = image_path.stem

                    if not args.manifest_only:
                        np.save(images_out / f"{tile_id}.npy", image)
                        np.save(labels_out / f"{tile_id}.npy", label)

                    manifest_rows.append([
                        tile_id,
                        str(image_path.resolve()),
                        str(label_path.resolve()),
                        row,
                        col,
                        args.tile_size,
                    ])
                    tile_ids.append(tile_id)
                    groups.append(group_id)

    if not tile_ids:
        raise RuntimeError("No valid training tiles were exported.")

    train_ids, val_ids = split_tile_ids(
        tile_ids,
        groups,
        val_ratio=args.val_ratio,
        seed=args.seed,
        split_mode=args.split_mode,
    )

    write_split(train_ids, args.output_dir / "splits" / "train.txt")
    write_split(val_ids, args.output_dir / "splits" / "val.txt")

    manifest_path = args.output_dir / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tile_id", "image_path", "label_path", "row", "col", "tile_size"])
        writer.writerows(manifest_rows)

    print(f"Exported {len(tile_ids)} tiles")
    print(f"Train tiles: {len(train_ids)}")
    print(f"Val tiles: {len(val_ids)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
