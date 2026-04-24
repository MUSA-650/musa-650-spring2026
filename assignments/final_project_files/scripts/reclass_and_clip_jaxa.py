#!/usr/bin/env python3
"""Clip JAXA land-cover tiles to an AOI and remap 15 classes into 5 classes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fiona
import numpy as np
import rasterio
import yaml
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import transform_geom
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with JAXA GeoTIFF tiles.")
    parser.add_argument("--aoi", type=Path, required=True, help="AOI vector file, such as SHP or GeoJSON.")
    parser.add_argument("--class-map", type=Path, required=True, help="YAML config with raw_to_target mapping.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--mosaic-name",
        default="aoi_5class_mosaic.tif",
        help="Filename for the merged AOI mosaic.",
    )
    return parser.parse_args()


def load_mapping(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["raw_to_target"] = {int(k): int(v) for k, v in config["raw_to_target"].items()}
    config["target_names"] = {int(k): str(v) for k, v in config["target_names"].items()}
    return config


def load_aoi(path: Path) -> tuple[list[dict], object]:
    with fiona.open(path) as src:
        geometries = [feature["geometry"] for feature in src if feature.get("geometry")]
        if not geometries:
            raise ValueError(f"No valid geometry found in AOI file: {path}")
        aoi_crs = src.crs_wkt or src.crs
    return geometries, aoi_crs


def remap_labels(labels: np.ndarray, raw_to_target: dict[int, int]) -> np.ndarray:
    output = np.zeros(labels.shape, dtype=np.uint8)
    for raw_class, target_class in raw_to_target.items():
        output[labels == raw_class] = target_class
    return output


def clip_and_reclassify(
    tif_path: Path,
    geometries: list[dict],
    aoi_crs: object,
    raw_to_target: dict[int, int],
    output_path: Path,
) -> bool:
    with rasterio.open(tif_path) as src:
        transformed_geometries = geometries
        if aoi_crs and src.crs and str(aoi_crs) != str(src.crs):
            transformed_geometries = [
                transform_geom(aoi_crs, src.crs, geometry) for geometry in geometries
            ]

        try:
            clipped, transform = mask(src, transformed_geometries, crop=True)
        except ValueError:
            return False

        labels = clipped[0]
        mapped = remap_labels(labels, raw_to_target)
        if np.count_nonzero(mapped) == 0:
            return False

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=mapped.shape[0],
            width=mapped.shape[1],
            transform=transform,
            count=1,
            dtype="uint8",
            nodata=0,
            compress="lzw",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mapped, 1)
        return True


def build_mosaic(tile_paths: list[Path], output_path: Path) -> None:
    datasets = [rasterio.open(path) for path in tile_paths]
    try:
        mosaic, transform = merge(datasets, method="first")
        profile = datasets[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            count=1,
            dtype="uint8",
            nodata=0,
            compress="lzw",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic[0], 1)
    finally:
        for dataset in datasets:
            dataset.close()


def save_legend(config: dict, output_dir: Path) -> None:
    legend = {
        "target_names": config["target_names"],
        "raw_to_target": config["raw_to_target"],
        "ignored_raw_classes": config.get("ignored_raw_classes", []),
        "nodata": 0,
    }
    with (output_dir / "class_legend.json").open("w", encoding="utf-8") as f:
        json.dump(legend, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    config = load_mapping(args.class_map)
    geometries, aoi_crs = load_aoi(args.aoi)

    input_tifs = sorted(args.input_dir.glob("*.tif"))
    if not input_tifs:
        raise FileNotFoundError(f"No tif files found under: {args.input_dir}")

    clipped_dir = args.output_dir / "clipped_tiles"
    clipped_paths: list[Path] = []

    for tif_path in tqdm(input_tifs, desc="Processing JAXA tiles"):
        out_path = clipped_dir / f"{tif_path.stem}_5class.tif"
        written = clip_and_reclassify(
            tif_path=tif_path,
            geometries=geometries,
            aoi_crs=aoi_crs,
            raw_to_target=config["raw_to_target"],
            output_path=out_path,
        )
        if written:
            clipped_paths.append(out_path)

    if not clipped_paths:
        raise RuntimeError("No intersecting tiles produced output. Check the AOI and CRS.")

    mosaic_path = args.output_dir / args.mosaic_name
    build_mosaic(clipped_paths, mosaic_path)
    save_legend(config, args.output_dir)

    print(f"Wrote {len(clipped_paths)} clipped tiles")
    print(f"Mosaic: {mosaic_path}")


if __name__ == "__main__":
    main()
