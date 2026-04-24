#!/usr/bin/env python3
"""Reproject a label raster onto the exact grid of an imagery raster."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Reference imagery GeoTIFF.")
    parser.add_argument("--label", type=Path, required=True, help="Input label GeoTIFF to be aligned.")
    parser.add_argument("--output", type=Path, required=True, help="Aligned output label GeoTIFF.")
    parser.add_argument("--label-nodata", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with rasterio.open(args.image) as img_ds, rasterio.open(args.label) as lbl_ds:
        destination = np.full((img_ds.height, img_ds.width), args.label_nodata, dtype=np.uint8)

        reproject(
            source=rasterio.band(lbl_ds, 1),
            destination=destination,
            src_transform=lbl_ds.transform,
            src_crs=lbl_ds.crs,
            src_nodata=lbl_ds.nodata if lbl_ds.nodata is not None else args.label_nodata,
            dst_transform=img_ds.transform,
            dst_crs=img_ds.crs,
            dst_nodata=args.label_nodata,
            resampling=Resampling.nearest,
        )

        profile = img_ds.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=args.label_nodata,
            compress="lzw",
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(args.output, "w", **profile) as dst:
            dst.write(destination, 1)

    print(f"Aligned label saved to: {args.output}")


if __name__ == "__main__":
    main()
