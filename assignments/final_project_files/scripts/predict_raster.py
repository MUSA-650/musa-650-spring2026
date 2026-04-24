#!/usr/bin/env python3
"""Run sliding-window inference over a multi-band raster and output a 5-class GeoTIFF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from rasterio.windows import Window
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-raster", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-raster", type=Path, required=True)
    parser.add_argument("--model", choices=["unet", "deeplabv3plus"], default="unet")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument(
        "--normalization",
        choices=["patch", "global"],
        default="patch",
        help="Use per-tile normalization or fixed train-set statistics.",
    )
    parser.add_argument("--norm-mean", type=Path, default=None, help="Path to .npy per-band mean.")
    parser.add_argument("--norm-std", type=Path, default=None, help="Path to .npy per-band std.")
    return parser.parse_args()


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


def load_norm_array(path: Path, channels: int) -> np.ndarray:
    values = np.load(path).astype(np.float32).reshape(-1)
    if values.size != channels:
        raise ValueError(f"{path} has {values.size} values, expected {channels}")
    return values.reshape(channels, 1, 1)


def normalize_tile(
    tile: np.ndarray,
    normalization: str,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    if normalization == "global":
        if mean is None or std is None:
            raise ValueError("Global normalization requires mean and std")
    else:
        mean = tile.mean(axis=(1, 2), keepdims=True)
        std = tile.std(axis=(1, 2), keepdims=True)
    return (tile - mean) / np.clip(std, 1e-6, None)


def iterate_positions(height: int, width: int, tile_size: int, stride: int) -> list[tuple[int, int]]:
    rows = list(range(0, max(height - tile_size, 0) + 1, stride))
    cols = list(range(0, max(width - tile_size, 0) + 1, stride))

    if not rows or rows[-1] != max(height - tile_size, 0):
        rows.append(max(height - tile_size, 0))
    if not cols or cols[-1] != max(width - tile_size, 0):
        cols.append(max(width - tile_size, 0))

    return [(row, col) for row in rows for col in cols]


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = build_model(args).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    norm_mean = norm_std = None
    if args.normalization == "global":
        if args.norm_mean is None or args.norm_std is None:
            raise ValueError("--normalization global requires --norm-mean and --norm-std")
        norm_mean = load_norm_array(args.norm_mean, args.in_channels)
        norm_std = load_norm_array(args.norm_std, args.in_channels)

    with rasterio.open(args.input_raster) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        positions = iterate_positions(height, width, args.tile_size, args.stride)

        prob_sum = np.zeros((args.num_classes, height, width), dtype=np.float32)
        count_sum = np.zeros((height, width), dtype=np.float32)

        with torch.no_grad():
            for row, col in tqdm(positions, desc="Predicting"):
                window = Window(col, row, min(args.tile_size, width - col), min(args.tile_size, height - row))
                chip = src.read(window=window).astype(np.float32)

                padded = np.zeros((args.in_channels, args.tile_size, args.tile_size), dtype=np.float32)
                padded[:, : chip.shape[1], : chip.shape[2]] = chip[: args.in_channels]
                padded = normalize_tile(
                    padded,
                    normalization=args.normalization,
                    mean=norm_mean,
                    std=norm_std,
                )

                tensor = torch.from_numpy(padded[None, ...]).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                valid_h, valid_w = chip.shape[1], chip.shape[2]
                prob_sum[:, row : row + valid_h, col : col + valid_w] += probs[:, :valid_h, :valid_w]
                count_sum[row : row + valid_h, col : col + valid_w] += 1.0

        count_sum = np.clip(count_sum, 1e-6, None)
        mean_probs = prob_sum / count_sum[None, :, :]
        pred = np.argmax(mean_probs, axis=0).astype(np.uint8) + 1

        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="lzw")
        args.output_raster.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(args.output_raster, "w", **profile) as dst:
            dst.write(pred, 1)

    print(f"Saved prediction raster to: {args.output_raster}")


if __name__ == "__main__":
    main()
