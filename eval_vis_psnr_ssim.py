import os
import sys
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def infer_num_cols(h: int, w: int, rows: int = 2) -> int:
    """
    根据图像尺寸自动推断列数。
    假设每个 tile 基本接近正方形，高度约为 h // rows。
    """
    tile_h = h // rows
    approx = w / tile_h
    cols = max(1, int(round(approx)))
    return cols


def split_grid_auto(img: np.ndarray, rows: int = 2):
    """
    自动切分 2 行 N 列网格。
    允许总宽度不能被列数整除，用 linspace 做近似分割。
    """
    h, w, c = img.shape
    assert h % rows == 0, f"height {h} not divisible by rows {rows}"

    cols = infer_num_cols(h, w, rows=rows)

    y_edges = np.linspace(0, h, rows + 1).round().astype(int)
    x_edges = np.linspace(0, w, cols + 1).round().astype(int)

    tiles = []
    for r in range(rows):
        row_tiles = []
        for cidx in range(cols):
            tile = img[y_edges[r]:y_edges[r + 1], x_edges[cidx]:x_edges[cidx + 1]]
            row_tiles.append(tile)
        tiles.append(row_tiles)

    return tiles, cols


def center_crop_to_same(a: np.ndarray, b: np.ndarray):
    """
    将两张 tile 中心裁到相同大小，避免因为 linspace 分割导致的 1~2 像素差异。
    """
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])

    def crop(img, h, w):
        y0 = (img.shape[0] - h) // 2
        x0 = (img.shape[1] - w) // 2
        return img[y0:y0 + h, x0:x0 + w]

    return crop(a, h, w), crop(b, h, w)


def compute_metrics(gt: np.ndarray, pred: np.ndarray):
    psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim = structural_similarity(gt, pred, channel_axis=2, data_range=1.0)
    return psnr, ssim


def process_one_image(fp: Path):
    img = Image.open(fp).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0

    tiles, cols = split_grid_auto(arr, rows=2)
    top_row = tiles[0]
    bottom_row = tiles[1]

    file_psnr = []
    file_ssim = []

    for a, b in zip(top_row, bottom_row):
        a, b = center_crop_to_same(a, b)

        # 太窄或空 tile 直接跳过
        if a.size == 0 or b.size == 0:
            continue
        if a.shape[0] < 8 or a.shape[1] < 8:
            continue

        psnr, ssim = compute_metrics(a, b)
        file_psnr.append(psnr)
        file_ssim.append(ssim)

    if not file_psnr:
        return cols, None, None

    return cols, float(np.mean(file_psnr)), float(np.mean(file_ssim))


def main(path_str: str):
    path = Path(path_str)

    if path.is_file():
        files = [path]
    else:
        files = sorted([p for p in path.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    if not files:
        print(f"No image files found in {path}")
        return

    all_psnr = []
    all_ssim = []

    for fp in files:
        try:
            cols, mean_psnr, mean_ssim = process_one_image(fp)
            if mean_psnr is None:
                print(f"{fp.name}: skipped (no valid tiles), inferred_cols={cols}")
                continue

            all_psnr.append(mean_psnr)
            all_ssim.append(mean_ssim)
            print(f"{fp.name}: inferred_cols={cols}, PSNR={mean_psnr:.4f}, SSIM={mean_ssim:.4f}")

        except Exception as e:
            print(f"{fp.name}: failed -> {e}")

    print("-" * 60)
    if all_psnr:
        print(f"Overall mean PSNR: {np.mean(all_psnr):.4f}")
        print(f"Overall mean SSIM: {np.mean(all_ssim):.4f}")
    else:
        print("No valid results.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval_vis_psnr_ssim.py <image_or_dir>")
        sys.exit(1)
    main(sys.argv[1])