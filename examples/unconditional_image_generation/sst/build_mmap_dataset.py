#!/usr/bin/env python3
# build_mmap_dataset.py
#
# 将多个 .pt 缓存文件合并为一个连续的 mmap 二进制文件，
# 供 MmapSSTDataset 以极低内存占用进行随机访问。
#
# 输出文件：
#   {output_dir}/train_data.bin    —— 纯 float16 tensor 数据，[N, 4, H, W]
#   {output_dir}/train_meta.bin    —— 纯 float32 元数据，[N, 3]（lat, lon, doy）
#   {output_dir}/train_index.json  —— 索引文件（patch 数量、shape、文件来源等）
#
# 使用方法：
#   python build_mmap_dataset.py \
#       --data_root /root/autodl-tmp/pt_cache \
#       --years 2020 2021 2022 \
#       --output_dir /root/autodl-fs/mmap_train
#
# 训练前将输出目录移到数据盘：
#   mv /root/autodl-fs/mmap_train /root/autodl-tmp/mmap_train

import argparse
import json
import os
from glob import glob

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Build mmap dataset from .pt cache files")
    parser.add_argument("--data_root",  type=str, required=True,
                        help="包含 YYYY_MM.pt 缓存文件的目录")
    parser.add_argument("--years",      type=int, nargs="+", required=True,
                        help="要合并的年份，如 2020 2021 2022")
    parser.add_argument("--months",     type=int, nargs="+", default=None,
                        help="月份过滤，默认全年")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录，会生成 train_data.bin / train_meta.bin / train_index.json")
    parser.add_argument("--prefix",     type=str, default="train",
                        help="输出文件名前缀，默认 train，可改为 val 等")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data_path  = os.path.join(args.output_dir, f"{args.prefix}_data.bin")
    meta_path  = os.path.join(args.output_dir, f"{args.prefix}_meta.bin")
    index_path = os.path.join(args.output_dir, f"{args.prefix}_index.json")

    # ── 扫描符合条件的 .pt 文件 ─────────────────────────────────
    year_set  = set(str(y) for y in args.years)
    month_set = set(args.months) if args.months is not None else None

    pt_files = []
    for pt_path in sorted(glob(os.path.join(args.data_root, "*.pt"))):
        basename = os.path.basename(pt_path)
        parts    = basename.replace(".pt", "").split("_")
        if len(parts) != 2:
            continue
        year_str, month_str = parts
        if year_str not in year_set:
            continue
        if month_set is not None:
            try:
                if int(month_str) not in month_set:
                    continue
            except ValueError:
                continue
        if "val_fixed" in basename:
            continue
        pt_files.append(pt_path)

    if len(pt_files) == 0:
        raise FileNotFoundError(
            f"No .pt files found in {args.data_root} for years={args.years}"
        )

    print(f"Found {len(pt_files)} .pt files to merge.")

    # ── 第一遍扫描：统计总 patch 数和 tensor shape ───────────────
    print("\nPass 1: scanning patch counts...")
    file_info = []   # list of (pt_path, n_patches)
    total_n   = 0
    patch_shape = None  # (C, H, W)

    for pt_path in pt_files:
        try:
            payload = torch.load(pt_path, map_location="cpu", weights_only=True)
        except Exception:
            payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        n = payload["tensors"].shape[0]
        if patch_shape is None:
            patch_shape = tuple(payload["tensors"].shape[1:])  # (4, H, W)
        del payload
        file_info.append((pt_path, n))
        total_n += n
        print(f"  {os.path.basename(pt_path)}: {n} patches")

    C, H, W = patch_shape
    print(f"\nTotal patches: {total_n}, shape per patch: [{C}, {H}, {W}]")

    # ── 预分配 mmap 文件 ─────────────────────────────────────────
    # train_data.bin: float16, shape [total_n, C, H, W]
    # train_meta.bin: float32, shape [total_n, 3]  (lat, lon, doy)
    data_size_gb = total_n * C * H * W * 2 / 1024**3
    meta_size_mb = total_n * 3 * 4 / 1024**2
    print(f"\nAllocating:")
    print(f"  {data_path}  ({data_size_gb:.2f} GB)")
    print(f"  {meta_path}  ({meta_size_mb:.1f} MB)")

    data_mmap = np.memmap(data_path, dtype=np.float16, mode="w+",
                          shape=(total_n, C, H, W))
    meta_mmap = np.memmap(meta_path, dtype=np.float32, mode="w+",
                          shape=(total_n, 3))

    # ── 第二遍：逐文件写入 mmap ──────────────────────────────────
    print("\nPass 2: writing data...")
    offset = 0
    source_files = []

    for pt_path, n in file_info:
        try:
            payload = torch.load(pt_path, map_location="cpu", weights_only=True)
        except Exception:
            payload = torch.load(pt_path, map_location="cpu", weights_only=False)

        # tensors: [N, C, H, W] float16 → numpy
        tensors = payload["tensors"].numpy()                          # float16
        lats    = payload["center_lat"].numpy().astype(np.float32)   # [N]
        lons    = payload["center_lon"].numpy().astype(np.float32)   # [N]
        doys    = payload["day_of_year"].numpy().astype(np.float32)  # [N]

        data_mmap[offset:offset + n] = tensors
        meta_mmap[offset:offset + n, 0] = lats
        meta_mmap[offset:offset + n, 1] = lons
        meta_mmap[offset:offset + n, 2] = doys

        del payload, tensors, lats, lons, doys

        source_files.append({
            "file":   os.path.basename(pt_path),
            "offset": offset,
            "count":  n,
        })
        offset += n
        print(f"  Written: {os.path.basename(pt_path)} ({n} patches, offset={offset - n})")

    # flush to disk
    data_mmap.flush()
    meta_mmap.flush()
    del data_mmap, meta_mmap

    # ── 写索引文件 ────────────────────────────────────────────────
    index = {
        "total_patches": total_n,
        "patch_shape":   list(patch_shape),   # [C, H, W]
        "dtype_data":    "float16",
        "dtype_meta":    "float32",
        "data_file":     f"{args.prefix}_data.bin",
        "meta_file":     f"{args.prefix}_meta.bin",
        "years":         args.years,
        "months":        args.months,
        "source_files":  source_files,
    }
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nIndex saved to {index_path}")
    print(f"\nDone. Output directory: {args.output_dir}")
    print(f"  {args.prefix}_data.bin  : {data_size_gb:.2f} GB")
    print(f"  {args.prefix}_meta.bin  : {meta_size_mb:.1f} MB")
    print(f"  {args.prefix}_index.json: (index)")
    print(f"\nNext step: move to local SSD before training:")
    print(f"  mv {args.output_dir} /root/autodl-tmp/")


if __name__ == "__main__":
    main()
