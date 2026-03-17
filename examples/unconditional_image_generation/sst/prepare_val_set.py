#!/usr/bin/env python3
# prepare_val_set.py
#
# 从验证集 .pt 文件中按月份比例随机抽取固定数量的 patch，
# 保存为单个轻量 .pt 文件供训练时直接加载。
#
# 使用方法：
#   python prepare_val_set.py \
#       --data_root /root/autodl-tmp/pt_cache \
#       --val_years 2023 \
#       --val_months 1 2 3 4 5 6 \
#       --num_samples 1000 \
#       --output_path /root/autodl-tmp/pt_cache/val_fixed.pt \
#       --seed 42

import argparse
import os
import random
import math
from glob import glob

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare fixed validation set")
    parser.add_argument("--data_root",   type=str, required=True,
                        help="目录，包含形如 YYYY_MM.pt 的缓存文件")
    parser.add_argument("--val_years",   type=int, nargs="+", required=True,
                        help="验证集年份，如 2023")
    parser.add_argument("--val_months",  type=int, nargs="+", default=None,
                        help="验证集月份，如 1 2 3 4 5 6；None 表示全年")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="总抽取 patch 数量（按月份文件比例分配）")
    parser.add_argument("--output_path", type=str, default=None,
                        help="输出 .pt 文件路径；默认保存到 data_root/val_fixed_{num_samples}.pt")
    parser.add_argument("--seed",        type=int, default=42,
                        help="随机种子，保证可复现")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.output_path is None:
        args.output_path = os.path.join(
            args.data_root, f"val_fixed_{args.num_samples}.pt"
        )

    # ── 扫描符合条件的 .pt 文件 ─────────────────────────────────
    year_set  = set(str(y) for y in args.val_years)
    month_set = set(args.val_months) if args.val_months is not None else None

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
        # 跳过已经是 val_fixed 输出文件的情况
        if "val_fixed" in basename:
            continue
        pt_files.append(pt_path)

    if len(pt_files) == 0:
        raise FileNotFoundError(
            f"No .pt files found in {args.data_root} for "
            f"years={args.val_years}, months={args.val_months}"
        )

    print(f"Found {len(pt_files)} validation .pt files:")
    for p in pt_files:
        print(f"  {os.path.basename(p)}")

    # ── 按文件 patch 数量比例分配抽取数量 ──────────────────────
    # 先扫描各文件的 patch 数量（只读 shape，不保留数据）
    file_sizes = []
    for pt_path in pt_files:
        try:
            payload = torch.load(pt_path, map_location="cpu", weights_only=True)
        except Exception:
            payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        n = payload["tensors"].shape[0]
        file_sizes.append(n)
        del payload
        print(f"  Scanned: {os.path.basename(pt_path)}  ({n} patches)")

    total_patches = sum(file_sizes)
    print(f"\nTotal patches: {total_patches}")
    print(f"Target samples: {args.num_samples}")

    # 按比例分配，保证总数 == num_samples
    quota = []
    remaining = args.num_samples
    for i, n in enumerate(file_sizes):
        if i == len(file_sizes) - 1:
            quota.append(remaining)  # 最后一个文件补足余量
        else:
            q = round(args.num_samples * n / total_patches)
            q = min(q, n, remaining)
            quota.append(q)
            remaining -= q

    print("\nPer-file quota:")
    for pt_path, q, n in zip(pt_files, quota, file_sizes):
        print(f"  {os.path.basename(pt_path)}: {q} / {n} patches")

    # ── 逐文件加载并随机抽取 ────────────────────────────────────
    all_tensors     = []
    all_center_lat  = []
    all_center_lon  = []
    all_day_of_year = []

    for pt_path, q, n in zip(pt_files, quota, file_sizes):
        if q <= 0:
            continue
        try:
            payload = torch.load(pt_path, map_location="cpu", weights_only=True)
        except Exception:
            payload = torch.load(pt_path, map_location="cpu", weights_only=False)

        indices = torch.tensor(random.sample(range(n), q))
        all_tensors.append(payload["tensors"][indices])          # [q, 4, H, W]
        all_center_lat.append(payload["center_lat"][indices])    # [q]
        all_center_lon.append(payload["center_lon"][indices])    # [q]
        all_day_of_year.append(payload["day_of_year"][indices])  # [q]
        del payload

    # ── 拼接并保存 ───────────────────────────────────────────────
    val_tensors     = torch.cat(all_tensors,     dim=0)  # [num_samples, 4, H, W]
    val_center_lat  = torch.cat(all_center_lat,  dim=0)  # [num_samples]
    val_center_lon  = torch.cat(all_center_lon,  dim=0)  # [num_samples]
    val_day_of_year = torch.cat(all_day_of_year, dim=0)  # [num_samples]

    # 打乱顺序（避免验证时前几个 batch 都是同一个月的数据）
    perm = torch.randperm(len(val_tensors))
    val_payload = {
        "tensors":     val_tensors[perm],
        "center_lat":  val_center_lat[perm],
        "center_lon":  val_center_lon[perm],
        "day_of_year": val_day_of_year[perm],
        # 元信息，方便后续核查
        "meta": {
            "val_years":   args.val_years,
            "val_months":  args.val_months,
            "num_samples": len(val_tensors),
            "seed":        args.seed,
            "source_files": [os.path.basename(p) for p in pt_files],
            "quota":        quota,
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(val_payload, args.output_path)

    size_mb = os.path.getsize(args.output_path) / 1024**2
    print(f"\nSaved {len(val_tensors)} patches → {args.output_path}  ({size_mb:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
