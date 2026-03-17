#!/usr/bin/env python
# compute_stats.py
#
# 运行一次即可，结果保存为 norm_stats.json。
# 必须在正式训练之前运行，且只能使用训练集年份。
#
# 用法：
#   cd examples/unconditional_image_generation/sst
#   python compute_stats.py \
#       --data_root /path/to/sst_patches \
#       --train_years 2019 2020 2021 2022 \
#       --num_samples 5000 \
#       --save_path norm_stats.json

import argparse
from sst_dataset import compute_norm_stats

def parse_args():
    parser = argparse.ArgumentParser(description="Compute SST Min-Max normalization parameters")
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Root directory of SST data (contains year subdirs)")
    parser.add_argument("--train_years", type=int, nargs="+", required=True,
                        help="Training set years ONLY, e.g. 2019 2020 2021 2022")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of patches to randomly sample. Omit or set to None to use ALL data (recommended)")
    parser.add_argument("--save_path",   type=str, default="norm_stats.json",
                        help="Output path for the JSON normalization parameters")
    return parser.parse_args()

def main():
    args = parse_args()

    print("Computing Min-Max normalization parameters...")
    print(f"  data_root   : {args.data_root}")
    print(f"  train_years : {args.train_years}")
    print(f"  num_samples : {args.num_samples}")
    print(f"  save_path   : {args.save_path}")
    print()

    stats = compute_norm_stats(
        data_root=args.data_root,
        years=args.train_years,
        num_samples=args.num_samples,
        save_path=args.save_path,
    )

    scale = stats["t_max"] - stats["t_min"]
    print("=" * 45)
    print("Result (QL=5 ocean pixels only):")
    print(f"  t_min = {stats['t_min']:.4f} K")
    print(f"  t_max = {stats['t_max']:.4f} K")
    print(f"  range = {scale:.4f} K")
    print()
    print("Normalization formula: x_norm = 2*(x - t_min)/(t_max - t_min) - 1")
    print("ERA5 uses the same t_min/t_max to preserve physical bias (e.g. cool-skin effect).")
    print("=" * 45)

if __name__ == "__main__":
    main()