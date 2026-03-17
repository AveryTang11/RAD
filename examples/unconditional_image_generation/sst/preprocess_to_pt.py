#!/usr/bin/env python
# preprocess_to_pt.py
#
# 将原始 NetCDF 预处理为 float16 的 .pt 缓存文件，解决训练时 IO 瓶颈。
#
# 打包策略：按【年份 × 月份】打包，每个 .pt 文件对应一个月的所有 patch。
#   - 4 年（2020-2023）× 12 个月 = 最多 48 个 .pt 文件（inode 极低）
#   - 每个文件约 1~2 GB，内存压力可控
#   - 文件命名：{YYYY}_{MM:02d}.pt，如 2020_01.pt
#
# 每个 .pt 文件的内容（dict）：
#   "tensors"  : Tensor [N, 4, H, W]  float16
#                  channel 0: sst_norm       归一化 SST [-1, 1]
#                  channel 1: era5_norm      归一化 ERA5 [-1, 1]
#                  channel 2: land_mask      陆地=1, 海洋=0
#                  channel 3: static_valid   QL=5 海洋像素=1
#   "center_lat" : Tensor [N]  float32
#   "center_lon" : Tensor [N]  float32
#   "day_of_year": Tensor [N]  float32
#   "file_names" : list[str]   原始文件名（便于 debug）
#
# 用法：
#   python preprocess_to_pt.py \
#       --data_root /root/autodl-tmp \
#       --years 2020 2021 2022 2023 \
#       --norm_stats_path norm_stats.json \
#       --output_dir /root/autodl-fs/pt_cache \
#       --num_workers 8
#
# 训练时：
#   先将 /root/autodl-fs/pt_cache/ 中的 .pt 文件 rsync 到本地数据盘，
#   再用 SSTDatasetFromCache 从本地盘读取。

import os
import argparse
import json
import math
from glob import glob
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch


# ============================================================
# 单文件处理函数（用于多进程）
# ============================================================

def process_one_file(args_tuple):
    """
    处理单个 L3S 文件，返回处理结果或 None（失败时）。
    独立函数以支持 multiprocessing（lambda 不可 pickle）。
    """
    l3s_path, norm_params = args_tuple

    try:
        import xarray as xr
        from datetime import datetime, timedelta
        import numpy as np

        era5_path = l3s_path.replace("_l3s.nc", "_era5.nc")
        if not os.path.exists(era5_path):
            return None

        t_min  = norm_params["t_min"]
        t_max  = norm_params["t_max"]
        scale  = t_max - t_min

        # ---- 读取 L3S ----
        ds_l3s    = xr.open_dataset(l3s_path, decode_times=False)
        sst_float = ds_l3s["sea_surface_temperature"].values.squeeze().astype(np.float32)
        ql_raw    = ds_l3s["quality_level"].values.squeeze()
        flags     = ds_l3s["l2p_flags"].values.squeeze()
        center_lat = float(ds_l3s.attrs.get("center_lat", 0.0))
        center_lon = float(ds_l3s.attrs.get("center_lon", 0.0))
        time_secs  = int(ds_l3s["time"].values[0])
        ds_l3s.close()

        obs_date    = datetime(1981, 1, 1) + timedelta(seconds=time_secs)
        day_of_year = float(obs_date.timetuple().tm_yday)

        # ---- 读取 ERA5 ----
        ds_era5  = xr.open_dataset(era5_path, decode_times=False)
        era5_sst = ds_era5["sst"].values.squeeze().astype(np.float32)
        ds_era5.close()

        # ---- 构建掩膜 ----
        land_bit     = ((flags & 2) | (flags & 1024)).astype(bool)
        ocean_mask   = (~land_bit).astype(np.float32)
        static_valid = ocean_mask * (ql_raw == 5).astype(np.float32)

        # ---- Min-Max 归一化 ----
        sst_norm  = (2.0 * (sst_float - t_min) / scale - 1.0).astype(np.float32)
        era5_norm = (2.0 * (era5_sst  - t_min) / scale - 1.0).astype(np.float32)

        # ---- 填充 ----
        sst_norm  = np.where(land_bit,            0.0, sst_norm)
        sst_norm  = np.where(np.isnan(sst_norm),  0.0, sst_norm)
        era5_norm = np.where(land_bit,            0.0, era5_norm)
        era5_norm = np.where(np.isnan(era5_norm), 0.0, era5_norm)

        # ---- 拼成 [4, H, W] ----
        tensor_fp32 = np.stack([
            sst_norm,
            era5_norm,
            land_bit.astype(np.float32),
            static_valid,
        ], axis=0)  # [4, H, W]

        return {
            "tensor":      tensor_fp32,          # np.float32 [4, H, W]，存时转 float16
            "center_lat":  center_lat,
            "center_lon":  center_lon,
            "day_of_year": day_of_year,
            "file_name":   os.path.basename(l3s_path),
        }

    except Exception as e:
        print(f"[WARN] Failed: {l3s_path}  reason: {e}")
        return None


# ============================================================
# 主函数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess SST NetCDF → float16 .pt cache")
    parser.add_argument("--data_root",       type=str, required=True,
                        help="原始数据根目录，包含年份子目录")
    parser.add_argument("--years",           type=int, nargs="+", required=True,
                        help="要处理的年份列表，如 2020 2021 2022 2023")
    parser.add_argument("--norm_stats_path", type=str, required=True,
                        help="norm_stats.json 路径（由 compute_stats.py 生成）")
    parser.add_argument("--output_dir",      type=str, required=True,
                        help="输出目录，.pt 文件将保存到此处")
    parser.add_argument("--num_workers",     type=int, default=4,
                        help="并行进程数（建议 4~8，受限于 IO 带宽）")
    parser.add_argument("--overwrite",       action="store_true",
                        help="若目标文件已存在，是否覆盖（默认跳过）")
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载归一化参数
    with open(args.norm_stats_path, "r") as f:
        norm_params = json.load(f)
    print(f"[Config] norm_params: {norm_params}")
    print(f"[Config] years: {args.years}")
    print(f"[Config] output_dir: {args.output_dir}")
    print(f"[Config] num_workers: {args.num_workers}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 扫描所有文件，按 (year, month) 分组 ----
    groups = defaultdict(list)  # key: (year, month) → list of l3s_path
    for year in args.years:
        pattern = os.path.join(args.data_root, str(year), "*", "*_l3s.nc")
        files   = sorted(glob(pattern))
        for f in files:
            basename = os.path.basename(f)
            try:
                month = int(basename[4:6])
            except (ValueError, IndexError):
                print(f"[WARN] Cannot parse month from filename: {basename}, skipping.")
                continue
            groups[(year, month)].append(f)

    total_groups = len(groups)
    print(f"[Scan] Found {sum(len(v) for v in groups.values())} patches across "
          f"{total_groups} (year, month) groups.\n")

    # ---- 逐月处理并保存 ----
    for group_idx, ((year, month), l3s_files) in enumerate(sorted(groups.items()), 1):
        out_filename = f"{year}_{month:02d}.pt"
        out_path     = os.path.join(args.output_dir, out_filename)

        if os.path.exists(out_path) and not args.overwrite:
            print(f"[{group_idx}/{total_groups}] SKIP (exists): {out_filename}")
            continue

        print(f"[{group_idx}/{total_groups}] Processing {year}-{month:02d}  "
              f"({len(l3s_files)} patches) ...", flush=True)

        task_args = [(f, norm_params) for f in l3s_files]

        results = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_one_file, a): a for a in task_args}
            done = 0
            for future in as_completed(futures):
                done += 1
                result = future.result()
                if result is not None:
                    results.append(result)
                # 每 500 个打一次进度
                if done % 500 == 0 or done == len(task_args):
                    print(f"  {done}/{len(task_args)} processed, "
                          f"{len(results)} succeeded", flush=True)

        if len(results) == 0:
            print(f"  [WARN] No valid patches for {year}-{month:02d}, skipping file creation.")
            continue

        # ---- 拼成大 tensor 并转 float16 ----
        tensors_fp16 = torch.from_numpy(
            np.stack([r["tensor"] for r in results], axis=0)  # [N, 4, H, W]
        ).to(torch.float16)

        center_lats  = torch.tensor([r["center_lat"]  for r in results], dtype=torch.float32)
        center_lons  = torch.tensor([r["center_lon"]  for r in results], dtype=torch.float32)
        days_of_year = torch.tensor([r["day_of_year"] for r in results], dtype=torch.float32)
        file_names   = [r["file_name"] for r in results]

        payload = {
            "tensors":     tensors_fp16,   # [N, 4, H, W] float16
            "center_lat":  center_lats,    # [N] float32
            "center_lon":  center_lons,    # [N] float32
            "day_of_year": days_of_year,   # [N] float32
            "file_names":  file_names,     # list[str]
            "norm_params": norm_params,    # dict，备查
        }

        torch.save(payload, out_path)
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  Saved: {out_filename}  "
              f"({len(results)} patches, {size_mb:.1f} MB)\n", flush=True)

    print("=" * 50)
    print("All done.")

    # ---- 打印汇总信息 ----
    pt_files = sorted(glob(os.path.join(args.output_dir, "*.pt")))
    total_size_gb = sum(os.path.getsize(f) for f in pt_files) / 1024**3
    print(f"Total .pt files : {len(pt_files)}")
    print(f"Total size      : {total_size_gb:.2f} GB")
    print(f"Output dir      : {args.output_dir}")


if __name__ == "__main__":
    main()