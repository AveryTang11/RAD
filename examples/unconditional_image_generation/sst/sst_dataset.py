# sst_dataset.py
# SST Dataset, Curriculum Scheduler, and Geo-Temporal Embedding for RAD-based SST reconstruction.

import os
import random
import math
import json
from collections import OrderedDict
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


# ============================================================
# 1. 全局归一化参数工具（Min-Max）
# ============================================================

def compute_norm_stats(data_root: str, years: list, num_samples: int = 5000, save_path: str = None):
    """
    从训练集中随机采样，统计 QL=5 海洋像素的全局 SST 最小值和最大值。
    ERA5 与 L3S 共用同一套归一化参数（保留冷肤效应等物理偏差）。

    归一化公式（输出范围 [-1, 1]）：
        x_norm = 2 * (x - t_min) / (t_max - t_min) - 1

    Args:
        data_root:   数据根目录，结构为 root/YYYY/YYYYMMDD/*.nc
        years:       训练集年份列表，严禁包含验证集/测试集年份
        num_samples: 随机采样的 patch 数量，None 表示使用全量数据
        save_path:   若提供，将归一化参数保存为 JSON 文件

    Returns:
        dict: {"t_min": float, "t_max": float}
    """
    l3s_files = []
    for year in years:
        pattern = os.path.join(data_root, str(year), "*", "*_l3s.nc")
        l3s_files.extend(glob(pattern))
    l3s_files = sorted(l3s_files)

    if len(l3s_files) == 0:
        raise FileNotFoundError(f"No L3S files found under {data_root} for years {years}")

    sampled = l3s_files if num_samples is None else random.sample(l3s_files, min(num_samples, len(l3s_files)))
    print(f"[NormStats] Processing {len(sampled)} / {len(l3s_files)} patches...")

    all_min, all_max = [], []
    for l3s_path in sampled:
        try:
            # xarray 默认 mask_and_scale=True：
            #   _FillValue/valid_min/valid_max → NaN，raw * scale_factor + add_offset → Kelvin
            # decode_times=False：跳过时间解码，避免版本兼容性溢出问题
            ds_l3s = xr.open_dataset(l3s_path, decode_times=False)
            sst    = ds_l3s["sea_surface_temperature"].values.squeeze().astype(np.float32)
            ql     = ds_l3s["quality_level"].values.squeeze()
            ds_l3s.close()

            # 只取 QL=5 的有效像素
            valid_mask = (ql == 5) & np.isfinite(sst)
            if valid_mask.sum() > 10:
                all_min.append(float(sst[valid_mask].min()))
                all_max.append(float(sst[valid_mask].max()))

        except Exception:
            continue

    if len(all_min) == 0:
        raise RuntimeError("No valid QL=5 pixels found. Check data paths and quality flags.")

    stats = {
        "t_min": float(np.min(all_min)),
        "t_max": float(np.max(all_max)),
    }

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[NormStats] Saved to {save_path}: {stats}")

    return stats


def load_norm_stats(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# 2. 课程学习调度器
# ============================================================

class MaskCurriculumScheduler:
    """
    三阶段掩膜课程学习调度器，以全局训练 step 为单位进行控制。

    阶段一（Warm-up, 0 ~ warmup_end）：
        缺失率在 [min_rate, 0.30] 随机采样，Perlin 高频（小碎云）

    阶段二（线性退火, warmup_end ~ anneal_end）：
        缺失率从 0.30 线性增长至 max_rate，Perlin 频率从高频线性过渡到低频

    阶段三（稳定训练, anneal_end ~ total_steps）：
        缺失率在 [0.60, max_rate] 随机采样，Perlin 低频（大片云系）
    """

    def __init__(
        self,
        total_steps: int,
        warmup_ratio: float = 0.15,
        anneal_ratio: float = 0.55,
        min_rate: float = 0.10,
        max_rate: float = 0.75,
        high_freq_sig: float = 15.0,
        low_freq_sig: float = 2.0,
    ):
        self.total_steps   = total_steps
        self.warmup_end    = int(total_steps * warmup_ratio)
        self.anneal_end    = int(total_steps * (warmup_ratio + anneal_ratio))
        self.min_rate      = min_rate
        self.max_rate      = max_rate
        self.high_freq_sig = high_freq_sig
        self.low_freq_sig  = low_freq_sig

    def get_params(self, current_step: int):
        """
        Returns:
            missing_rate (float): 目标缺失率（0~1）
            perlin_sig   (float): 传递给 generate_perlin_noise_2d 的 sig 参数
        """
        if current_step < self.warmup_end:
            missing_rate = np.random.uniform(self.min_rate, 0.30)
            perlin_sig   = self.high_freq_sig

        elif current_step < self.anneal_end:
            progress     = (current_step - self.warmup_end) / max(self.anneal_end - self.warmup_end, 1)
            base_rate    = 0.30 + progress * (self.max_rate - 0.30)
            missing_rate = float(np.clip(base_rate + np.random.uniform(-0.05, 0.05),
                                         self.min_rate, self.max_rate))
            perlin_sig   = self.high_freq_sig + progress * (self.low_freq_sig - self.high_freq_sig)

        else:
            missing_rate = float(np.random.uniform(0.60, self.max_rate))
            perlin_sig   = self.low_freq_sig

        return missing_rate, perlin_sig

    def state_dict(self):
        return {"total_steps": self.total_steps,
                "warmup_end":  self.warmup_end,
                "anneal_end":  self.anneal_end}


# ============================================================
# 3. 地理时空嵌入（Gaussian Fourier Geo-Temporal Embedding）
# ============================================================

class GeoTemporalEmbedding(torch.nn.Module):
    """
    将 4 维时空坐标 [Lat_norm, Lon_norm, cos(DoY), sin(DoY)] 映射到
    高频 Fourier 特征空间，输出 [B, out_dim] 的条件向量，
    用作 LocalUNet2DModel 的 class_labels 输入。

    out_dim = 2 * num_fourier_features（建议 128，则 out_dim=256）。
    """

    def __init__(self, num_fourier_features: int = 128, sigma: float = 1.0):
        super().__init__()
        B_matrix = torch.randn(num_fourier_features, 4) * sigma
        self.register_buffer("B_matrix", B_matrix)
        self.out_dim = 2 * num_fourier_features

    def forward(self, center_lat: torch.Tensor, center_lon: torch.Tensor,
                day_of_year: torch.Tensor) -> torch.Tensor:
        """
        Args:
            center_lat:  [B] patch 中心纬度（degrees）
            center_lon:  [B] patch 中心经度（degrees）
            day_of_year: [B] 一年中的第几天（1~366）

        Returns:
            geo_emb: [B, out_dim]
        """
        lat_norm = center_lat / 90.0
        lon_norm = center_lon / 180.0
        doy_rad  = day_of_year.float() / 365.0 * 2 * math.pi

        coords    = torch.stack([lat_norm, lon_norm,
                                 torch.cos(doy_rad), torch.sin(doy_rad)], dim=-1).float()
        projected = coords @ self.B_matrix.T * 2 * math.pi
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


# ============================================================
# 4. SST Dataset
# ============================================================

class SSTDataset(Dataset):
    """
    读取 SST 数据，支持两种模式，通过 data_root 内容自动判断：

    模式一（NetCDF 模式）：data_root 下存在年份子目录和 .nc 文件，直接读取原始数据。
        适用于 smoke test、调试，或尚未完成预处理时。

    模式二（缓存模式）：data_root 下存在形如 {YYYY}_{MM:02d}.pt 的缓存文件，
        从 float16 .pt 文件读取，IO 速度大幅提升。
        缓存文件由 preprocess_to_pt.py 生成。

    自动判断逻辑：
        构造时扫描 data_root，若发现 .pt 文件则进入缓存模式，否则进入 NetCDF 模式。
        也可通过 force_mode="netcdf" 或 force_mode="cache" 强制指定。

    两种模式返回完全相同的字典结构：
        sst          [1, H, W]  归一化后的 L3S SST
        era5         [1, H, W]  归一化后的 ERA5 SST
        land_mask    [1, H, W]  陆地=1, 海洋=0
        static_valid [1, H, W]  QL=5 海洋像素=1（用于 loss mask）
        center_lat   scalar tensor float32
        center_lon   scalar tensor float32
        day_of_year  scalar tensor float32
    """

    def __init__(
        self,
        data_root:   str,
        years:       list,
        norm_params: dict,
        months:      list = None,
        force_mode:  str  = None,   # None=自动, "netcdf", "cache"
    ):
        """
        Args:
            data_root:   数据根目录（NetCDF 模式）或 .pt 缓存目录（缓存模式）
            years:       使用的年份列表
            norm_params: 归一化参数字典 {"t_min": float, "t_max": float}
                         缓存模式下仍需传入（接口一致），但实际归一化在预处理时已完成
            months:      可选，月份过滤列表，如 [1,2,3,4,5,6]；None 表示全年
            force_mode:  强制指定模式，None 时自动检测
        """
        self.data_root   = data_root
        self.norm_params = norm_params
        self._scale      = norm_params["t_max"] - norm_params["t_min"]
        self.months      = set(months) if months is not None else None

        # ---- 自动检测模式 ----
        if force_mode is not None:
            self._mode = force_mode
        else:
            pt_files = glob(os.path.join(data_root, "*.pt"))
            self._mode = "cache" if len(pt_files) > 0 else "netcdf"

        month_info = f", months={sorted(self.months)}" if self.months is not None else ""

        if self._mode == "cache":
            self._init_cache(years)
            print(f"[SSTDataset] cache mode | {len(self)} patches | "
                  f"years={years}{month_info}")
        else:
            self.l3s_files = self._scan_netcdf_files(years)
            print(f"[SSTDataset] netcdf mode | {len(self.l3s_files)} patches | "
                  f"years={years}{month_info}")

    # ----------------------------------------------------------
    # 缓存模式初始化
    # ----------------------------------------------------------

    def _init_cache(self, years: list, lru_size: int = 34):
        """
        扫描符合条件的 .pt 文件，记录路径和索引，按需加载并用 LRU cache 缓存。

        lru_size=34：每个文件约 2GB，34 个文件约 68GB，加上模型/系统开销共约 76GB，
        在 90GB 内存下保留约 14GB 安全余量，同时避免了 torch.cat 全量拼接时
        的双倍内存峰值问题。首轮 epoch 结束后 34 个文件常驻内存，之后几乎无磁盘 IO。

        线程安全：磁盘 IO 在锁外执行，不阻塞其他 worker。
        """
        import threading
        self._lru_lock = threading.Lock()

        year_set = set(str(y) for y in years)
        pt_files = []
        for pt_path in sorted(glob(os.path.join(self.data_root, "*.pt"))):
            basename = os.path.basename(pt_path)          # e.g. 2020_01.pt
            parts    = basename.replace(".pt", "").split("_")
            if len(parts) != 2:
                continue
            year_str, month_str = parts
            if year_str not in year_set:
                continue
            if self.months is not None:
                try:
                    if int(month_str) not in self.months:
                        continue
                except ValueError:
                    continue
            pt_files.append(pt_path)

        if len(pt_files) == 0:
            raise FileNotFoundError(
                f"No .pt cache files found in {self.data_root} for years {years}"
            )

        self._cache_paths = []   # list of pt_path
        self._cache_index = []   # list of (data_idx, local_idx)
        self._lru_size    = lru_size
        self._lru_cache   = OrderedDict()  # data_idx -> payload（LRU）

        for pt_path in pt_files:
            try:
                payload = torch.load(pt_path, map_location="cpu", weights_only=True)
            except Exception:
                payload = torch.load(pt_path, map_location="cpu", weights_only=False)
            data_idx = len(self._cache_paths)
            n        = payload["tensors"].shape[0]
            del payload
            self._cache_paths.append(pt_path)
            for i in range(n):
                self._cache_index.append((data_idx, i))
            print(f"  Scanned: {os.path.basename(pt_path)}  ({n} patches)")

        print(f"  [Cache] lru_size={lru_size} (~{lru_size * 2:.0f} GB), "
              f"total files={len(self._cache_paths)}, total patches={len(self._cache_index)}")

    # ----------------------------------------------------------
    # NetCDF 模式：文件扫描
    # ----------------------------------------------------------

    def _scan_netcdf_files(self, years: list) -> list:
        all_files = []
        for year in years:
            pattern = os.path.join(self.data_root, str(year), "*", "*_l3s.nc")
            for f in sorted(glob(pattern)):
                if not os.path.exists(f.replace("_l3s.nc", "_era5.nc")):
                    continue
                if self.months is not None:
                    basename = os.path.basename(f)
                    try:
                        month = int(basename[4:6])
                    except (ValueError, IndexError):
                        continue
                    if month not in self.months:
                        continue
                all_files.append(f)
        return all_files

    # ----------------------------------------------------------
    # 通用接口
    # ----------------------------------------------------------

    def __len__(self):
        if self._mode == "cache":
            return len(self._cache_index)
        return len(self.l3s_files)

    def __getitem__(self, idx: int) -> dict:
        if self._mode == "cache":
            return self._getitem_cache(idx)
        return self._getitem_netcdf(idx)

    # ----------------------------------------------------------
    # 缓存模式：读取
    # ----------------------------------------------------------

    def _getitem_cache(self, idx: int) -> dict:
        data_idx, local_idx = self._cache_index[idx]

        # 先在锁内查找 LRU cache
        with self._lru_lock:
            if data_idx in self._lru_cache:
                self._lru_cache.move_to_end(data_idx)
                payload = self._lru_cache[data_idx]
            else:
                payload = None

        # 未命中时在锁外执行磁盘 IO，避免长时间持锁阻塞其他 worker
        if payload is None:
            try:
                new_payload = torch.load(
                    self._cache_paths[data_idx], map_location="cpu", weights_only=True
                )
            except Exception:
                new_payload = torch.load(
                    self._cache_paths[data_idx], map_location="cpu", weights_only=False
                )
            with self._lru_lock:
                # 二次检查，防止其他 worker 已加载同一文件
                if data_idx not in self._lru_cache:
                    self._lru_cache[data_idx] = new_payload
                    if len(self._lru_cache) > self._lru_size:
                        self._lru_cache.popitem(last=False)
                self._lru_cache.move_to_end(data_idx)
                payload = self._lru_cache[data_idx]

        # float16 → float32（训练时由 accelerator 按需转 bf16）
        tensor = payload["tensors"][local_idx].to(torch.float32)  # [4, H, W]

        return {
            "sst":          tensor[0:1],                               # [1, H, W]
            "era5":         tensor[1:2],                               # [1, H, W]
            "land_mask":    tensor[2:3],                               # [1, H, W]
            "static_valid": tensor[3:4],                               # [1, H, W]
            "center_lat":   payload["center_lat"][local_idx],          # scalar float32
            "center_lon":   payload["center_lon"][local_idx],          # scalar float32
            "day_of_year":  payload["day_of_year"][local_idx],         # scalar float32
        }

    # ----------------------------------------------------------
    # NetCDF 模式：读取（保持原有逻辑不变）
    # ----------------------------------------------------------

    def _minmax_norm(self, x: np.ndarray) -> np.ndarray:
        """Min-Max 归一化到 [-1, 1]"""
        return (2.0 * (x - self.norm_params["t_min"]) / self._scale - 1.0).astype(np.float32)

    def _getitem_netcdf(self, idx: int) -> dict:
        l3s_path  = self.l3s_files[idx]
        era5_path = l3s_path.replace("_l3s.nc", "_era5.nc")

        # ---- 读取 L3S ----
        # decode_times=False：跳过时间自动解码，手动解析原始整数秒，
        # 避免 xarray/pandas 版本兼容性导致的溢出问题。
        # mask_and_scale 仍默认 True：_FillValue → NaN，scale_factor/add_offset 自动应用。
        ds_l3s    = xr.open_dataset(l3s_path, decode_times=False)
        sst_float = ds_l3s["sea_surface_temperature"].values.squeeze().astype(np.float32)  # [H, W]
        ql_raw    = ds_l3s["quality_level"].values.squeeze()                               # [H, W]
        flags     = ds_l3s["l2p_flags"].values.squeeze()                                   # [H, W]
        center_lat = float(ds_l3s.attrs.get("center_lat", 0.0))
        center_lon = float(ds_l3s.attrs.get("center_lon", 0.0))

        # 解析时间 → day_of_year
        # decode_times=False 后 time 保持原始 int32（seconds since 1981-01-01）
        time_secs   = int(ds_l3s["time"].values[0])
        obs_date    = datetime(1981, 1, 1) + timedelta(seconds=time_secs)
        day_of_year = obs_date.timetuple().tm_yday  # 1~366
        ds_l3s.close()

        # ---- 读取 ERA5 ----
        # decode_times=False：ERA5 time 单位 'seconds since 1970-01-01' 在部分
        # xarray/pandas 版本下会触发溢出，且此处完全不需要时间信息
        ds_era5  = xr.open_dataset(era5_path, decode_times=False)
        era5_sst = ds_era5["sst"].values.squeeze().astype(np.float32)  # [H, W]
        ds_era5.close()

        # ---- 构建掩膜 ----
        # l2p_flags 中有两个陆地标志位：
        #   mask=2    (bit2):  第一个 land bit
        #   mask=1024 (bit11): 第二个 land bit（0=ocean; 1=land）
        land_bit   = ((flags & 2) | (flags & 1024)).astype(bool)  # True=land
        ocean_mask = (~land_bit).astype(np.float32)                # 1=ocean, 0=land

        # static_valid：QL=5 的海洋像素，是唯一有可信 GT 的区域，用于 loss mask
        # ql_raw 经 xarray 解码为 float，NaN != 5，不会误判
        static_valid = ocean_mask * (ql_raw == 5).astype(np.float32)  # [H, W]

        # ---- Min-Max 归一化（先对整个场归一化，包括陆地）----
        sst_norm  = self._minmax_norm(sst_float)
        era5_norm = self._minmax_norm(era5_sst)

        # ---- 归一化后填充 ----
        # SST 通道：
        #   - 陆地像素 → 0（中性值，屏蔽陆地信号）
        #   - NaN（真实缺失，QL 通常也 <5）→ 0（消除 NaN，给模型安全占位值）
        #   - QL<5 非 NaN → 保留归一化值（模型需学习重建，不应填假值）
        sst_norm = np.where(land_bit,              0.0, sst_norm)
        sst_norm = np.where(np.isnan(sst_norm),    0.0, sst_norm)

        # ERA5 通道：
        #   - 陆地 → 0（ERA5 在陆地有定义值但不属于预测域，屏蔽掉）
        #   - NaN → 0（保险处理）
        era5_norm = np.where(land_bit,             0.0, era5_norm)
        era5_norm = np.where(np.isnan(era5_norm),  0.0, era5_norm)

        # ---- 转为张量 ----
        return {
            "sst":          torch.from_numpy(sst_norm[None]),                        # [1, H, W]
            "era5":         torch.from_numpy(era5_norm[None]),                       # [1, H, W]
            "land_mask":    torch.from_numpy(land_bit.astype(np.float32)[None]),     # [1, H, W]
            "static_valid": torch.from_numpy(static_valid[None]),                    # [1, H, W]
            "center_lat":   torch.tensor(center_lat,       dtype=torch.float32),
            "center_lon":   torch.tensor(center_lon,       dtype=torch.float32),
            "day_of_year":  torch.tensor(float(day_of_year), dtype=torch.float32),
        }

# ============================================================
# 5. 固定验证集 Dataset（读取 prepare_val_set.py 生成的文件）
# ============================================================

class FixedValDataset(Dataset):
    """
    读取由 prepare_val_set.py 预先生成的固定验证集 .pt 文件。
    数据在构造时一次性全量加载到内存（约 500MB），之后 __getitem__
    直接切片，无任何磁盘 IO，不占用训练集的 LRU cache 内存空间。

    与 SSTDataset 返回完全相同的字典结构，可直接替换用于 DataLoader。
    """

    def __init__(self, val_pt_path: str):
        """
        Args:
            val_pt_path: prepare_val_set.py 生成的 .pt 文件路径
        """
        if not os.path.exists(val_pt_path):
            raise FileNotFoundError(
                f"Fixed val set not found: {val_pt_path}\n"
                f"Please run prepare_val_set.py first."
            )
        try:
            payload = torch.load(val_pt_path, map_location="cpu", weights_only=True)
        except Exception:
            payload = torch.load(val_pt_path, map_location="cpu", weights_only=False)

        # 全量加载到内存，float16 保持原样，__getitem__ 时转 float32
        self._tensors     = payload["tensors"]      # [N, 4, H, W] float16
        self._center_lat  = payload["center_lat"]   # [N]
        self._center_lon  = payload["center_lon"]   # [N]
        self._day_of_year = payload["day_of_year"]  # [N]

        meta = payload.get("meta", {})
        size_mb = os.path.getsize(val_pt_path) / 1024**2
        print(f"[FixedValDataset] Loaded {len(self._tensors)} patches "
              f"from {os.path.basename(val_pt_path)} ({size_mb:.1f} MB)")
        if meta:
            print(f"  years={meta.get('val_years')}, months={meta.get('val_months')}, "
                  f"seed={meta.get('seed')}, sources={meta.get('source_files')}")

    def __len__(self):
        return len(self._tensors)

    def __getitem__(self, idx: int) -> dict:
        tensor = self._tensors[idx].to(torch.float32)  # [4, H, W]
        return {
            "sst":          tensor[0:1],                 # [1, H, W]
            "era5":         tensor[1:2],                 # [1, H, W]
            "land_mask":    tensor[2:3],                 # [1, H, W]
            "static_valid": tensor[3:4],                 # [1, H, W]
            "center_lat":   self._center_lat[idx],       # scalar float32
            "center_lon":   self._center_lon[idx],       # scalar float32
            "day_of_year":  self._day_of_year[idx],      # scalar float32
        }


# ============================================================
# 6. mmap Dataset（读取 build_mmap_dataset.py 生成的文件）
# ============================================================

class MmapSSTDataset(Dataset):
    """
    基于 numpy.memmap 的 SST 训练集，内存占用极低。

    原理：
        memmap 不把文件读入内存，而是将文件映射到虚拟地址空间。
        操作系统只在实际访问某个 patch 时才把对应的页（4KB）从 SSD
        读入物理内存，访问完后可随时换出。因此无论数据集多大，
        常驻物理内存只有当前正在访问的少量页面（通常 < 1GB）。

        配合本地 SSD（随机读 > 500MB/s），随机访问延迟极低，
        不会成为训练瓶颈。

    与 SSTDataset / FixedValDataset 返回完全相同的字典结构。
    """

    def __init__(self, index_path: str):
        """
        Args:
            index_path: build_mmap_dataset.py 生成的 *_index.json 路径
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"mmap index not found: {index_path}\n"
                f"Please run build_mmap_dataset.py first."
            )

        with open(index_path, "r") as f:
            self._index = json.load(f)

        index_dir = os.path.dirname(os.path.abspath(index_path))
        data_path = os.path.join(index_dir, self._index["data_file"])
        meta_path = os.path.join(index_dir, self._index["meta_file"])

        for p in [data_path, meta_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"mmap data file not found: {p}")

        total_n     = self._index["total_patches"]
        C, H, W     = self._index["patch_shape"]

        # mode="r"：只读映射，不占用物理内存，OS 按需换入/换出
        self._data = np.memmap(data_path, dtype=np.float16, mode="r",
                               shape=(total_n, C, H, W))
        self._meta = np.memmap(meta_path, dtype=np.float32, mode="r",
                               shape=(total_n, 3))

        data_gb = total_n * C * H * W * 2 / 1024**3
        print(f"[MmapSSTDataset] {total_n} patches, shape=[{C},{H},{W}], "
              f"mapped {data_gb:.2f} GB (physical RAM usage near zero)")

    def __len__(self):
        return self._index["total_patches"]

    def __getitem__(self, idx: int) -> dict:
        # 从 mmap 读取单个 patch，OS 只加载对应的磁盘页。
        # np.array(...) 显式触发一次从 mmap 到普通内存的拷贝（float16），
        # 再 astype(float32) 转精度，最后 torch.from_numpy 零拷贝建 tensor。
        # 这比直接 self._data[idx].astype(float32) 少一次内存分配，
        # 因为 astype 在 mmap 上会产生两次拷贝（先读 mmap 页，再转类型）。
        raw    = np.array(self._data[idx], dtype=np.float16)   # mmap → RAM，一次拷贝
        tensor = torch.from_numpy(raw.astype(np.float32))      # float16 → float32

        meta   = np.array(self._meta[idx], dtype=np.float32)   # [3]

        return {
            "sst":          tensor[0:1],                                       # [1, H, W]
            "era5":         tensor[1:2],                                       # [1, H, W]
            "land_mask":    tensor[2:3],                                       # [1, H, W]
            "static_valid": tensor[3:4],                                       # [1, H, W]
            "center_lat":   torch.tensor(meta[0], dtype=torch.float32),        # scalar
            "center_lon":   torch.tensor(meta[1], dtype=torch.float32),        # scalar
            "day_of_year":  torch.tensor(meta[2], dtype=torch.float32),        # scalar
        }