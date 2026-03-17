#!/usr/bin/env python
# visualize_perlin.py
#
# 绕过 noise_gen.py 中 sig 非 None 时 k 未赋值的 bug：
#   调用时传 sig=None（让函数内部随机），生成原始掩膜后，
#   在脚本里手动用指定 sigma 再做一次高斯模糊 + 二值化，
#   从而安全地模拟三个训练阶段的 Perlin 效果。
#
# 用法：
#   cd ~/autodl-fs/RAD_SST
#   python examples/unconditional_image_generation/sst/visualize_perlin.py

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.transforms import GaussianBlur
from diffusers.utils.noise_gen import generate_perlin_noise_2d

# ===== 设置中文字体 =====
plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Micro Hei',   # 文泉驿微米黑（优先）
    'Noto Sans CJK SC',      # 思源黑体简体（备用）
    'DejaVu Sans'            # 最后回退
]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ── 参数 ──────────────────────────────────────────────────────────────────────
H, W     = 256, 256
SEED     = 42
SAVEPATH = "perlin_effect.png"

# 三个训练阶段: (sig, 缺失率目标, 标签)
STAGES = [
    (15, 0.15, "阶段一 Warm-up\n高频/小碎云  sig=15  rate≈15%"),
    ( 7, 0.50, "阶段二 退火中期\n中频过渡   sig=7   rate≈50%"),
    ( 2, 0.70, "阶段三 稳定训练\n低频/大片云  sig=2   rate≈70%"),
]

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 合成平滑 SST 场 ───────────────────────────────────────────────────────────
def make_fake_sst(H, W):
    y = np.linspace(0, 2 * np.pi, H)
    x = np.linspace(0, 2 * np.pi, W)
    X, Y = np.meshgrid(x, y)
    sst = (  0.4 * np.sin(Y * 1.5 + 0.5)
           + 0.3 * np.cos(X * 1.2 - 0.8)
           + 0.2 * np.sin((X + Y) * 0.7)
           + 0.1 * np.cos(X * 2.5) * np.sin(Y * 2.0) )
    sst = (sst - sst.min()) / (sst.max() - sst.min()) * 2 - 1
    return sst.astype(np.float32)

sst_clean = make_fake_sst(H, W)
sst_t     = torch.from_numpy(sst_clean).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

# ── 方法一核心：先生成原始 Perlin（sig=None），再手动模糊+二值化 ───────────────
def make_mask(sig, target_rate, H, W):
    """
    1. 调用 generate_perlin_noise_2d(sig=None) 得到连续 Perlin 场（已内部随机模糊）
       → 此处不传 sig，规避 UnboundLocalError
    2. 丢弃内部二值化结果，取出二值化前的连续场（通过 rand=False 获得）
    3. 手动用指定 sigma 做高斯模糊，再按 target_rate 百分位二值化
    """
    # rand=False → 函数内部用 triangular 分布阈值，返回 0/1 掩膜
    # 我们需要连续场来重新模糊，所以改用 rand=True 但 sig=None
    # 拿到的已经是 0/1，不够灵活——改为手动重建连续场：

    # 直接调用内部 Perlin 噪声生成逻辑（不做模糊和二值化）
    from diffusers.utils.noise_gen import interpolant
    mask_raw = torch.zeros(1, H, W)
    min_v, max_v = 1, H
    div = (max_v - min_v) * np.random.rand(1) + min_v
    res   = np.array((H / div, W / div))
    res_ceil = np.ceil(res).astype(np.int64)
    delta = (res[0] / H, res[1] / W)

    x = torch.arange(0, res[0][0], delta[0][0])
    y = torch.arange(0, res[1][0], delta[1][0])
    grid = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1) % 1

    angles    = 2 * torch.pi * torch.rand(1, res_ceil[0][0]+1, res_ceil[1][0]+1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    x0 = x.floor().to(torch.int); x1 = x0 + 1
    y0 = y.floor().to(torch.int); y1 = y0 + 1

    tmp = gradients[:, x0]
    g00 = tmp[:, :, y0]; g01 = tmp[:, :, y1]
    tmp = gradients[:, x1]
    g10 = tmp[:, :, y0]; g11 = tmp[:, :, y1]

    n00 = torch.sum(torch.stack((grid[None,:,:,0],   grid[None,:,:,1]  ), dim=-1) * g00, -1)
    n10 = torch.sum(torch.stack((grid[None,:,:,0]-1, grid[None,:,:,1]  ), dim=-1) * g10, -1)
    n01 = torch.sum(torch.stack((grid[None,:,:,0],   grid[None,:,:,1]-1), dim=-1) * g01, -1)
    n11 = torch.sum(torch.stack((grid[None,:,:,0]-1, grid[None,:,:,1]-1), dim=-1) * g11, -1)

    t_interp = interpolant(grid)
    n0 = n00*(1-t_interp[:,:,0]) + t_interp[:,:,0]*n10
    n1 = n01*(1-t_interp[:,:,0]) + t_interp[:,:,0]*n11
    mask_raw[0] = torch.sqrt(torch.tensor([2.])) * ((1-t_interp[:,:,1])*n0 + t_interp[:,:,1]*n1)

    # 手动高斯模糊（sig 控制云团大小）
    if sig > 0:
        k = sig * 4 + 1
        mask_raw = GaussianBlur(kernel_size=k, sigma=(sig, sig))(mask_raw)

    # 按 target_rate 百分位二值化（模拟 MaskCurriculumScheduler 的缺失率控制）
    flat      = mask_raw.view(-1)
    threshold = torch.kthvalue(flat, max(1, int((1 - target_rate) * len(flat))))[0]
    binary    = (mask_raw >= threshold).float()          # [1, H, W]  1=缺失

    return binary[0].numpy(), mask_raw[0].numpy()        # binary, 连续场

# ── 绘图 ──────────────────────────────────────────────────────────────────────
ncols = len(STAGES)
fig   = plt.figure(figsize=(5*ncols + 1, 16))
fig.suptitle("Perlin 掩膜噪声效果可视化（合成 SST 场）", fontsize=14, y=0.99)
gs    = gridspec.GridSpec(4, ncols, figure=fig, hspace=0.38, wspace=0.25)

CMAP_SST  = "RdYlBu_r"
CMAP_CONT = "viridis"
CMAP_MASK = "gray_r"

for col, (sig, rate, title) in enumerate(STAGES):
    mask_bin, mask_cont = make_mask(sig, rate, H, W)
    actual_rate = mask_bin.mean()

    noise    = torch.randn_like(sst_t)
    mask_t   = torch.from_numpy(mask_bin).unsqueeze(0).unsqueeze(0)
    noisy    = sst_t * (1 - mask_t) + noise * mask_t
    noisy_np = noisy[0, 0].numpy()

    # 行 0：原始 SST
    ax0 = fig.add_subplot(gs[0, col])
    im0 = ax0.imshow(sst_clean, cmap=CMAP_SST, vmin=-1, vmax=1)
    ax0.set_title(f"{title}\n\n① 原始 SST（归一化）", fontsize=8.5)
    ax0.axis("off")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # 行 1：连续 Perlin 场（高斯模糊后，二值化之前）
    ax1 = fig.add_subplot(gs[1, col])
    im1 = ax1.imshow(mask_cont, cmap=CMAP_CONT)
    ax1.set_title(f"② Perlin 连续场\n（高斯模糊后，二值化前）", fontsize=8.5)
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 行 2：二值掩膜
    ax2 = fig.add_subplot(gs[2, col])
    ax2.imshow(mask_bin, cmap=CMAP_MASK, vmin=0, vmax=1)
    ax2.set_title(f"③ 二值掩膜（白=缺失）\n实际缺失率 {actual_rate:.1%}", fontsize=8.5)
    ax2.axis("off")

    # 行 3：noisy SST（模型输入）
    ax3 = fig.add_subplot(gs[3, col])
    im3 = ax3.imshow(noisy_np, cmap=CMAP_SST, vmin=-1, vmax=1)
    ax3.set_title("④ noisy SST（模型输入）", fontsize=8.5)
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

plt.savefig(SAVEPATH, dpi=150, bbox_inches="tight")
print(f"[Done] 保存至 {SAVEPATH}")