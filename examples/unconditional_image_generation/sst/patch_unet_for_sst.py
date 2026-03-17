# patch_unet_for_sst.py
#
# 封装 LocalUNet2DModel，适配 SST 重建任务：
#   - in_channels=3:  [noisy_SST, ERA5, LandMask]
#   - out_channels=1: 仅预测 SST 通道
#   - Geo-Temporal 嵌入通过 class_embed_type="identity" 注入
#
# Geo 嵌入注入原理：
#   LocalUNet2DModel 的 embedding 流程全程使用 4D 张量（Conv2d 1×1）。
#   class_embed_type="identity" 时：
#     class_embedding = nn.Identity，直接将输入加到 time embedding 上，
#     跳过 time_proj（LocalTimesteps sinusoidal 编码），语义正确。
#
#   注入流程：
#     geo_emb  [B, geo_emb_dim=256]
#       → geo_proj  Linear(256 → time_embed_dim)     # time_embed_dim = block_out_channels[0] * 4
#       → [B, time_embed_dim]
#       → unsqueeze(-1).unsqueeze(-1)
#       → [B, time_embed_dim, 1, 1]                  # Conv2d 自动广播到 [B, C, H, W]
#       → nn.Identity（class_embedding）
#       → [B, time_embed_dim, H, W]                  # 与 time embedding 相加
#
#   不修改 unet_2d_local.py 原文件。

import torch
import torch.nn as nn
from diffusers.models.unets.unet_2d_local import LocalUNet2DModel


# 在 patch_unet_for_sst.py 顶部 import 之后添加
from diffusers.models.unets.unet_2d_local_blocks import AttnDownBlock2D, AttnUpBlock2D, UNetMidBlock2D

# 动态补上缺失的属性声明（不修改源文件）
for cls in [AttnDownBlock2D, AttnUpBlock2D, UNetMidBlock2D]:
    original_init = cls.__init__
    def make_patched_init(orig):
        def patched_init(self, *args, **kwargs):
            orig(self, *args, **kwargs)
            if not hasattr(self, 'gradient_checkpointing'):
                self.gradient_checkpointing = False
        return patched_init
    cls.__init__ = make_patched_init(original_init)


def build_sst_unet(
    sample_size: int = 256,
    geo_emb_dim: int = 256,                          # GeoTemporalEmbedding 的 out_dim
    block_out_channels=(128, 256, 256, 256),
    layers_per_block: int = 2,
    num_train_timesteps: int = 2000,
) -> nn.Module:
    """
    构建适用于 SST 重建任务的 LocalUNet2DModel。

    关键参数说明：
      - class_embed_type="identity"：
          class_embedding = nn.Identity，输入直接加到 time embedding 上，
          不经过 time_proj（避免对 geo 向量做无意义的 sinusoidal 编码）。
          要求输入维度 = time_embed_dim = block_out_channels[0] * 4。
      - geo_proj: Linear(geo_emb_dim → time_embed_dim)
          在 SSTUNet.forward 中将 geo_emb 投影到正确维度，
          再 unsqueeze 为 [B, time_embed_dim, 1, 1] 满足 4D 要求。

    Returns:
        SSTUNet: 包含 .unet 和 .geo_proj 两个子模块
    """
    unet = LocalUNet2DModel(
        sample_size=sample_size,
        in_channels=3,
        out_channels=1,
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        time_embedding_type="positional",
        num_train_timesteps=num_train_timesteps,
        # identity: class_embedding = nn.Identity，直接将投影后的 geo 向量加到 emb 上
        # 不走 time_proj（sinusoidal），geo 向量语义不被破坏
        class_embed_type="identity",
        norm_num_groups=32,
        dropout=0.0,
    )

    # time_embed_dim = block_out_channels[0] * 4（与 LocalUNet2DModel 内部一致）
    # geo_proj 将 geo_emb_dim 维的 Fourier 特征投影到 time_embed_dim
    time_embed_dim = block_out_channels[0] * 4
    geo_proj = nn.Linear(geo_emb_dim, time_embed_dim)
    # 较小的初始化：避免 early training 被 geo 条件主导
    nn.init.normal_(geo_proj.weight, std=0.02)
    nn.init.zeros_(geo_proj.bias)

    return SSTUNet(unet, geo_proj)


class SSTUNet(nn.Module):
    """
    封装 LocalUNet2DModel + geo_proj 的轻量包装器。

    forward 中的 geo 注入流程：
      geo_emb [B, geo_emb_dim]
        → geo_proj → [B, time_embed_dim]
        → unsqueeze(-1).unsqueeze(-1) → [B, time_embed_dim, 1, 1]
        → unet(..., class_labels=...)
        → nn.Identity 直接将其加到 time embedding [B, time_embed_dim, H, W] 上
    """

    def __init__(self, unet: LocalUNet2DModel, geo_proj: nn.Linear):
        super().__init__()
        self.unet     = unet
        self.geo_proj = geo_proj

    def forward(self, sample, timestep, geo_emb, return_dict=True):
        """
        Args:
            sample:    [B, 3, H, W]  (noisy_SST, ERA5, LandMask)
            timestep:  [B, 1, H, W]  RAD 逐像素时间步
            geo_emb:   [B, geo_emb_dim]  来自 GeoTemporalEmbedding
            return_dict: bool

        Returns:
            LocalUNet2DOutput 或 tuple，output.sample: [B, 1, H, W]
        """
        # [B, geo_emb_dim] → [B, time_embed_dim] → [B, time_embed_dim, 1, 1]
        # LocalUNet2DModel 中 class_embed_type="identity" 时：
        #   class_embedding = nn.Identity，输入直接加到 emb（4D），1×1 会自动广播
        class_labels = self.geo_proj(geo_emb).unsqueeze(-1).unsqueeze(-1)

        return self.unet(
            sample,
            timestep,
            class_labels=class_labels,
            return_dict=return_dict,
        )

    # 代理常用属性，使 accelerator / EMA 等工具正常工作
    @property
    def config(self):
        return self.unet.config

    @property
    def dtype(self):
        return self.unet.dtype

    def enable_gradient_checkpointing(self):
        self.unet.enable_gradient_checkpointing()

    def enable_xformers_memory_efficient_attention(self):
        self.unet.enable_xformers_memory_efficient_attention()