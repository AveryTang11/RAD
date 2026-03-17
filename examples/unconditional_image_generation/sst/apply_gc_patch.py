#!/usr/bin/env python
"""
apply_gc_patch.py
-----------------
为 unet_2d_local_blocks.py 中的三个 block 类补上：
    self.gradient_checkpointing = False

受影响的类（forward 中已有 checkpoint 分支但 __init__ 缺少属性声明）：
  1. AttnDownBlock2D  —— __init__ 末尾是 self.downsamplers = ... （无 add_downsample 分支，直接赋值）
  2. AttnUpBlock2D    —— __init__ 末尾是 self.upsamplers = None 或 Upsample2D，然后 self.resolution_idx
  3. UNetMidBlock2D   —— __init__ 末尾是 self.resnets = nn.ModuleList(resnets)，无 downsamplers

用法：
    python apply_gc_patch.py \
        ~/autodl-fs/RAD_SST/src/diffusers/models/unets/unet_2d_local_blocks.py
"""

import sys
import shutil
from pathlib import Path


def patch(filepath: str):
    path = Path(filepath)
    assert path.exists(), f"File not found: {path}"

    # 备份原文件
    backup = path.with_suffix(".py.bak")
    shutil.copy2(path, backup)
    print(f"Backup saved to: {backup}")

    content = path.read_text(encoding="utf-8")
    original = content  # 留存对比

    # ---------------------------------------------------------------
    # Patch 1: AttnDownBlock2D
    #   __init__ 末尾特征（唯一）：
    #     self.downsamplers 的赋值后面直接跟着空行 + def forward(
    #   注意：local_blocks 里 AttnDownBlock2D 的 downsamplers 赋值方式与
    #   标准 AttnDownBlock2D 不同——它没有 add_downsample 分支，而是直接：
    #     if downsample_type == "resnet":  ...  else:  ...
    #   通过 forward 签名中的 additional_residuals 来定位（local 版独有）
    # ---------------------------------------------------------------
    OLD_ATN_DOWN = (
        "        self.attentions = nn.ModuleList(attentions)\n"
        "        self.resnets = nn.ModuleList(resnets)\n"
        "\n"
        "    def forward(\n"
        "        self,\n"
        "        hidden_states: torch.FloatTensor,\n"
        "        temb: Optional[torch.FloatTensor] = None,\n"
        "        encoder_hidden_states: Optional[torch.FloatTensor] = None,\n"
        "        attention_mask: Optional[torch.FloatTensor] = None,\n"
        "        cross_attention_kwargs: Optional[Dict[str, Any]] = None,\n"
        "        encoder_attention_mask: Optional[torch.FloatTensor] = None,\n"
        "        additional_residuals: Optional[torch.FloatTensor] = None,\n"
        "    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:"
    )
    NEW_ATN_DOWN = (
        "        self.attentions = nn.ModuleList(attentions)\n"
        "        self.resnets = nn.ModuleList(resnets)\n"
        "\n"
        "        self.gradient_checkpointing = False\n"
        "\n"
        "    def forward(\n"
        "        self,\n"
        "        hidden_states: torch.FloatTensor,\n"
        "        temb: Optional[torch.FloatTensor] = None,\n"
        "        encoder_hidden_states: Optional[torch.FloatTensor] = None,\n"
        "        attention_mask: Optional[torch.FloatTensor] = None,\n"
        "        cross_attention_kwargs: Optional[Dict[str, Any]] = None,\n"
        "        encoder_attention_mask: Optional[torch.FloatTensor] = None,\n"
        "        additional_residuals: Optional[torch.FloatTensor] = None,\n"
        "    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:"
    )

    # ---------------------------------------------------------------
    # Patch 2: AttnUpBlock2D
    #   __init__ 末尾特征（唯一）：
    #     self.resolution_idx = resolution_idx
    #   紧接着 def forward( ... upsample_size ... )
    #   local 版的 AttnUpBlock2D.forward 有 upsample_size 参数（标准版没有）
    # ---------------------------------------------------------------
    OLD_ATN_UP = (
        "        self.resolution_idx = resolution_idx\n"
        "\n"
        "    def forward(\n"
        "        self,\n"
        "        hidden_states: torch.FloatTensor,\n"
        "        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],\n"
        "        temb: Optional[torch.FloatTensor] = None,\n"
        "        upsample_size: Optional[int] = None,\n"
        "        encoder_hidden_states: Optional[torch.FloatTensor] = None,\n"
        "        cross_attention_kwargs: Optional[Dict[str, Any]] = None,\n"
        "        attention_mask: Optional[torch.FloatTensor] = None,\n"
        "        encoder_attention_mask: Optional[torch.FloatTensor] = None,\n"
        "    ) -> torch.FloatTensor:"
    )
    NEW_ATN_UP = (
        "        self.resolution_idx = resolution_idx\n"
        "        self.gradient_checkpointing = False\n"
        "\n"
        "    def forward(\n"
        "        self,\n"
        "        hidden_states: torch.FloatTensor,\n"
        "        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],\n"
        "        temb: Optional[torch.FloatTensor] = None,\n"
        "        upsample_size: Optional[int] = None,\n"
        "        encoder_hidden_states: Optional[torch.FloatTensor] = None,\n"
        "        cross_attention_kwargs: Optional[Dict[str, Any]] = None,\n"
        "        attention_mask: Optional[torch.FloatTensor] = None,\n"
        "        encoder_attention_mask: Optional[torch.FloatTensor] = None,\n"
        "    ) -> torch.FloatTensor:"
    )

    # ---------------------------------------------------------------
    # Patch 3: UNetMidBlock2D
    #   __init__ 末尾特征（唯一）：
    #     self.attentions = nn.ModuleList(attentions)
    #     self.resnets = nn.ModuleList(resnets)
    #   紧接着 def forward( ... ) 且 forward 签名只有 hidden_states + temb
    # ---------------------------------------------------------------
    OLD_MID = (
        "        self.attentions = nn.ModuleList(attentions)\n"
        "        self.resnets = nn.ModuleList(resnets)\n"
        "\n"
        "    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:\n"
        "        hidden_states = self.resnets[0](hidden_states, temb)\n"
        "        for attn, resnet in zip(self.attentions, self.resnets[1:]):\n"
        "            if attn is not None:\n"
        "                hidden_states = attn(hidden_states, temb=temb)\n"
        "            hidden_states = resnet(hidden_states, temb)"
    )
    NEW_MID = (
        "        self.attentions = nn.ModuleList(attentions)\n"
        "        self.resnets = nn.ModuleList(resnets)\n"
        "        self.gradient_checkpointing = False\n"
        "\n"
        "    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:\n"
        "        hidden_states = self.resnets[0](hidden_states, temb)\n"
        "        for attn, resnet in zip(self.attentions, self.resnets[1:]):\n"
        "            if attn is not None:\n"
        "                hidden_states = attn(hidden_states, temb=temb)\n"
        "            hidden_states = resnet(hidden_states, temb)"
    )

    patches = [
        ("AttnDownBlock2D", OLD_ATN_DOWN, NEW_ATN_DOWN),
        ("AttnUpBlock2D",   OLD_ATN_UP,   NEW_ATN_UP),
        ("UNetMidBlock2D",  OLD_MID,      NEW_MID),
    ]

    success = True
    for name, old, new in patches:
        count = content.count(old)
        if count == 0:
            print(f"✗ {name}: pattern NOT found — 请检查文件内容是否与预期一致")
            success = False
        elif count > 1:
            print(f"✗ {name}: pattern found {count} times (expected 1) — 跳过，需手动处理")
            success = False
        else:
            content = content.replace(old, new, 1)
            print(f"✓ {name}: patched")

    if not success:
        print("\n部分 patch 失败，已恢复备份。请手动检查文件。")
        path.write_text(original, encoding="utf-8")
        sys.exit(1)

    path.write_text(content, encoding="utf-8")
    print(f"\n✓ All patches applied to: {path}")
    print("验证命令：")
    print('python -c "')
    print('from patch_unet_for_sst import build_sst_unet')
    print('m = build_sst_unet()')
    print('m.enable_gradient_checkpointing()')
    print('results = {')
    print('    "down[0] DownBlock2D":    m.unet.down_blocks[0].gradient_checkpointing,')
    print('    "down[1] AttnDownBlock2D":m.unet.down_blocks[1].gradient_checkpointing,')
    print('    "down[2] AttnDownBlock2D":m.unet.down_blocks[2].gradient_checkpointing,')
    print('    "down[3] AttnDownBlock2D":m.unet.down_blocks[3].gradient_checkpointing,')
    print('    "mid UNetMidBlock2D":     m.unet.mid_block.gradient_checkpointing,')
    print('    "up[0] AttnUpBlock2D":    m.unet.up_blocks[0].gradient_checkpointing,')
    print('    "up[3] UpBlock2D":        m.unet.up_blocks[3].gradient_checkpointing,')
    print('}')
    print('for k,v in results.items(): print(f"  {k}: {v}")')
    print('print("ALL True:", all(results.values()))')
    print('"')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path/to/unet_2d_local_blocks.py>")
        sys.exit(1)
    patch(sys.argv[1])
