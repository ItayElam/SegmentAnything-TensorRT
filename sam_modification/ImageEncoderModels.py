import torch
import torch.nn as nn

from typing import Optional, Tuple, Type

from segment_anything.modeling.common import LayerNorm2d

from segment_anything.modeling.image_encoder import PatchEmbed, Block
from functools import partial


def build_first_half():
    encoder_embed_dim = 1280
    encoder_depth = 16
    encoder_num_heads = 16
    encoder_global_attn_indexes = (7, 15, 23, 31)
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    first_half = ImageEncoderViTFirstHalf(depth=encoder_depth, embed_dim=encoder_embed_dim, img_size=image_size,
                                          mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                                          num_heads=encoder_num_heads, patch_size=vit_patch_size, qkv_bias=True,
                                          use_rel_pos=True, global_attn_indexes=encoder_global_attn_indexes,
                                          window_size=14, out_chans=prompt_embed_dim)
    return first_half


def build_second_half():
    encoder_embed_dim = 1280
    encoder_depth = 16
    encoder_num_heads = 16
    encoder_global_attn_indexes = (7, 15, 23, 31)
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    second_half = ImageEncoderViTSecondHalf(depth=encoder_depth, embed_dim=encoder_embed_dim, img_size=image_size,
                                            mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                                            num_heads=encoder_num_heads, patch_size=vit_patch_size, qkv_bias=True,
                                            use_rel_pos=True, global_attn_indexes=encoder_global_attn_indexes,
                                            window_size=14, out_chans=prompt_embed_dim)
    return second_half


class ImageEncoderViTFirstHalf(nn.Module):
    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 1024,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, torch.floor_divide(torch.tensor(img_size), patch_size).item(),
                            torch.floor_divide(torch.tensor(img_size), patch_size).item(), embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(torch.floor_divide(torch.tensor(img_size), patch_size).item(),
                            torch.floor_divide(torch.tensor(img_size), patch_size).item()),
            )
            self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        return x


class ImageEncoderViTSecondHalf(nn.Module):
    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(torch.floor_divide(torch.tensor(img_size), patch_size).item(),
                            torch.floor_divide(torch.tensor(img_size), patch_size).item()),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x
