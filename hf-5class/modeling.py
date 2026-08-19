import math
from functools import cached_property, lru_cache

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    create_block_mask,
    flex_attention,
)
from torch.nn.utils import parametrize
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.backbone_utils import load_backbone

from .configuration import LSPDetrConfig, STAConfig


flex_attention = torch.compile(flex_attention, dynamic=True)


def init_freqs(head_dim: int, num_heads: int, pos_dim: int, theta: float) -> Tensor:
    """Taken from https://github.com/naver-ai/rope-vit/blob/main/self-attn/rope_self_attn.py."""
    freqs_x = []
    freqs_y = []
    freqs = 1 / (theta ** (torch.arange(0, head_dim, 2 * pos_dim).float() / head_dim))
    for _ in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi
        fx = torch.cat(
            [freqs * torch.cos(angles), freqs * torch.cos(torch.pi / 2 + angles)],
            dim=-1,
        )
        fy = torch.cat(
            [freqs * torch.sin(angles), freqs * torch.sin(torch.pi / 2 + angles)],
            dim=-1,
        )
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    return torch.stack([freqs_x, freqs_y], dim=0)


class Skew(nn.Module):
    """Skew-symmetric matrix parameterization."""

    def forward(self, x: Tensor) -> Tensor:
        a = x.triu(1)
        return a - a.transpose(-1, -2)

    def right_inverse(self, x: Tensor) -> Tensor:
        return x.triu(1)


class CayleySTRING(nn.Module):
    """Implements the Cayley-STRING positional encoding.

    Based on "Learning the RoPEs: Better 2D and 3D Position Encodings with STRING"
    (https://arxiv.org/abs/2502.02562).

    Applies RoPE followed by multiplication with a learnable orthogonal matrix P
    parameterized by the Cayley transform: P = (I - S)(I + S)^-1, where S is
    a learnable skew-symmetric matrix.

    Args:
        dim (int): The feature dimension of the input tensor. Must be even.
        max_seq_len (int): The maximum sequence length.
        base (int): The base value for the RoPE frequency calculation. Defaults to 10000.
        pos_dim (int): The dimensionality of the position vectors (e.g., 1 for 1D, 2 for 2D). Defaults to 1.
    """

    def __init__(
        self, dim: int, num_heads: int, pos_dim: int = 2, theta: float = 100.0
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "Dimension must be divisible by num_heads."

        head_dim = dim // num_heads

        self.freqs = nn.Parameter(init_freqs(head_dim, num_heads, pos_dim, theta))

        self.S = nn.Parameter(torch.zeros(head_dim, head_dim))
        parametrize.register_parametrization(self, "S", Skew())

        self.register_buffer("I", torch.eye(head_dim), persistent=False)

        self.init_weights()

    def init_weights(self) -> None:
        self.S = nn.init.kaiming_uniform_(self.S, a=math.sqrt(5))

    @cached_property
    def P(self) -> Tensor:
        i_plus_s_inv = torch.linalg.inv(self.I + self.S)
        return torch.matmul(self.I - self.S, i_plus_s_inv)

    @parametrize.cached()
    @torch.autocast("cuda", enabled=False)
    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """Apply Cayley-STRING positional encoding.

        Args:
            x ([b, h, n, d]): Input tensor.
            positions ([b, n, pos_dim]): Positions tensor.
        """
        # Compute (I + S)^-1 @ x
        if self.training:
            # Use linalg.solve during training for numerical stability.
            y = torch.linalg.solve(
                self.I + self.S, rearrange(x.float(), "b h n d -> (b h) d n")
            )
            px = torch.matmul(self.I - self.S, y)
            px = rearrange(px, "(b h) d n -> b h n d", b=x.size(0))
        else:
            # During inference, use the pre-calculated matrix P for performance.
            px = x.float() @ self.P.T

        px = px.contiguous()

        # apply RoPE-Mixed
        angles = torch.einsum("bnk,khc->bhnc", positions, self.freqs)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        px_ = torch.view_as_complex(rearrange(px, "... (d two) -> ... d two", two=2))
        out = rearrange(torch.view_as_real(px_ * freqs_cis), "... d two -> ... (d two)")

        return out.type_as(x)


class MLP(nn.Sequential):
    """Very simple multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act_layer: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ) -> None:
        assert num_layers > 1

        layers = []
        h = [hidden_dim] * (num_layers - 1)
        for n, k in zip([input_dim, *h], h, strict=False):
            layers.append(nn.Linear(n, k))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        super().__init__(*layers)


class FeedForward(nn.Module):
    """FeedForward module.

    Taken from https://github.com/meta-llama/llama-models/blob/main/models/llama4/ffn.py
    """

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256) -> None:
        """Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def generate_sta_mask(
    q_canvas_w: int,
    kv_canvas_hw: tuple[int, int],
    kernel: int,
    q_tile: int,
    kv_tile: int,
) -> _mask_mod_signature:
    q_canvas_tile_w = q_canvas_w // q_tile
    kv_canvas_tile_h = kv_canvas_hw[0] // kv_tile
    kv_canvas_tile_w = kv_canvas_hw[1] // kv_tile

    def q_tile_rescale(x: Tensor):
        # Computes round(x * (kv_canvas_tile_w - 1) / (q_canvas_tile_w - 1))
        scale_numerator = kv_canvas_tile_w - 1
        scale_denominator = q_canvas_tile_w - 1
        return (x * scale_numerator + scale_denominator // 2) // scale_denominator

    def get_tile_xy(
        idx: Tensor, tile_size: int, canvas_tile_w: int
    ) -> tuple[Tensor, Tensor]:
        tile_id = idx // (tile_size * tile_size)
        tile_x = tile_id % canvas_tile_w
        tile_y = tile_id // canvas_tile_w
        return tile_x, tile_y

    def sta_mask_2d(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        q_x_tile, q_y_tile = get_tile_xy(q_idx, q_tile, q_canvas_tile_w)
        kv_x_tile, kv_y_tile = get_tile_xy(kv_idx, kv_tile, kv_canvas_tile_w)

        q_x_tile = q_tile_rescale(q_x_tile)
        q_y_tile = q_tile_rescale(q_y_tile)

        center_x = q_x_tile.clamp(kernel // 2, (kv_canvas_tile_w - 1) - kernel // 2)
        center_y = q_y_tile.clamp(kernel // 2, (kv_canvas_tile_h - 1) - kernel // 2)

        # Apply kernel mask in canvas coordinates (not tile coordinates)
        x_mask = torch.abs(center_x - kv_x_tile) <= kernel // 2
        y_mask = torch.abs(center_y - kv_y_tile) <= kernel // 2

        return x_mask & y_mask

    return sta_mask_2d


@lru_cache
def create_sta_block_mask(
    q_len: int,
    kv_len: int,
    q_width: int,
    kv_width: int,
    kernel: int,
    q_tile: int,
    kv_tile: int,
) -> BlockMask:
    return create_block_mask(
        generate_sta_mask(
            q_width, (kv_len // kv_width, kv_width), kernel, q_tile, kv_tile
        ),
        B=None,
        H=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        Q_LEN=q_len,
        KV_LEN=kv_len,
        _compile=True,
    )


@torch.autocast("cuda", enabled=False)
def relative_to_absolute_pos(pos: Tensor, step_x: float, step_y: float) -> Tensor:
    pos = pos.sigmoid()
    h, w = pos.shape[1:3]

    anchor_x = torch.arange(w, dtype=torch.float32, device=pos.device) * step_x
    anchor_y = torch.arange(h, dtype=torch.float32, device=pos.device) * step_y

    absolute_x = pos[..., 0] * step_x + anchor_x
    absolute_y = pos[..., 1] * step_y + anchor_y.unsqueeze(1)
    return torch.stack((absolute_x, absolute_y), dim=-1)


class STAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        src_dim: int,
        num_heads: int,
        kernel: int,
        q_tile: int,
        kv_tile: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.kernel = kernel
        self.q_tile = q_tile
        self.kv_tile = kv_tile

        self.pe = CayleySTRING(dim, num_heads)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(src_dim, dim * 2, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def maybe_pad(self, x: Tensor, tile: int) -> Tensor:
        h, w = x.shape[1:3]
        pad_right = (tile - w % tile) % tile
        pad_bottom = (tile - h % tile) % tile
        return F.pad(x, (0, 0, 0, pad_right, 0, pad_bottom))

    def tile(self, x: Tensor, height: int, tile: int) -> tuple[Tensor, int, int]:
        x = rearrange(x, "b head (h w) dim -> b h w (head dim)", h=height)
        x = self.maybe_pad(x, tile)
        h, w = x.shape[1:3]
        x = rearrange(
            x,
            "b (n_h ts_h) (n_w ts_w) (h d) -> b h (n_h n_w ts_h ts_w) d",
            ts_h=tile,
            ts_w=tile,
            h=self.num_heads,
        )
        return x, h, w

    def forward(
        self, tgt: Tensor, src: Tensor, q_coords: Tensor, k_coords: Tensor
    ) -> Tensor:
        h, w = tgt.shape[1:3]

        q = rearrange(
            self.q(tgt), "b h w (head d) -> b head (h w) d", head=self.num_heads
        )
        k, v = rearrange(
            self.kv(src),
            "b h w (two head d) -> two b head (h w) d",
            two=2,
            head=self.num_heads,
        )

        # RoPE
        q = self.pe(q, q_coords)
        k = self.pe(k, k_coords)

        # tile
        q, q_h, q_w = self.tile(q, h, self.q_tile)
        k, _, kv_w = self.tile(k, src.shape[1], self.kv_tile)
        v, _, _ = self.tile(v, src.shape[1], self.kv_tile)

        # flex attention
        block_mask = create_sta_block_mask(
            q_len=q.shape[2],
            kv_len=k.shape[2],
            q_width=q_w,
            kv_width=kv_w,
            kernel=self.kernel,
            q_tile=self.q_tile,
            kv_tile=self.kv_tile,
        )
        x = flex_attention(q, k, v, block_mask=block_mask)

        # un-tile
        x = rearrange(
            x,
            "b h (n_h n_w ts_h ts_w) d -> b (n_h ts_h) (n_w ts_w) (h d)",
            n_h=q_h // self.q_tile,
            n_w=q_w // self.q_tile,
            ts_h=self.q_tile,
            ts_w=self.q_tile,
        )

        # remove padding
        x = x[:, :h, :w, :].contiguous()

        return self.wo(x)


class Layer(nn.Module):
    def __init__(
        self,
        dim: int,
        src_dim: int,
        num_heads: int,
        self_sta_config: STAConfig,
        cross_sta_config: STAConfig,
    ) -> None:
        super().__init__()

        self.self_attention = STAttention(
            dim,
            dim,
            num_heads,
            kernel=self_sta_config["kernel"],
            q_tile=self_sta_config["q_tile"],
            kv_tile=self_sta_config["kv_tile"],
        )
        self.self_attention_norm = nn.LayerNorm(dim)

        self.cross_attention = STAttention(
            dim,
            src_dim,
            num_heads,
            kernel=cross_sta_config["kernel"],
            q_tile=cross_sta_config["q_tile"],
            kv_tile=cross_sta_config["kv_tile"],
        )
        self.cross_attention_norm = nn.LayerNorm(dim)

        self.ffn = FeedForward(dim, dim * 4)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(
        self, tgt: Tensor, src: Tensor, tgt_coords: Tensor, src_coords: Tensor
    ) -> Tensor:
        x = self.self_attention(tgt, tgt, tgt_coords, tgt_coords)
        tgt = self.self_attention_norm(tgt + x)

        x = self.cross_attention(tgt, src, tgt_coords, src_coords)
        tgt = self.cross_attention_norm(tgt + x)

        return self.ffn_norm(tgt + self.ffn(tgt))


class LSPTransformer(nn.Module):
    def __init__(self, config: LSPDetrConfig, feature_channels: list[int]) -> None:
        super().__init__()

        self.query_block_size = config.query_block_size
        self.num_radial_distances = config.num_radial_distances
        self.feature_levels = config.feature_levels
        self.num_classes = config.num_classes + 1

        self.layers = nn.ModuleList()
        for level in config.feature_levels:
            layer = Layer(
                dim=config.dim,
                src_dim=feature_channels[level],
                num_heads=config.num_heads,
                self_sta_config=config.self_sta_config,
                cross_sta_config=config.cross_sta_config[level],
            )
            self.layers.append(layer)

        # output heads
        self.class_head = nn.Linear(config.dim, self.num_classes)
        self.point_head = nn.ModuleList(
            MLP(config.dim, config.dim, 2, 3) for _ in config.feature_levels
        )
        self.radial_distances_head = nn.ModuleList(
            MLP(config.dim, config.dim, config.num_radial_distances, 3)
            for _ in config.feature_levels
        )

        self.init_weights()

    def init_weights(self) -> None:
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_head.bias, bias_value)

        # initialize regression layers
        for head in self.point_head:
            nn.init.constant_(head[-1].weight, 0)
            nn.init.constant_(head[-1].bias, 0)

        for head in self.radial_distances_head:
            nn.init.constant_(head[-1].weight, 0)
            nn.init.constant_(head[-1].bias, 0)

    def forward(
        self,
        tgt: Tensor,
        ref_points: Tensor,
        features: list[Tensor],
        height: int,
        width: int,
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        src = []
        src_coords = []
        for feature in features:
            b, _, h, w = feature.shape
            coords = torch.zeros(b, h, w, 2, dtype=torch.float32, device=feature.device)
            coords = relative_to_absolute_pos(
                coords, step_x=math.ceil(width / w), step_y=math.ceil(height / h)
            )
            # the outputs from SwinV2 are already normalized
            src.append(rearrange(feature, "b c h w -> b h w c"))
            src_coords.append(rearrange(coords, "b h w pos -> b (h w) pos"))

        radial_distances = torch.full(
            (*tgt.shape[:3], self.num_radial_distances),
            math.log1p(self.query_block_size / 2),
            dtype=torch.float32,
            device=tgt.device,
        )

        logits_list: list[Tensor] = []
        ref_points_list: list[Tensor] = []
        radial_distances_list: list[Tensor] = []

        # for look forward twice
        new_ref_points = ref_points.clone()
        new_radial_distances = radial_distances.clone()

        for i, layer in enumerate(self.layers):
            tgt = layer(
                tgt=tgt,
                src=src[self.feature_levels[i]],
                tgt_coords=relative_to_absolute_pos(
                    ref_points, self.query_block_size, self.query_block_size
                ).flatten(1, 2),
                src_coords=src_coords[self.feature_levels[i]],
            )

            # output heads
            delta_point = self.point_head[i](tgt)
            delta_distances = self.radial_distances_head[i](tgt)
            logits = self.class_head(tgt)

            ref_points_list.append(
                relative_to_absolute_pos(
                    new_ref_points + delta_point,
                    step_x=self.query_block_size / width,
                    step_y=self.query_block_size / height,
                ).flatten(1, 2)
            )
            logits_list.append(logits.flatten(1, 2))
            radial_distances_list.append(
                torch.flatten(new_radial_distances + delta_distances, 1, 2)
            )

            new_ref_points = ref_points + delta_point
            new_radial_distances = radial_distances + delta_distances
            ref_points = new_ref_points.detach()
            radial_distances = new_radial_distances.detach()

        return {
            "logits": logits_list[-1],
            "points": ref_points_list[-1],
            "radial_distances": radial_distances_list[-1],
            "absolute_points": relative_to_absolute_pos(
                ref_points, self.query_block_size, self.query_block_size
            ).flatten(1, 2),
            "embeddings": tgt.flatten(1, 2),
            "aux_outputs": [
                {
                    "logits": a,
                    "points": b,
                    "radial_distances": c,
                }
                for a, b, c in zip(
                    logits_list[:-1],
                    ref_points_list[:-1],
                    radial_distances_list[:-1],
                    strict=True,
                )
            ],
        }


class FeatureSampling(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.reduction = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, points: Tensor, feature: Tensor) -> Tensor:
        x = F.grid_sample(self.reduction(feature), points * 2 - 1, align_corners=False)
        return self.norm(rearrange(x, "b c h w -> b h w c"))


class LSPDetrModel(PreTrainedModel):
    config_class = LSPDetrConfig

    def __init__(self, config: LSPDetrConfig) -> None:
        super().__init__(config)
        self.query_block_size = config.query_block_size

        self.backbone = load_backbone(config)
        _, *feature_channels, neck = self.backbone.num_features

        self.feature_sampling = FeatureSampling(neck, config.dim)
        self.decode_head = LSPTransformer(config, feature_channels)

    def forward(self, pixel_values: Tensor) -> dict[str, Tensor]:
        b, _, h, w = pixel_values.shape

        *features, neck = self.backbone(pixel_values).feature_maps

        ref_points = torch.zeros(
            b,
            math.ceil(h / self.query_block_size),
            math.ceil(w / self.query_block_size),
            2,
            dtype=torch.float32,
            device=neck.device,
        )  # center positions
        tgt = self.feature_sampling(
            relative_to_absolute_pos(
                ref_points, self.query_block_size, self.query_block_size
            ),
            neck,
        )

        return self.decode_head(tgt, ref_points, features, h, w)
