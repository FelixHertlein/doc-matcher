import torch.nn as nn
import numpy as np
import string
from collections import OrderedDict
from functools import partial
from typing import Callable
import torch
import math
from einops import rearrange

from torchvision.models.vision_transformer import Encoder


class LineEncoder(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        line_length: int,
        line_width: int,
        line_channels: int,
        text_max_length: int,
        text_alphabet_size: int,
        text_hidden_layers: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        representation_size: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        torch._assert(
            line_length % patch_size == 0, "Input shape indivisible by patch size!"
        )
        torch._assert(line_width == patch_size, "Patch size must match line width!")

        self.line_length = line_length
        self.line_width = line_width
        self.line_channels = line_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        self.conv_proj = nn.Conv2d(
            in_channels=line_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        seq_length = line_length // patch_size

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.line_encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.text_encoder = TextEncoder(
            max_length=text_max_length,
            alphabet_size=text_alphabet_size,
            output_dim=hidden_dim,
            num_hidden_layer=text_hidden_layers,
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
        heads_layers["act"] = nn.Tanh()

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = (
                self.conv_proj.in_channels
                * self.conv_proj.kernel_size[0]
                * self.conv_proj.kernel_size[1]
            )
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(
            self.heads.pre_logits, nn.Linear
        ):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(
                self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in)
            )
            nn.init.zeros_(self.heads.pre_logits.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(
            h == self.line_width,
            f"Wrong image height! Expected {self.line_width} but got {h}!",
        )
        torch._assert(
            w == self.line_length,
            f"Wrong image width! Expected {self.line_length} but got {w}!",
        )
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor, text: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # encode line (visual and structural information)
        x = self.line_encoder(x)

        # encode text
        text = self.text_encoder(text)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        # fuse visual and structural information with text
        x = self.feature_fusion(torch.cat([x, text], dim=1))

        x = self.heads(x)

        return x


class TextEncoder(nn.Module):

    def __init__(
        self,
        max_length: int,
        alphabet_size: int,
        output_dim: int,
        num_hidden_layer: int,
    ):
        super().__init__()
        self.max_length = max_length
        self.alphabet_size = alphabet_size
        self.hidden_dim = output_dim

        input_dim = max_length * alphabet_size

        dims = (
            np.linspace(input_dim, output_dim, num_hidden_layer + 2)
            .astype(int)
            .tolist()
        )
        dims[0] = input_dim  # avoid rounding issues
        dims[-1] = output_dim  # avoid roundign issues

        layers = []
        for dim_0, dim_1 in zip(dims, dims[1:]):
            layers.append(nn.Linear(dim_0, dim_1))
            layers.append(nn.ReLU())

        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "n l c -> n (l c)")
        return self.fc(x)


def levit_b_8x256_11_text() -> LineEncoder:
    return LineEncoder(
        line_length=256,
        line_width=8,
        line_channels=11,  # 3 rgb channels + 8 positional encoding channels
        text_max_length=32,
        text_alphabet_size=len(string.printable),
        text_hidden_layers=2,
        patch_size=8,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        representation_size=256,
    )
