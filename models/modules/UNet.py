import torch
import torch.nn as nn

from .Conv import DownConv, UpConv, MaskUpConv
from .utils import sinusodial
from .Attention import AttentionBlock

import warnings


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels,
        emb_channels=None,
        depth=2,
        Masking=False,
        Attention=False,
    ):
        """
        in channels is the actual channel in, 3 for rgb
        out channels is also the actual channel out
        channels should be a list of something like [32, 64, 128, 256] or similar

        emb_channels=None for no embedding
        """
        super(UNet, self).__init__()
        assert len(channels) >= 3
        self.emb_channels = emb_channels

        UP = MaskUpConv if Masking else UpConv
        ATT = AttentionBlock if Attention else nn.Identity

        pre = nn.ModuleList()
        for _ in range(depth):
            pre.append(nn.Conv2d(in_channels, channels[0], 3, 1, 1))
            pre.append(nn.ReLU())
            in_channels = channels[0]
        pre.append(ATT(channels[0]))
        self.pre = nn.Sequential(*pre)
        self.post = nn.Conv2d(channels[0], out_channels, 3, 1, 1)

        self.down = nn.ModuleList()
        self.downatt = nn.ModuleList()
        self.up = nn.ModuleList()
        self.upatt = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down.append(
                DownConv(
                    channels[i],
                    channels[i + 1],
                    depth=depth,
                    emb_channels=emb_channels,
                )
            )
            self.downatt.append(ATT(channels[i + 1]))
            self.up.append(
                UP(
                    channels[-i - 1],
                    channels[-i - 2],
                    depth=depth,
                    emb_channels=emb_channels,
                )
            )
            self.upatt.append(ATT(channels[-i - 2]))

    def forward(self, x, emb=None):
        if self.emb_channels is None and emb is not None:
            emb = None
            warnings.warn("model does not support embeddings. Using None as the value.")
        shorts = []
        x = self.pre(x)
        shorts.append(x)

        for block, att in zip(self.down, self.downatt):
            x = block(x, emb)
            x = att(x)
            shorts.append(x)
        shorts.pop()
        shorts.reverse()
        for block, short, att in zip(self.up, shorts, self.upatt):
            x = block(x, short, emb)
            x = att(x)

        return self.post(x)
