import torch
import torch.nn as nn


class DownConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=2,
        gn_channels=8,
        dropout=0.1,
        emb_channels=None,
    ):
        super(DownConv, self).__init__()
        assert depth >= 1
        assert in_channels % gn_channels == 0
        assert out_channels % gn_channels == 0

        self.activation = nn.ReLU()
        self.downconv = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
        self.downnorm = nn.GroupNorm(gn_channels, in_channels)
        convs = []
        for _ in range(depth):
            convs.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
            convs.append(nn.Dropout(dropout))
            convs.append(nn.GroupNorm(gn_channels, out_channels))
            convs.append(nn.ReLU())
            in_channels = out_channels
        self.convs = nn.Sequential(*convs)

        if emb_channels is not None:
            self.emb_linear = nn.Linear(emb_channels, out_channels)
        else:
            self.emb_linear = None

    def forward(self, x, emb=None):
        x = self.downconv(x)
        x = self.downnorm(x)
        x = self.activation(x)

        x = self.convs(x)

        if self.emb_linear is not None and emb is not None:
            emb = self.activation(self.emb_linear(emb))
            emb = emb[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])
            x = x + emb

        return x


class UpConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=2,
        gn_channels=8,
        dropout=0.1,
        emb_channels=None,
    ):
        super(UpConv, self).__init__()
        assert depth >= 1
        assert in_channels % gn_channels == 0
        assert out_channels % gn_channels == 0

        self.activation = nn.ReLU()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2, 0)
        self.upnorm = nn.GroupNorm(gn_channels, out_channels)
        convs = []
        in_channels = out_channels * 2
        for _ in range(depth):
            convs.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
            convs.append(nn.Dropout(dropout))
            convs.append(nn.GroupNorm(gn_channels, out_channels))
            convs.append(nn.ReLU())
            in_channels = out_channels
        self.convs = nn.Sequential(*convs)

        if emb_channels is not None:
            self.emb_linear = nn.Linear(emb_channels, out_channels)
        else:
            self.emb_linear = None

    def forward(self, x, short, emb=None):
        x = self.upconv(x)
        x = self.upnorm(x)
        x = self.activation(x)
        x = torch.cat((x, short), dim=1)

        x = self.convs(x)

        if self.emb_linear is not None and emb is not None:
            emb = self.activation(self.emb_linear(emb))
            emb = emb[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])
            x = x + emb

        return x


class MaskUpConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=2,
        gn_channels=8,
        dropout=0.1,
        emb_channels=None,
    ):
        super(MaskUpConv, self).__init__()
        assert depth >= 1
        assert in_channels % gn_channels == 0
        assert out_channels % gn_channels == 0

        self.activation = nn.ReLU()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2, 0)
        self.upnorm = nn.GroupNorm(gn_channels, out_channels)
        convs = []
        for _ in range(depth):
            convs.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
            convs.append(nn.Dropout(dropout))
            convs.append(nn.GroupNorm(gn_channels, out_channels))
            convs.append(nn.ReLU())
        self.convs = nn.Sequential(*convs)

        if emb_channels is not None:
            self.emb_linear = nn.Linear(emb_channels, out_channels)
        else:
            self.emb_linear = None

    def forward(self, x, short, emb=None):
        x = self.upconv(x)
        x = self.upnorm(x)
        x = self.activation(x)
        x = x * short

        x = self.convs(x)

        if self.emb_linear is not None and emb is not None:
            emb = self.activation(self.emb_linear(emb))
            emb = emb[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])
            x = x + emb

        return x
