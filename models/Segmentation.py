import torch
import torch.nn as nn
import lightning as L

from .modules.UNet import UNet
from .modules.EMA import EMA


class Segmentation(L.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        unet_channels,
        depth=2,
        Masking=False,
        ema_beta=0.99,
        ema_step_start=2000,
        Attention=False,
    ):
        super(Segmentation, self).__init__()

        self.model = UNet(
            in_channels,
            out_channels,
            channels=unet_channels,
            emb_channels=None,
            depth=depth,
            Masking=Masking,
            Attention=Attention,
        )
        self.ema_model = EMA(self.model, ema_beta, ema_step_start)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        _, seg, img = batch
        preds = self.model(img)
        loss = torch.nn.functional.mse_loss(preds, seg)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, seg, img = batch
        preds = self.model(img)
        loss = torch.nn.functional.mse_loss(preds, seg)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def on_before_zero_grad(self, optimizer):
        self.ema_model.update_model(self.model, self.global_step)
