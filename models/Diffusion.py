import torch
import torch.nn as nn
import lightning as L

from .modules.UNet import UNet
from .modules.utils import sinusodial
from .modules.EMA import EMA

import random


class Diffusion(L.LightningModule):
    def __init__(
        self,
        timesteps=1000,
        class_rate=0.9,
        Masking=False,
        unet_channels=[32, 64, 128, 256, 512],
        emb_channels=256,
        num_classes=None,
        ema_beta=0.99,
        ema_step_start=2000,
    ):
        super(Diffusion, self).__init__()
        self.t = timesteps
        self.cr = class_rate
        self.emb_channels = emb_channels

        self.model = UNet(3, 3, unet_channels, emb_channels, Masking=Masking)
        self.ema_model = EMA(self.model, ema_beta, ema_step_start)
        self.sched = Scheduler(self.t)

        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes, emb_channels)
        else:
            self.class_emb = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def emb(self, t, labels):
        t_emb = sinusodial(t, self.emb_channels, self.device)
        if self.class_emb is None or labels is None:
            return t_emb

        c_emb = self.class_emb(labels).squeeze()
        return t_emb + c_emb

    def random_t(self, batch_size):
        return torch.randint(1, self.t, size=(batch_size,), device=self.device)

    def training_step(self, batch, batch_idx):
        labels, _, imgs = batch
        t = self.random_t(labels.shape[0])
        noised_img, noise = self.sched(imgs, t)

        if random.random() > self.cr:
            labels = None
        emb = self.emb(t, labels)

        preds = self.model(noised_img, emb)
        loss = torch.nn.functional.mse_loss(preds, noise)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_fit_end(self):
        self.ema_model.update_model(self.model, self.global_step)

    def sample(self, labels, n, cfg_scale=3, model=0):
        """
        n is only used when labels is None
        model is either 0, 1, or 2
        0 means the original model
        1 means the EMA model
        2 means both
        """
        if model == 0 or model == 2:
            MODEL = self.model
        if model == 1:
            MODEL = self.ema_model
        with torch.no_grad():
            if labels is not None:
                n = labels.shape[0]
            x = torch.randn((n, 3, 64, 64)).to(self.device)
            y = x if model == 2 else None

            for i in reversed(range(1, self.t)):
                timesteps = (torch.ones((n)) * i).long().to(self.device)
                emb = self.emb(timesteps, labels)

                preds = MODEL(x, emb)
                if labels is not None and cfg_scale != 1:
                    temb = self.emb(timesteps, None)
                    unlabeled = MODEL(x, temb)
                    preds = unlabeled + (preds - unlabeled) * cfg_scale
                x, noise = self.sched.sample_step(x, preds, timesteps, None)

                if model == 2:
                    preds2 = self.ema_model(y, emb)
                    if labels is not None and cfg_scale != 1:
                        unlabeled = self.ema_model(y, temb)
                        preds2 = unlabeled + (preds2 - unlabeled) * cfg_scale
                    y, _ = self.sched.sample_step(y, preds2, timesteps, noise)

        if model == 2:
            x = torch.cat((x, y), dim=0)
        return x


class Scheduler(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super(Scheduler, self).__init__()
        self.t = timesteps

        self.beta = torch.linspace(beta_start, beta_end, self.t)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.beta = nn.Parameter(self.beta, requires_grad=False)
        self.alpha = nn.Parameter(self.alpha, requires_grad=False)
        self.alpha_hat = nn.Parameter(self.alpha_hat, requires_grad=False)

    def forward(self, img, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        minus_sqrt_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(img)
        return sqrt_alpha_hat * img + minus_sqrt_alpha_hat * noise, noise

    def sample_step(self, x, predicted, t, noise=None):
        """
        noise is for reproducibility
        """
        if type(t) == int:
            t = [t]
        if noise is None:
            if t[0] > 1:
                noise = torch.randn_like(predicted)
            else:
                noise = torch.zeros_like(predicted)

        a = self.alpha[t[0]]
        a_h = self.alpha_hat[t[0]]
        b = self.beta[t[0]]

        return (
            1 / torch.sqrt(a) * (x - ((1 - a) / (torch.sqrt(1 - a_h))) * predicted)
            + torch.sqrt(b) * noise
        ), noise
