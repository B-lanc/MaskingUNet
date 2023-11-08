import torch.nn as nn

import copy


class EMA(nn.Module):
    def __init__(self, model, beta=0.99, step_start=2000):
        super(EMA, self).__init__()
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.b = beta
        self.step_start = step_start

    def update_model(self, model, step=None):
        if step is not None and step < self.step_start:
            return None
        for ema_param, orig_param in zip(self.model.parameters(), model.parameters()):
            ema_w = ema_param.data
            orig_w = orig_param.data

            ema_param.data = self.b * orig_w + (1 - self.b) * ema_w
