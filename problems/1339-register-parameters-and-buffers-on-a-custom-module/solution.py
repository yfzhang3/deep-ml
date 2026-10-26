import torch
import torch.nn as nn


class ScaledShift(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # TODO: register `scale` (ones of shape (dim,)) as a learnable parameter
        self.scale = nn.Parameter(torch.ones(dim))
        # TODO: register `shift` (zeros of shape (dim,)) as a learnable parameter
        self.shift = nn.Parameter(torch.zeros(dim))
        # TODO: register `calls` as a BUFFER holding a scalar long tensor at 0
        self.register_buffer("calls", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # TODO: if the module is in training mode, increment `calls` by 1
        if self.training:
            self.calls += 1
        # TODO: return x * scale + shift
        return x * self.scale + self.shift
