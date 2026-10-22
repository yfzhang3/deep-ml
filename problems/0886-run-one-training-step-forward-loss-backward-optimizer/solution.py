import torch
import torch.nn as nn
import torch.nn.functional as F

def train_one_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    # TODO: build an SGD optimizer, run one full forward/loss/backward/step cycle,
    # and return the pre-update loss as a Python float.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # 1. zero out grad
    optimizer.zero_grad()
    # 2. make pred
    pred = model(x)
    # 3. calc loss
    loss = F.mse_loss(pred, y)
    # 4. calc gradients
    loss.backward()
    # 5. take a step
    optimizer.step()
    
    return loss.item()
    