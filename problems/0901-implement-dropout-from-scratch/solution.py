import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    # TODO: implement inverted dropout
    n = len(x)
    if training and p != 1:
        mask = torch.empty(n).bernoulli(1-p) #(n,)
        scaling = mask * 1/(1-p)
        x_scaled = torch.where(mask.bool(), scaling * x, 0)
        return x_scaled
    else:
        return x
