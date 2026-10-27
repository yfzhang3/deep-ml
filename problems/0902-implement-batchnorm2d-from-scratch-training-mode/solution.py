import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    # TODO: training-mode batchnorm2d
    N, C, H, W = x.shape
    x_mu = torch.mean(x, dim=(2, 3), keepdim=True)
    x_std = torch.std(x, dim=(2,3), keepdim=True, correction=0)
    x_norm = (x - x_mu) / torch.sqrt(x_std**2 + eps) # (N, C, H, W)
    gamma_ = torch.reshape(gamma, (1, C, 1, 1))
    beta_ = torch.reshape(beta, (1, C, 1, 1))
    return gamma_ * x_norm + beta_