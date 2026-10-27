import torch

def layer_norm(x, gamma, beta, eps=1e-5):
    # TODO: normalize over the last dim, then affine-transform with gamma and beta
    #last_dim = x[..., -1]
    # (n, h, d)
    mu_last_dim = torch.mean(x, dim=-1, keepdim=True) # (n, h, None)
    std_last_dim = torch.std(x, dim=-1, keepdim=True, correction=0) # (n, h, None)
    print(torch.mean(x, dim=-1).shape, mu_last_dim.shape)
    return gamma * ((x - mu_last_dim) / torch.sqrt(std_last_dim**2 + eps)) + beta

