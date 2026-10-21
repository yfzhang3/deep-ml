import torch

def to_float_tensor(values):
    # TODO: return a torch.float32 tensor built from `values`
    # create np array from list
    return torch.tensor(values, dtype=torch.float32)
