import torch

def grad_of_quadratic(x_value: float) -> float:
    # TODO: build a tracked leaf for x, compute f(x), run backprop, return df/dx as a float
    x = torch.tensor(x_value, requires_grad=True, dtype=torch.float)
    # x = torch.stack([x**2], [x], [2], dim=0)

    output = x**2 + 3*x + 2
    output.backward()
    
    return x.grad.item()
