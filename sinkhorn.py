import torch
from torch import nn


def logsumexp(inputs: torch.Tensor, dim: int = 0, keepdim: bool = False) -> torch.Tensor:
    if dim == 0:
        inputs = inputs.view(-1)
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def norm(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x - logsumexp(x, dim=dim, keepdim=True)


def sinkhorn_step(x: torch.Tensor) -> torch.Tensor:
    return norm(norm(x, dim=1), dim=2)


def sinkhorn(x: torch.Tensor, tau: float, iters: int, noise: float) -> torch.Tensor:
    x = x / tau
    for t in range(iters):
        x = sinkhorn_step(x)
    return torch.exp(x) + noise


class Sinkhorn(nn.Module):
    def __init__(self, sinkhorn_iters=5, tau=0.01):
        super(Sinkhorn, self).__init__()
        self.tau = tau
        self.sinkhorn_iters = sinkhorn_iters

    def forward(self, x, eps=1e-6):
        return sinkhorn(x, self.tau, self.sinkhorn_iters, eps)

