import torch
from torch import nn
from torch.distributions.gumbel import Gumbel
from torch import Tensor


def logsumexp(inputs: Tensor, dim: int = 0, keepdim: bool = False) -> Tensor:
    if dim == 0:
        inputs = inputs.view(-1)
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def norm(x: Tensor, dim: int) -> Tensor:
    return x - logsumexp(x, dim=dim, keepdim=True)


def sinkhorn_step(x: Tensor) -> Tensor:
    return norm(norm(x, dim=1), dim=2)


def sinkhorn(x: Tensor, tau: float, iters: int, eps: float = 1e-20) -> Tensor:
    x = x / tau
    for t in range(iters):
        x = sinkhorn_step(x)
    return torch.exp(x) + eps


def gumbel_sinkhorn(x: Tensor, tau: float, iters: int, noise: float, eps: float = 1e-20) -> Tensor:
    gumbel = Gumbel(0, 1).sample(x.shape)
    return sinkhorn(x + gumbel * noise, tau, iters, eps)


def averaged_gumbel_sinkhorn(x: Tensor, tau: float, iters: int, noise: float, reps: int, eps: float = 1e-20) -> Tensor:
    x = x.repeat(reps, 1, 1, 1)
    x = gumbel_sinkhorn(x, tau, iters, noise, eps)
    return x.sum(dim=0) / reps
