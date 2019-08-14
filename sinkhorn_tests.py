from sinkhorn import *
from numpy.random import permutation, random, randint
from math import sqrt

from typing import Tuple, Sequence, Callable


Tensor_Map = Callable[[torch.Tensor], torch.Tensor]


def permutor(pairs: Sequence[Tuple[int, int]]) -> torch.Tensor:
    P = torch.zeros(len(pairs), len(pairs))
    for p1, p2 in pairs:
        P[p1, p2] = 1
    return P


def linear_transformation() -> Tensor_Map:
    w = random() * 20
    b = random() * 100
    print('f(x) = {}x + {}'.format(w, b))
    return lambda x: w*x + b


def quadratic_transformation() -> Tensor_Map:
    w_1 = random() * 20
    w_2 = random() * 20
    b = random() * 100
    print('f(x) = {}x2 + {}x + {}'.format(w_1, w_2, b))
    return lambda x: w_1 * x**2 + w_2 * x + b


def test(B: int, N: int, d: int = 1024, E: int = 10, transformation_fn: Callable[[], Tensor_Map] = lambda x: x):
    loss_fn = nn.KLDivLoss()
    trans_fn = transformation_fn()
    pf = PermutationFinder(d=d)
    opt = torch.optim.Adam(pf.parameters())

    for e in range(E):
        b = randint(1, B)
        n = randint(1, N)
        X = torch.rand(b, n, d)
        permutations = [list(zip(*(list(range(n)), permutation(list(range(n)))))) for _ in range(b)]
        permutors = list(map(permutor, permutations))
        P = torch.stack(permutors)
        X_hat = P@(trans_fn(X))
        matches = pf(X, X_hat)
        loss = loss_fn(matches.log(), P.transpose(2, 1))
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), torch.sum(matches.round() - P.transpose(2, 1)).item()


class PermutationFinder(nn.Module):
    def __init__(self, d: int):
        super(PermutationFinder, self).__init__()
        self.d = d
        self.fst = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )
        self.snd = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )

    def forward(self, X, X_hat):
        X = self.fst(X)
        X_hat = self.snd(X_hat)
        match = torch.bmm(X, X_hat.transpose(2, 1)) / sqrt(self.d)
        return sinkhorn(match, 0.1, 3, 1e-06)