from sinkhorn_tests import *
from functools import reduce

snd = lambda x: x[1]


def rand_dyck(n: int, multi: int = 0):
    val = {
        0: n,
        1: 0,
    }
    ret = []
    while True:
        choices = [k for k, v in val.items() if v > 0]
        if not choices:
            break
        choice = choices[randint(0, len(choices))]
        ret.append(choice)
        val[choice] -= 1
        if choice == 0:
            val[1] += 1
    return list(map(lambda x: 2 * multi if x == 0 else 2 * multi + 1, ret))


def rand_dyck_multi(n: int, multi: int):
    dyck = []
    for m in range(multi):
        dyck.append(rand_dyck(n, m))
    counters = {m: 0 for m in range(multi)}
    Dyck = []
    while True:
        choices = [(k, v) for k, v in counters.items() if v < 2 * n]
        if not choices:
            break
        m, v = choices[randint(0, len(choices))]
        counters[m] += 1
        Dyck.append(dyck[m][v])
    return Dyck


def split(dyck: Sequence[int]) -> Tuple[Sequence[int], Sequence[int]]:
    alphas = list(filter(lambda i: dyck[i] == 0, range(len(dyck))))
    betas = list(filter(lambda i: i not in alphas, range(len(dyck))))
    return alphas, betas


def split_multi(dyck: Sequence[int], multiplicity: int) -> Sequence[Tuple[Sequence[int], Sequence[int]]]:
    ret = []
    for m in range(multiplicity):
        alphas = list(filter(lambda i: dyck[i] == 2 * m, range(len(dyck))))
        betas = list(filter(lambda i: dyck[i] == 2 * m + 1, range(len(dyck))))
        ret.append((alphas, betas))
    return ret


def closest_match_policy(dyck: Sequence[int]) -> Tuple[Sequence[int], Sequence[int], Sequence[Tuple[int, int]]]:
    alphas, betas = split(dyck)
    pairs = []
    a_buffer = []
    for i in range(len(dyck)):
        if dyck[i] == 0:
            a_buffer.append(i)
        elif dyck[i] == 1:
            a_index = a_buffer.pop(-1)
            pairs.append((alphas.index(a_index), betas.index(i)))
    return alphas, betas, pairs


def closest_match_policy_multi(dyck: Sequence[int]) -> Tuple[Sequence[int], Sequence[int], Sequence[Tuple[int, int]]]:
    multiplicity = max(dyck) // 2 + 1
    alphabetas = split_multi(dyck, multiplicity)
    alphas, betas = list(zip(*alphabetas))
    Alphas, Betas = reduce(lambda x, y: x+y, alphas), reduce(lambda x, y: x+y, betas)
    Alphas = sorted(Alphas)
    Betas = sorted(Betas)
    pairs = []
    for m, (alphas, betas) in enumerate(alphabetas):
        a_buffer = []
        for i in range(len(dyck)):
            if dyck[i] == 2 * m:
                a_buffer.append(i)  # this points to the dyck-index of the m-alphas seen so far
            elif dyck[i] == 2 * m + 1:
                a_index = a_buffer.pop(-1)  # pop last m-alpha
                pairs.append((Alphas.index(a_index), Betas.index(i)))
    return Alphas, Betas, pairs


def first_match_policy(dyck: Sequence[int]) -> Tuple[Sequence[int], Sequence[int], Sequence[Tuple[int, int]]]:
    alphas, betas = split(dyck)
    pairs = [(i, i) for i, _ in enumerate(alphas)]
    return alphas, betas, pairs


def accu(matches: Tensor, permutors: torch.LongTensor) -> float:
    b, n = matches.shape[0:2]
    return torch.sum(matches.round() * permutors).item() / (b * n)


def make_dn(d: int, m: int = 1) -> nn.Module:
    return DyckNet(d, m).cuda()


def test(dn: nn.Module, B: int, N: int, E: int = 50):
    # loss_fn = nn.KLDivLoss(reduction='batchmean')
    loss_fn = nn.NLLLoss(reduction='mean')
    opt = torch.optim.Adam(dn.parameters())

    for e in range(E):
        n = randint(low=N//2, high=N)
        dycks = [rand_dyck(n) for _ in range(B)]
        alphas, betas, policies = list(zip(*list(map(closest_match_policy, dycks))))
        permutors = list(map(permutor, policies))
        P = torch.stack(permutors).cuda()
        alphas = torch.stack(list(map(torch.tensor, alphas))).cuda()
        betas = torch.stack(list(map(torch.tensor, betas))).cuda()
        dycks = torch.stack(list(map(torch.tensor, dycks))).cuda()

        loss, accu = dn.train_batch(dycks, alphas, betas, P, loss_fn, opt)

        if e%50 == 0:
            print('   Epoch {}'.format(e))
            print('   Loss: {}'.format(loss))
            print('   Accu: {}\n'.format(accu))


def test_multi(dn: nn.Module, B: int, N: int, M: int, E: int = 500):
    # loss_fn = nn.KLDivLoss(reduction='batchmean')
    loss_fn = nn.NLLLoss(reduction='mean')
    opt = torch.optim.Adam(dn.parameters())

    for e in range(E):
        n = randint(low=N // 2, high=N)
        dycks = [rand_dyck_multi(n, M) for _ in range(B)]
        alphas, betas, policies = list(zip(*list(map(closest_match_policy_multi, dycks))))
        permutors = list(map(permutor, policies))
        P = torch.stack(permutors).cuda()
        alphas = torch.stack(list(map(torch.tensor, alphas))).cuda()
        betas = torch.stack(list(map(torch.tensor, betas))).cuda()
        dycks = torch.stack(list(map(torch.tensor, dycks))).cuda()

        loss, accu = dn.train_batch(dycks, alphas, betas, P, loss_fn, opt)

        if e % 50 == 0:
            print('   Epoch {}'.format(e))
            print('   Loss: {}'.format(loss))
            print('   Accu: {}\n'.format(accu))


class DyckNet(nn.Module):
    def __init__(self, d: int, m: int = 1):
        super(DyckNet, self).__init__()
        self.d = d
        self.embedder = nn.Embedding(2 * m, 30)
        self.encoder = nn.LSTM(input_size=30, hidden_size=d, bidirectional=True, num_layers=2, batch_first=True)
        self.fst = nn.Sequential(
            nn.Linear(2*d, d),
            nn.ReLU(),
            nn.Linear(d, d//2),
            nn.ReLU()
        )
        self.snd = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, d//2),
            nn.ReLU()
        )
        self.bi = nn.Linear(d//2, d//2, bias=False)

    def forward(self, sequence: torch.LongTensor, positives: torch.LongTensor, negatives: torch.LongTensor,
                tau: float = 0.01, num_iters: int = 10, noise_factor: float = 0.25, reps: int = 20):
        embedded = self.embedder(sequence)
        encoded, _ = self.encoder(embedded)

        positives = positives.unsqueeze(-1).repeat(1, 1, encoded.shape[-1])
        negatives = negatives.unsqueeze(-1).repeat(1, 1, encoded.shape[-1])

        positive_encs = encoded.gather(dim=1, index=positives)
        negative_encs = encoded.gather(dim=1, index=negatives)

        positive_encs = self.fst(positive_encs)
        negative_encs = self.snd(negative_encs)

        match = torch.bmm(positive_encs, negative_encs.transpose(2, 1))
        return gumbel_sinkhorn(match, tau, num_iters, noise_factor)
        # return sinkhorn(match, 0.01, 50, 1e-08)
        # return averaged_gumbel_sinkhorn(match,
        #                                 0.01, num_iters, noise_factor, reps)

    def train_batch(self, sequence: torch.LongTensor, positives: torch.LongTensor, negatives: torch.LongTensor,
                    permutors: torch.LongTensor, loss_fn: Callable[[Tensor, Tensor], Tensor],
                    opt: torch.optim.Optimizer):
        matches = self.forward(sequence, positives, negatives)


        b, n = matches.shape[0:2]
        correct = permutors.view(b*n, n).max(dim=-1)[1]
        loss = loss_fn(matches.view(b*n, n).log(), correct)
        loss.backward()
        opt.step()
        opt.zero_grad()
        return loss.item(), accu(matches, permutors)




