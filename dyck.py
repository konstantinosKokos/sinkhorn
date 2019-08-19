from sinkhorn_tests import *

snd = lambda x: x[1]


def rand_dyck(n):
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
    return ret


def split(dyck: Sequence[int]) -> Tuple[Sequence[int], Sequence[int]]:
    alphas = list(filter(lambda i: dyck[i] == 0, range(len(dyck))))
    betas = list(filter(lambda i: i not in alphas, range(len(dyck))))
    return alphas, betas


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


def first_match_policy(dyck: Sequence[int]) -> Tuple[Sequence[int], Sequence[int], Sequence[Tuple[int, int]]]:
    alphas, betas = split(dyck)
    pairs = [(i, i) for i, _ in enumerate(alphas)]
    return alphas, betas, pairs


def accu(matches: Tensor, permutors: torch.LongTensor) -> float:
    b, n = matches.shape[0:2]
    return torch.sum(matches.round() * permutors).item() / (b * n)


def make_dn(d: int) -> nn.Module:
    return DyckNet(d).cuda()


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


class DyckNet(nn.Module):
    def __init__(self, d: int):
        super(DyckNet, self).__init__()
        self.d = d
        self.embedder = nn.Embedding(2, 30)
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




