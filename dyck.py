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


def test(B: int, N: int, d: int, E: int = 50):
    dn = DyckNet(d)
    loss_fn = nn.KLDivLoss(reduction='sum')
    opt = torch.optim.Adam(dn.parameters())

    for e in range(E):
        dycks = [rand_dyck(N) for _ in range(B)]
        alphas, betas, policies = list(zip(*list(map(closest_match_policy, dycks))))
        permutors = list(map(permutor, policies))
        P = torch.stack(permutors)
        alphas = torch.stack(list(map(torch.tensor, alphas)))
        betas = torch.stack(list(map(torch.tensor, betas)))
        dycks = torch.stack(list(map(torch.tensor, dycks)))

        matches = dn(dycks, alphas, betas)
        loss = loss_fn(matches.log(), P.transpose(2, 1))
        if e > 0:
            loss.backward()
            opt.step()
        opt.zero_grad()

        corr = torch.sum(matches.round() * P.transpose(2, 1)).item()
        accu = corr / (B * N)
        print('   Epoch {}'.format(e))
        print('   Loss: {}'.format(loss.item()))
        print('   Accu: {}\n'.format(accu))
    return loss.item(), accu


class DyckNet(nn.Module):
    def __init__(self, d: int):
        super(DyckNet, self).__init__()
        self.d = d
        self.embedder = nn.Embedding(2, 30)
        self.encoder = nn.LSTM(input_size=30, hidden_size=d, bidirectional=True, num_layers=2)
        self.fst = nn.Sequential(
            nn.Linear(2*d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )
        self.snd = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )
        self.bi = nn.Linear(d, d, bias=False)
        self.bias = nn.Linear(d, 1, bias=False)

    def forward(self, sequence: torch.LongTensor, positives: torch.LongTensor, negatives: torch.LongTensor):
        embedded = self.embedder(sequence)
        encoded, _ = self.encoder(embedded)

        b, n, d = encoded.shape

        # positive_encs = torch.zeros(b, int(n/2), d)
        # negative_encs = torch.zeros(b, int(n/2), d)

        # for i in range(b):
        #     for j in range(int(n/2)):
        #         positive_encs[i, j, :] = encoded[i, positives[i, j], :]
        #         negative_encs[i, j, :] = encoded[i, negatives[i, j], :]


        positives = positives.unsqueeze(-1).repeat(1, 1, encoded.shape[-1])
        negatives = negatives.unsqueeze(-1).repeat(1, 1, encoded.shape[-1])

        positive_encs = encoded.gather(dim=1, index=positives)
        negative_encs = encoded.gather(dim=1, index=negatives)
        positive_encs = self.fst(positive_encs)
        negative_encs = self.snd(negative_encs)

        match = torch.bmm(self.bi(positive_encs), negative_encs.transpose(2, 1)) + self.bias(negative_encs)
        # return match.softmax(dim=-1)
        return sinkhorn(match, 0.1, 20, 1e-06)
#



