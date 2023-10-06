import torch
from torch.nn.parameter import Parameter
from einops import rearrange


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, offset=0):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        freqs = torch.einsum(
            'i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        from einops import rearrange
        return rearrange(emb, 'n d -> n 1 1 d')


def main():
    # p = Parameter(torch.randn(2, 3, 5))
    # print(p.size())
    # print(p)

    rope = RotaryEmbedding(8)
    emb = rope(16)
    print(emb.size())


def loss_check():
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    y = torch.chunk(x, 2, dim=-1)
    print(y)


def parameter_grad():
    weight = torch.nn.Parameter(torch.randn(3, 5, requires_grad=True))


def grad_set_test():
    w = torch.tensor([1.0], requires_grad=True)
    x = torch.tensor([2.0], requires_grad=True)
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    a.retain_grad()

    y.backward()
    print(w.grad, w.is_leaf)
    print(a.grad, a.is_leaf)


if __name__ == '__main__':
    # main()
    grad_set_test()
