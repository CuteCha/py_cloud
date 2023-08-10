import torch
from torch.nn.parameter import Parameter


def main():
    p = Parameter(torch.randn(2, 3, 5))
    print(p.size())
    print(p)


if __name__ == '__main__':
    main()
