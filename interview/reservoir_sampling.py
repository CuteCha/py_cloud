# -*- encoding:utf-8 -*-
'''
i<=k \prod_{m=k+1}^{n}(1-(k/m)*(1/k))=k/n
j>k (k/j)\prod_{m=j+1}^{n}(1-(k/m)*(1/k))=k/n
'''

import random


class ReservoirSampling(object):
    def __init__(self, k, n):
        self.k = k
        self.n = n
        self.pool = [i for i in range(k)]

    def sampling(self):
        for j in range(self.k, self.n):
            p = random.randint(0, j - 1)
            if p < self.k:  # probability k/j
                r = random.randint(0, self.k - 1)
                self.pool[r] = j

    def show_result(self):
        print(",".join([str(i) for i in self.pool]))

    def run(self):
        self.sampling()
        self.show_result()


def main():
    ReservoirSampling(8, 1000000).run()


if __name__ == '__main__':
    main()
