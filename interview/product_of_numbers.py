class ProductOfNumbers(object):
    def __init__(self):
        self.prefix = []
        self.zero_idx = -1
        self.idx = 0

    def add(self, num):
        if num != 0:
            if self.idx:
                self.prefix.append(self.prefix[-1] * num)
            else:
                self.prefix.append(num)
        else:
            self.zero_idx = self.idx
            self.prefix.append(1)
        self.idx = self.idx + 1

    def get_product(self, k):
        if k > self.idx:
            print(f"ERROR! k={k} lt num of element {self.idx}")
            return None
        elif self.zero_idx >= self.idx - k:
            return 0
        elif k == self.idx:
            return self.prefix[-1]
        else:
            return self.prefix[-1] / self.prefix[-k - 1]


def main():
    pn = ProductOfNumbers()
    arr = [1, 2, 0, 2, 3]
    for i in arr:
        pn.add(i)

    print(pn.get_product(2))
    print(pn.get_product(4))
    print(pn.get_product(6))


if __name__ == '__main__':
    main()
