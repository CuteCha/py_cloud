# -*- encoding:utf-8 -*-
def bag_problem_solution():
    c = [0, 2, 3, 4, 5]
    v = [0, 3, 4, 5, 6]
    capacity = 8
    num = 4
    dp = [[0 for _ in range(capacity + 1)] for _ in range(num + 1)]
    item = [0 for _ in range(num + 1)]

    def cal_max_value():
        for i in range(5):
            for j in range(capacity + 1):
                if j < c[i]:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - c[i]] + v[i])

    def track_item(i, j):
        if i >= 0:
            if dp[i][j] == dp[i - 1][j]:
                item[i] = 0
                track_item(i - 1, j)
            elif j - c[i] >= 0 and dp[i][j] == dp[i - 1][j - c[i]] + v[i]:
                item[i] = 1
                track_item(i - 1, j - c[i])

    def show_result():
        for each in dp:
            print(each)
        print("-" * 36)
        print(item)

    cal_max_value()
    track_item(num, capacity)
    show_result()


class BagProblemSolution(object):
    def __init__(self, caps, vals, capacity):
        self.c = [0] + caps
        self.v = [0] + vals
        self.capacity = capacity
        self.num = len(caps)
        self.dp = [[0 for _ in range(self.capacity + 1)] for _ in range(self.num + 1)]
        self.item = [0 for _ in range(self.num + 1)]

    def cal_max_value(self):
        for i in range(1, self.num + 1):
            for j in range(self.capacity + 1):
                if j < self.c[i]:
                    self.dp[i][j] = self.dp[i - 1][j]
                else:
                    self.dp[i][j] = max(self.dp[i - 1][j], self.dp[i - 1][j - self.c[i]] + self.v[i])

    def track_item(self, i, j):
        if i <= 0 or j <= 0:
            return

        if self.dp[i][j] == self.dp[i - 1][j]:
            self.item[i] = 0
            self.track_item(i - 1, j)
        elif j - self.c[i] >= 0 and self.dp[i][j] == self.dp[i - 1][j - self.c[i]] + self.v[i]:
            self.item[i] = 1
            self.track_item(i - 1, j - self.c[i])

    def show_result(self):
        for each in self.dp:
            print(each)

        print("=" * 36)
        print(f"max value: {self.dp[-1][-1]}")
        print("-" * 36)
        print(f"item: {self.item}")

    def run(self):
        self.cal_max_value()
        self.track_item(self.num, self.capacity)
        self.show_result()


class BagProblemSolution2(object):
    def __init__(self, caps, vals, capacity):
        self.c = caps
        self.v = vals
        self.capacity = capacity
        self.num = len(caps)
        self.dp = [[0 for _ in range(self.capacity + 1)] for _ in range(self.num)]
        self.rec = [[0 for _ in range(self.capacity + 1)] for _ in range(self.num)]
        self.item = [0 for _ in range(self.num)]

    def cal_max_value(self):
        for j in range(self.capacity + 1):
            if self.v[0] < j:
                self.dp[0][j] = self.v[0]
                self.rec[0][j] = 1

        for i in range(1, self.num):
            for j in range(self.capacity + 1):
                if j >= self.c[i] and self.dp[i - 1][j - self.c[i]] + self.v[i] > self.dp[i - 1][j]:
                    self.dp[i][j] = self.dp[i - 1][j - self.c[i]] + self.v[i]
                    self.rec[i][j] = 1
                else:
                    self.dp[i][j] = self.dp[i - 1][j]

    def track_item(self):
        t = self.capacity
        for i in range(self.num - 1, -1, -1):
            if self.rec[i][t] == 1:
                self.item[i] = 1
                t -= self.c[i]

    def show_result(self):
        for each in self.dp:
            print(each)

        print("=" * 36)
        print(f"max value: {self.dp[-1][-1]}")
        print("-" * 36)
        print(f"item: {self.item}")
        for each in self.rec:
            print(each)

    def run(self):
        self.cal_max_value()
        self.track_item()
        self.show_result()


class Record(object):
    def __init__(self, idx_lst, max_val):
        self.idx_lst = idx_lst
        self.max_val = max_val

    def to_str(self):
        return f"({self.idx_lst},{self.max_val})"


class BagProblemSolution3(object):
    def __init__(self, caps, vals, capacity):
        self.c = caps
        self.v = vals
        self.capacity = capacity
        self.num = len(caps)
        self.d = [Record([], 0) for _ in range(capacity + 1)]

    def cal_max_value(self):
        for i in range(self.num):
            for j in range(self.capacity, 0, -1):
                from_cap = j - self.c[i]
                if from_cap >= 0 and self.d[from_cap].max_val + self.v[i] > self.d[j].max_val:
                    self.d[j].max_val = self.d[from_cap].max_val + self.v[i]
                    self.d[j].idx_lst = self.d[from_cap].idx_lst + [i]

    def show_result(self):
        print("=" * 36)
        print(f"max value: {self.d[-1].max_val}")
        print("-" * 36)
        print(f"item: {self.d[-1].idx_lst}")

    def run(self):
        self.cal_max_value()
        self.show_result()


def bag_problem_test():
    caps = [2, 4, 5, 6, 10, 3]  # [2, 3, 4, 5]  #
    vals = [1, 7, 4, 5, 11, 1]  # [3, 4, 5, 6]  #
    capacity = 7
    BagProblemSolution(caps, vals, capacity).run()
    BagProblemSolution3(caps, vals, capacity).run()


class Item(object):
    def __init__(self, idx, num, c, v):
        self.idx = idx
        self.num = num
        self.c = c
        self.v = v

    def to_str(self):
        return f"{self.idx},{self.num}: {self.c},{self.v}"


class CompleteBagProblemSolution(object):
    def __init__(self, caps, vals, capacity):
        self.capacity = capacity
        self.items = []
        self.num = 0
        self.gen_item(caps, vals, capacity)
        self.d = [Record([], 0) for _ in range(capacity + 1)]

    def gen_item(self, caps, vals, capacity):
        num = len(caps)
        for i in range(num):
            n = capacity // caps[i]
            if n < 1:
                continue
            for k in range(1, n + 1):
                self.items.append(Item(i, k, k * caps[i], k * vals[i]))
                self.num += 1

    def cal_max_value(self):
        for i in range(self.num):
            for j in range(self.capacity, 0, -1):
                from_cap = j - self.items[i].c
                if from_cap >= 0 and self.d[from_cap].max_val + self.items[i].v > self.d[j].max_val:
                    self.d[j].max_val = self.d[from_cap].max_val + self.items[i].v
                    self.d[j].idx_lst = self.d[from_cap].idx_lst + [(self.items[i].idx, self.items[i].num)]

    def show_result(self):
        print("=" * 36)
        print(f"max value: {self.d[-1].max_val}")
        print("-" * 36)
        print(f"item: {self.d[-1].idx_lst}")

    def run(self):
        self.cal_max_value()
        self.show_result()


def complete_bag_problem_test():
    caps = [2, 3, 4, 5]  # [2, 3, 4, 5]
    vals = [50, 160, 180, 190]  # [30, 50, 100, 200]
    capacity = 8
    CompleteBagProblemSolution(caps, vals, capacity).run()
    print("done")


class MultiBagProblemSolution(object):
    def __init__(self, caps, vals, nums, capacity):
        self.capacity = capacity
        self.items = []
        self.num = 0
        self.gen_item(caps, vals, nums, capacity)
        self.d = [Record([], 0) for _ in range(capacity + 1)]

    def gen_item(self, caps, vals, nums, capacity):
        num = len(caps)
        for i in range(num):
            n = min(capacity // caps[i], nums[i])
            if n < 1:
                continue
            for k in range(1, n + 1):
                self.items.append(Item(i, k, k * caps[i], k * vals[i]))
                self.num += 1

    def cal_max_value(self):
        for i in range(self.num):
            for j in range(self.capacity, 0, -1):
                from_cap = j - self.items[i].c
                if from_cap >= 0 and self.d[from_cap].max_val + self.items[i].v > self.d[j].max_val:
                    self.d[j].max_val = self.d[from_cap].max_val + self.items[i].v
                    self.d[j].idx_lst = self.d[from_cap].idx_lst + [(self.items[i].idx, self.items[i].num)]

    def show_result(self):
        print("=" * 36)
        print(f"max value: {self.d[-1].max_val}")
        print("-" * 36)
        print(f"item: {self.d[-1].idx_lst}")

    def run(self):
        self.cal_max_value()
        self.show_result()


def multi_bag_problem_test():
    caps = [2, 3, 4, 5]  # [2, 3, 4, 5]
    vals = [50, 160, 180, 190]  # [30, 50, 100, 200]
    nums = [4, 1, 2, 1]
    capacity = 8
    MultiBagProblemSolution(caps, vals, nums, capacity).run()
    print("done")


def main():
    multi_bag_problem_test()


if __name__ == '__main__':
    main()
