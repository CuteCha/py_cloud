# -*- encoding:utf-8 -*-

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
    def __init__(self, from_cap, idx_lst, max_val):
        self.from_cap = from_cap
        self.idx_lst = idx_lst
        self.max_val = max_val

    def to_str(self):
        return f"({self.from_cap},{self.idx_lst},{self.max_val})"


class BagProblemSolution3(object):
    def __init__(self, caps, vals, capacity):
        self.c = caps
        self.v = vals
        self.capacity = capacity
        self.num = len(caps)
        self.d = [Record(0, [], 0) for _ in range(capacity + 1)]

    def cal_max_value(self):
        for i in range(self.num):
            for j in range(self.capacity, 0, -1):
                if j >= self.c[i] and self.d[j - self.c[i]].max_val + self.v[i] > self.d[j].max_val:
                    self.d[j].max_val = self.d[j - self.c[i]].max_val + self.v[i]
                    from_cap = j - self.c[i]
                    self.d[j].from_cap = from_cap
                    self.d[j].idx_lst = self.d[from_cap].idx_lst + [i]

            # print("\t".join([each.to_str() for each in self.d]))

    def show_result(self):
        print("=" * 36)
        print(f"max value: {self.d[-1].max_val}")
        print("-" * 36)
        print(f"item: {self.d[-1].idx_lst}")

    def run(self):
        self.cal_max_value()
        self.show_result()


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


def main():
    # bag_problem_solution()
    caps = [2, 4, 5, 6, 10, 3]  # [2, 3, 4, 5]  #
    vals = [1, 7, 4, 5, 11, 1]  # [3, 4, 5, 6]  #
    capacity = 7
    BagProblemSolution(caps, vals, capacity).run()
    BagProblemSolution3(caps, vals, capacity).run()


if __name__ == '__main__':
    main()
