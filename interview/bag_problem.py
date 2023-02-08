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
    caps = [2, 3, 4, 5]
    vals = [3, 4, 5, 6]
    capacity = 8
    BagProblemSolution(caps, vals, capacity).run()


if __name__ == '__main__':
    main()
