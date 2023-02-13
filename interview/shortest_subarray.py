# -*- encoding:utf-8 -*-


def shortest_sub_arr(arr, k):
    from collections import deque
    n = len(arr)
    pre_sum_arr = [0]

    for num in arr:
        pre_sum_arr.append(pre_sum_arr[-1] + num)

    print(f"pre_sum_arr: {pre_sum_arr}")
    q = deque()
    res = n + 1
    for i, cur_sum in enumerate(pre_sum_arr):
        print("-" * 36)
        print(f"q1: {q}")
        while q and cur_sum - pre_sum_arr[q[0]] >= k:
            res = min(res, i - q.popleft())

        print(f"q2: {q}")
        while q and pre_sum_arr[q[-1]] >= cur_sum:
            q.pop()
        q.append(i)
        print(f"i: {i}; q: {q}; res: {res}")

    return res if res < n + 1 else -1


def shortest_sub_arr2(arr, k):
    n = len(arr)
    pre_sum_arr = [0]

    for num in arr:
        pre_sum_arr.append(pre_sum_arr[-1] + num)

    print(f"pre_sum_arr: {pre_sum_arr}")
    q = list()
    res = n + 1
    for i, cur_sum in enumerate(pre_sum_arr):
        print("-" * 36)
        print(f"q1: {q}")
        while q and cur_sum - pre_sum_arr[q[0]] >= k:
            res = min(res, i - q[0])
            q = q[1:]

        print(f"q2: {q}")
        while q and pre_sum_arr[q[-1]] >= cur_sum:
            q = q[:-1]
        q.append(i)
        print(f"i: {i}; q: {q}; res: {res}")

    return res if res < n + 1 else -1


def main():
    arr = [1, 1, 1, 1, -3, 1, 2]
    k = 3
    print(shortest_sub_arr2(arr, k))


if __name__ == '__main__':
    main()
