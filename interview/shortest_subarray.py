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


def max_sum_of_sub(arr):
    if arr is None or len(arr) == 0:
        return None

    res = -1000
    s = arr[0]
    for i in range(1, len(arr), 1):
        if s < 0:
            s = arr[i]
        else:
            s += arr[i]

        res = max(res, s)

    return res


def max_sum_of_sub2(arr):
    if arr is None or len(arr) == 0:
        return None

    res = -1000
    pre_s = 0
    s = 0
    for i in range(1, len(arr), 1):
        s = max(pre_s + arr[i], arr[i])
        pre_s = s
        res = max(s, res)

    return res


def main():
    # arr = [1, 1, 1, 1, -3, 1, 2]
    # k = 3
    # print(shortest_sub_arr2(arr, k))
    arr = [0, 31, -41, 59, 26, -53, 58, 97, -93, -23, 84]
    print(max_sum_of_sub(arr))
    print(max_sum_of_sub2(arr))


if __name__ == '__main__':
    main()
