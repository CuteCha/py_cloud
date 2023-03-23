# -*- encoding:utf-8 -*-

def longest_consecutive(arr):
    if arr is None or len(arr) == 0:
        return
    max_val = -2
    d = dict()

    for i in arr:
        if i not in d:
            l = 0
            r = 0

            if (i - 1) in d:
                l = d[i - 1]
            if (i + 1) in d:
                r = d[i + 1]

            s = r + l
            print(f"i={i},l={l},r={r},s={s},d={d}")
            d[i] = s + 1
            d[i - l] = s + 1
            d[i + r] = s + 1

            max_val = max(max_val, s + 1)

    return max_val


def longest_consecutive2(arr):
    if arr is None or len(arr) == 0:
        return None

    d = dict()
    max_val = -1
    max_ele = -1
    for i in arr:
        if i not in d:
            l = 0
            r = 0

            if (i - 1) in d:
                l = d[i - 1]
            if (i + 1) in d:
                r = d[i + 1]

            s = r + l
            d[i] = s + 1
            d[i - l] = s + 1
            d[i + r] = s + 1

            max_ele = i + r
            max_val = max(max_val, s + 1)

    return max_val, max_ele


def go_step(n):
    if n < 3:
        return n
    n1 = 1
    n2 = 2
    s = 0
    for _ in range(3, n + 1):
        s = n1 + n2
        n1 = n2
        n2 = s

    return s


def go_step2(n):
    if n == 1:
        return 1
    elif n == 2:
        return 2
    else:
        return go_step2(n - 1) + go_step2(n - 2)


def main():
    arr = [100, 4, 200, 2, 3, 2]
    print(longest_consecutive2(arr))
    print(list(map(lambda x: go_step2(x), range(1, 7))))
    print(list(map(lambda x: go_step(x), range(1, 7))))


if __name__ == '__main__':
    main()
