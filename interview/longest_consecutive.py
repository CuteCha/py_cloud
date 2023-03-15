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
            d[i] = s + 1
            d[i - l] = s + 1
            d[i + r] = s + 1

            max_val = max(max_val, s + 1)

    return max_val


def longest_consecutive2(arr):
    if arr is None or len(arr) == 0:
        return None

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
            d[i] = s + 1
            d[i - l] = s + 1
            d[i + r] = s + 1

    max_ele = -1
    max_val = -2

    for k, v in d.items():
        if v > max_val:
            max_ele = k

    return max_val, max_ele


def main():
    arr = [100, 4, 200, 1, 3, 2]
    print(longest_consecutive2(arr))


if __name__ == '__main__':
    main()
