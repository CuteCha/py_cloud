# -*- encoding:utf-8 -*-

def intersection(arr1, arr2):
    res = []
    d = dict()
    for i in arr1:
        d.setdefault(i, 1)
    for i in arr2:
        if i in d and d.get(i) == 1:
            res.append(i)
        d[i] = 2

    return res


def main():
    arr1 = [1, 2, 2, 1, 5]
    arr2 = [2, 5, 2]
    print(intersection(arr1, arr2))


if __name__ == '__main__':
    main()
