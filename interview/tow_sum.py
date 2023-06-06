# -*- encoding:utf-8 -*-

def tow_sum(lst, val):
    n = len(lst)
    d = dict()
    for i in range(n):
        k = val - lst[i]
        if k in d:
            return d[k], i
        d.setdefault(lst[i], i)


def main():
    nums = [2, 7, 11, 15]
    target = 22
    print(tow_sum(nums, target))


if __name__ == '__main__':
    main()
