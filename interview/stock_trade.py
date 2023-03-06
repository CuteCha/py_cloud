# -*- encoding:utf-8 -*-

def cal_idx_val(arr, n):
    res = list()
    min_idx = 0
    max_val = 0
    res.append((min_idx, max_val))
    for i in range(1, n):
        if arr[i] < arr[min_idx]:
            min_idx = i
        max_val = arr[i] - arr[min_idx]
        res.append((min_idx, max_val))

    return res


def cal_max_ret(arr):
    n = len(arr)
    idx_max_val_lst = cal_idx_val(arr, n)
    min_idx, mav_val = idx_max_val_lst[0]
    sale_idx = 0
    for i in range(1, n):
        idx, val = idx_max_val_lst[i]
        if val > mav_val:
            min_idx = idx
            sale_idx = i
            mav_val = val

    return mav_val, min_idx, sale_idx


def main():
    arr = [2, 2, 6, 2, 1, 9, 3]
    print(cal_max_ret(arr))


if __name__ == '__main__':
    main()
