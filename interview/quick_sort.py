# -*- encoding:utf-8 -*-

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


def partition(arr, l, h):
    key = arr[l]
    low = l + 1
    high = h
    while low <= high:
        while low <= high and arr[high] >= key:
            high -= 1
        while low <= high and arr[low] < key:
            low += 1

        if low < high:
            swap(arr, low, high)

    swap(arr, l, high)

    return high


def quick_sort(arr, start, end):
    if start >= end:
        return
    index = partition(arr, start, end)
    quick_sort(arr, start, index - 1)
    quick_sort(arr, index + 1, end)


def quick_sort_top_k(arr, start, end, k):
    p = partition(arr, start, end)
    if p == k:
        return
    if p < k:
        quick_sort_top_k(arr, p + 1, end, k)
    if p > k:
        quick_sort_top_k(arr, start, p - 1, k)


def top_k(arr, k):
    n = len(arr)
    if k >= n:
        return arr
    quick_sort_top_k(arr, 0, n - 1, k)

    return arr[:k]


def find_idx(arr, target, start, end):
    if start >= end and arr[start] != target:
        return -1

    p = (start + end) // 2
    if arr[p] == target:
        return p

    if arr[start] <= arr[p]:
        if arr[start] <= target < arr[p]:
            return find_idx(arr, target, start, p - 1)
        else:
            return find_idx(arr, target, p + 1, end)
    else:
        if arr[p] < target <= arr[end]:
            return find_idx(arr, target, p + 1, end)
        else:
            return find_idx(arr, target, start, p - 1)


def main():
    arr = [3, 1, 5, 9, 2, 3, 5, 8, 4, 6, 7]
    print(top_k(arr, 7))
    arr = [3, 1, 5, 9, 2, 3, 5, 8, 4, 6, 7]
    quick_sort(arr, 0, len(arr) - 1)
    print(arr)

    arr2 = [4, 5, 6, 7, 0, 1, 2]
    print(find_idx(arr2, 0, 0, 6))
    print(find_idx(arr2, 3, 0, 6))


if __name__ == '__main__':
    main()
