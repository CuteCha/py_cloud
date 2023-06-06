# -*- encoding:utf-8 -*-

def seg_str(s):
    stack1 = []
    ss = s.split(' ')
    for i in ss:
        stack1.append(i)
    stack2 = []

    while stack1:
        stack2.append(stack1[-1])
        stack2.append(' ')
        del stack1[-1]
    del stack2[-1]

    return ''.join(stack2)


def reverse_lst(s):
    l, r = 0, len(s) - 1
    while l < r:
        s[l], s[r] = s[r], s[l]
        l += 1
        r -= 1
    return s


def reverse_str(p_str, s, e):
    return p_str[s:e + 1][::-1]


def reverse_words(p_str):
    n = len(p_str)
    s = 0
    e = 0
    k = 0

    while k < n:
        if k == n - 1:
            pass


def find_idx(arr, target, start, end):
    if start >= end and arr[start] != target:
        return -1

    p = (start + end) // 2
    if arr[p] == target:
        return p
    elif arr[start] < arr[end]:
        if arr[p] > target:
            return find_idx(arr, target, start, p - 1)
        else:
            return find_idx(arr, target, p + 1, end)
    else:
        if arr[start] > target:
            return find_idx(arr, target, start + 1, end)
        else:
            return find_idx(arr, target, start, end - 1)


def find_idx2(arr, target, start, end):
    if start >= end and arr[start] != target:
        return -1

    p = (start + end) // 2
    if arr[p] == target:
        return p

    if arr[start] <= arr[p]:
        if arr[start] <= target < arr[p]:
            return find_idx2(arr, target, start, p - 1)
        else:
            return find_idx2(arr, target, p + 1, end)
    else:
        if arr[p] < target <= arr[end]:
            return find_idx2(arr, target, p + 1, end)
        else:
            return find_idx2(arr, target, start, p - 1)


def search(nums, target, low, high):
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == target:
            return mid
        elif nums[low] < nums[high]:
            if nums[mid] > target:
                high = mid - 1
            else:
                low = mid + 1
        else:
            if nums[low] > target:
                low += 1
            else:
                high -= 1
    return -1


def main():
    # s = "abc def gh"
    # reversed(s)
    # # print(seg_str(s))
    # print(reverse_str(s, 0, 3))
    arr = [4, 5, 6, 7, 0, 1, 2]
    print(find_idx2(arr, 0, 0, 6))
    print(search(arr, 3, 0, 6))


if __name__ == '__main__':
    main()
