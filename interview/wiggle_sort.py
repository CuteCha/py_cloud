def swap(arr, k, l):
    arr[k], arr[l] = arr[l], arr[k]


def wiggle_sort(arr):
    """
    如果 a > b, 遇到下一个是 c， b > c，那么可知 a > c，交换b,c 得到 a > c < b
    同理另一种情况也对
    """
    n = len(arr)
    for i in range(n - 1):
        if i % 2 == 0 and arr[i] > arr[i + 1]:
            swap(arr, i, i + 1)
        elif i % 2 == 1 and arr[i] < arr[i + 1]:
            swap(arr, i, i + 1)


def main():
    arr = [1, 3, 2, 5, 4, 7, 6]  # [1, 2, 3, 4, 5, 6, 7]
    wiggle_sort(arr)
    print(arr)


if __name__ == '__main__':
    main()
