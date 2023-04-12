def reverse(arr, l, r):
    while l < r:
        swap(arr, l, r)
        l += 1
        r -= 1


def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


def next_perm(arr):
    n = len(arr)
    i = n - 2
    while i >= 0 and arr[i] >= arr[i + 1]:
        i -= 1

    if i >= 0:
        j = n - 1
        while j >= 0 and arr[i] > arr[j]:
            j -= 1

        swap(arr, i, j)
    else:
        return None

    reverse(arr, i + 1, n - 1)

    return arr


def main():
    arr = [1, 2, 3]
    print(next_perm(arr))


if __name__ == '__main__':
    main()