# -*- encoding:utf-8 -*-


def build_max_heap(arr):
    arr_len = len(arr)
    for i in range(arr_len // 2, -1, -1):
        heapify(arr, arr_len, i)


def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


def heapify(arr, arr_len, i):
    left = 2 * i + 1
    right = 2 * i + 2
    largest = i

    if left < arr_len and arr[left] > arr[largest]:
        largest = left

    if right < arr_len and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        swap(arr, i, largest)
        heapify(arr, arr_len, largest)


def heap_sort(arr):
    arr_len = len(arr)
    build_max_heap(arr)

    for i in range(arr_len - 1, 0, -1):
        swap(arr, 0, i)
        arr_len -= 1
        heapify(arr, arr_len, 0)


def main():
    arr = [3, 1, 5, 9, 2, 3, 5, 8, 4, 6, 7]
    heap_sort(arr)

    print(arr)


if __name__ == '__main__':
    main()
