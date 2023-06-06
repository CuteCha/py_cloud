# -*- encoding:utf-8 -*-

def find_disappear_numbers(arr):
    n = len(arr)
    for num in arr:
        x = (num - 1) % n
        arr[x] += n

    return [i + 1 for i, num in enumerate(arr) if num <= n]


def get_median02(arr1, arr2):
    m = len(arr1)
    n = len(arr2)

    start = 0
    end = m
    k = (m + n + 1) // 2

    while start <= end:
        mid = (start + end) // 2
        i = mid
        j = k - mid

        a1 = arr1[i - 1] if (i > 0) else float('-inf')
        a2 = arr2[j - 1] if (j > 0) else float('-inf')
        b1 = arr1[i] if (i < m) else float('inf')
        b2 = arr2[j] if (j < n) else float('inf')

        if a1 <= b2 and a2 <= b1:
            if (n + m) % 2 == 0:
                return (max(a1, a2) + min(b1, b2)) / 2.0
            return max(a1, a2)

        elif a1 > b2:
            end = mid - 1
        else:
            start = mid + 1


def get_median03(arr1, arr2):
    n1 = len(arr1)
    n2 = len(arr2)
    k = (n1 + n2 + 1) // 2
    print(f"k={k}")
    l = 0
    r = n1
    while l < r:
        m1 = l + (r - 1) // 2
        m2 = k - m1
        print(f"m1={m1}, m2={m2}")
        if arr1[m1] < arr2[m2 - 1]:  # arr1里的取的个数偏少了
            l = m1 + 1
        else:
            r = m1

    m1 = l
    m2 = k - l
    print(f">>> m1={m1}, m2={m2}")

    c1 = max(float("-inf") if m1 <= 0 else arr1[m1 - 1], float("-inf") if m2 <= 0 else arr2[m2 - 1])
    if (n1 + n2) % 2 == 1:
        return c1

    c2 = min(float("inf") if m1 >= n1 else arr1[m1], float("inf") if m2 >= n2 else arr2[m2])
    return (c1 + c2) / 2


def main():
    arr1 = [9, 10, 11]
    arr2 = [5, 8, 10, 20]

    print(get_median02(arr1, arr2))
    print(get_median03(arr1, arr2))


if __name__ == '__main__':
    main()
