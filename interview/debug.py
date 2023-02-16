# -*- encoding:utf-8 -*-

def find_disappear_numbers(arr):
    n = len(arr)
    for num in arr:
        x = (num - 1) % n
        arr[x] += n

    ret = [i + 1 for i, num in enumerate(arr) if num <= n]
    return ret


def main():
    arr = [2, 3, 1, 2, 1]
    print(find_disappear_numbers(arr))


if __name__ == '__main__':
    main()
