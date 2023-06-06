# -*- encoding:utf-8 -*-

def find_lcs_sub(a, b):
    m = len(a)
    n = len(b)
    d = [[0 for _ in range(n + 1)] for _ in range(m + 1)]  # row: m, col: n
    m_lcs = 0
    p = 0
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                d[i + 1][j + 1] = d[i][j] + 1
                if d[i + 1][j + 1] > m_lcs:
                    m_lcs = d[i + 1][j + 1]
                    p = i

    print(f"m: {m}, n: {n}\n")
    r = [[str(e) for e in each] for each in d]
    for j in range(n):
        r[0][1 + j] = b[j]
    for i in range(m):
        r[1 + i][0] = a[i]
    print("\n".join([",".join(each) for each in r]))
    print("=" * 36)

    return m_lcs, a[p + 1 - m_lcs: p + 1]


def find_lcs_seq(a, b):
    m = len(a)
    n = len(b)
    if m == 0 or n == 0:
        return 0

    d = [[0 for _ in range(n + 1)] for _ in range(m + 1)]  # row: m, col: n

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                d[i][j] = d[i - 1][j - 1] + 1
            else:
                d[i][j] = max(d[i - 1][j], d[i][j - 1])

    lcs = ""
    i, j = m, n
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1] and d[i][j] == d[i - 1][j - 1] + 1:
            lcs = a[i - 1] + lcs
            i, j = i - 1, j - 1
            continue
        if d[i][j] == d[i - 1][j]:
            i, j = i - 1, j
            continue
        if d[i][j] == d[i][j - 1]:
            i, j = i, j - 1
            continue

    return d[m][n], lcs


def print_matrix(a):
    print("\n".join([",".join([str(e) for e in each]) for each in a]))


def recursive_lcs_seq(str_a, str_b):
    if len(str_a) == 0 or len(str_b) == 0:
        return 0

    if str_a[0] == str_b[0]:
        return recursive_lcs_seq(str_a[1:], str_b[1:]) + 1
    else:
        return max([recursive_lcs_seq(str_a[1:], str_b), recursive_lcs_seq(str_a, str_b[1:])])


def main():
    print(find_lcs_sub('abfcdfg', 'abcdfg'))
    print(find_lcs_seq('abfcdfg', 'axbcdfg'))
    print(recursive_lcs_seq('abfcdfg', 'axbcdfg'))


if __name__ == '__main__':
    main()
