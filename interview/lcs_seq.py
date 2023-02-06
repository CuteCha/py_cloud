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
    # print("\n".join([",".join([str(e) for e in each]) for each in d]))
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
    d = [[0] * (n + 1)] * (m + 1)
    m_lcs = 0
    p = 0
    for i in range(m):
        for j in range(n):
            if a[i] == a[j]:
                d[i + 1][j + 1] = d[i][j] + 1
            else:
                d[i + 1][j + 1] = max(d[i][j + 1], d[i + 1][j])

    return m_lcs


def main():
    print(find_lcs_sub('abfcdfg', 'abcdfg'))


if __name__ == '__main__':
    main()
