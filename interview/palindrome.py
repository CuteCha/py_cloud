# -*- encoding:utf-8 -*-

def is_palindrome(n):
    if n < 0:
        return False

    rever = 0
    t = n
    while n > 0:
        rever = rever * 10 + n % 10
        n //= 10

    return rever == t


def is_palindrome2(n):
    if n < 0:
        return False

    rever = 0
    while n > rever:
        rever = rever * 10 + n % 10
        n //= 10
        print(f"r={rever}, n={n}")

    return rever == n or rever // 10 == n


def is_palindrome_str(s):
    l, r = 0, len(s) - 1
    res = True
    while l < r:
        if s[l] == s[r]:
            l += 1
            r -= 1
        else:
            res = False
            break

    return res


def longest_palindrome(s):
    """
    d[i]代表s[:i]的最长回文子串的长度
    递推公式:
          d[i-1]+2,  s[i-l-1:i+1]回文;
    d[i]= d[i-1]+1,  s[i-l:i+1]回文;
          d[i-1],    其他;
    """
    n = len(s)
    l = 0
    start = 0
    for i in range(n):
        print("-" * 36)
        print(f"1. i={i}, l={l}, start={start}")
        if i - l >= 1 and s[i - l - 1: i + 1] == s[i - l - 1: i + 1][::-1]:
            start = i - l - 1
            l += 2
            print(f"2. i={i}, l={l}, start={start}")
            continue

        if i - l >= 0 and s[i - l: i + 1] == s[i - l: i + 1][::-1]:
            start = i - l
            l += 1
            print(f"3. i={i}, l={l}, start={start}")

        print("=" * 36)
    return s[start: start + l]


def longest_palindrome2(s):
    n = len(s)
    l = 0
    start = 0
    dp = [[False for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if (i - j) < 2:
                dp[j][i] = (s[i] == s[j])
            else:
                dp[j][i] = dp[j + 1][i - 1] and s[i] == s[j]

            if dp[j][i] and (i - j + 1) > l:
                l = i - j + 1
                start = j

    return s[start: start + l]


def main():
    # print(is_palindrome2(56765))
    # print(is_palindrome2(56763))
    # print(is_palindrome_str("abc"))
    # print(is_palindrome_str("aba"))
    # print(is_palindrome_str("abba"))
    # print(is_palindrome_str("b"))
    print(longest_palindrome("xaxbbab"))


if __name__ == '__main__':
    main()
