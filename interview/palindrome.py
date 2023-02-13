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


def main():
    # print(is_palindrome2(56765))
    # print(is_palindrome2(56763))
    print(is_palindrome_str("abc"))
    print(is_palindrome_str("aba"))
    print(is_palindrome_str("abba"))
    print(is_palindrome_str("b"))


if __name__ == '__main__':
    main()
