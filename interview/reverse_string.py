# -*- encoding:utf-8 -*-

def reverse_str(s):
    l, r = 0, len(s)-1
    while l < r:
        s[l], s[r] = s[r], s[l]
        l += 1
        r -= 1
    print(s)


def main():
    s = "hello"
    reverse_str(list(s))


if __name__ == '__main__':
    main()
