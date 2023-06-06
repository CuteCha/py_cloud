# -*- encoding:utf-8 -*-

def rand(n=5):
    import random
    return random.randint(0, n - 1)


def rand2():
    n = 5
    if n % 2 == 0:
        return rand(n) % 2
    else:
        x = rand(n)
        m = n - 1
        while x >= m:
            x = rand(n)
        return x % 2


def rand7a():
    x = rand() * 5 + rand()
    while x > 21:
        x = rand() * 5 + rand()

    return x % 7


def rand7b():
    x = rand2() * 2 ** 2 + rand2() * 2 + rand2()
    while x >= 7:
        x = rand2() * 2 ** 2 + rand2() * 2 + rand2()
    return x


def rand8a():
    x = rand() * 5 + rand()
    while x > 24:
        x = rand() * 5 + rand()

    return x % 8


def rand8b():
    x = rand2() * 2 ** 2 + rand2() * 2 + rand2()
    return x


def check():
    d = {k: 0 for k in range(8)}
    for _ in range(10000):
        k = rand8b()
        d[k] += 1

    print(d)


def main():
    # print([rand7a() for _ in range(50)])
    # print([rand7b() for _ in range(50)])
    # print([rand8a() for _ in range(50)])
    # print([rand8b() for _ in range(50)])
    check()


if __name__ == '__main__':
    main()
