import os
import time


def main():
    print(os.getcwd())
    print(time.ctime(os.path.getmtime(f"{os.getcwd()}/test/main.py")))


if __name__ == '__main__':
    main()
