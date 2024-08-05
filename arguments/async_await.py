import asyncio
from datetime import datetime

from time import sleep


# 定义一个异步函数
async def greet(name):
    print("Hello, " + name + str(datetime.now()))
    await asyncio.sleep(2)  # 使用异步的sleep函数
    print("Goodbye, " + name + str(datetime.now()))


# 执行异步函数
async def main():
    # 创建任务并发执行
    task1 = asyncio.create_task(greet("码农"))
    task2 = asyncio.create_task(greet("研究僧"))

    # 等待所有任务完成
    await asyncio.gather(task1, task2)


def run00():
    start = datetime.now()  # 记录程序开始执行的时间
    asyncio.run(main())  # 运行主函数
    end = datetime.now()  # 记录程序结束执行的时间
    print('elapsed time =', end - start)  # 输出执行时间


def greet01(name):
    print("Hello, " + name + str(datetime.now()))
    sleep(2)
    print("Goodbye, " + name + str(datetime.now()))


def run01():
    start = datetime.now()
    greet01("码农")
    greet01("研究僧")
    end = datetime.now()
    print('elapsed time =', end - start)


if __name__ == "__main__":
    run00()
    print("--" * 36)
    run01()
