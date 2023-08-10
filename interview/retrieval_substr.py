import re


def find(sub_str, str_ori):
    idx = str_ori.find(sub_str)

    arr = []

    while idx != -1:
        arr.append(idx)
        idx = str_ori.find(sub_str, (idx + 1))

    cnt = len(arr)
    if cnt > 0:
        print(f"match {cnt}, position:\n {arr}")
    else:
        print(f"not match")


def find02(sub_str, str_ori):
    for m in re.finditer(sub_str, str_ori):
        print(m.start(), m.end())


def kmp(m_str, s_str):
    # m_str表示主串，s_str表示模式串

    # 求next数组
    next_ls = [-1] * len(s_str)
    m = 1  # 从1开始匹配
    s = 0
    next_ls[1] = 0
    while m < len(s_str) - 1:
        if s_str[m] == s_str[s] or s == -1:
            m += 1
            s += 1
            next_ls[m] = s
        else:
            s = next_ls[s]
    #  print(next_ls)  检查next数组
    # KMP
    i = j = 0  # i,j位置指针初始值为0
    while i < len(m_str) and j < len(s_str):
        # 模式串遍历结束匹配成功，主串遍历结束匹配失败
        # 匹配成功或失败后退出
        if m_str[i] == s_str[j] or j == -1:
            # 把j==-1时纳入到条件判断中，实现i+1，j归零
            i += 1
            j += 1
        else:
            j = next_ls[j]

    if j == len(s_str):
        return i - j  # 匹配成功
    return -1  # 匹配失败


def main():
    str_ori = "中国陆地面积约960万平方千米，东部和南部大陆海岸线1.8万多千米，海域总面积约473万平方千米 [2] 。海域分布有大小岛屿7600多个，其中台湾岛最大，面积35798平方千米 [2] 。中国同14国接壤，与8国海上相邻。省级行政区划为23个省、5个自治区、4个直辖市、2个特别行政区。"
    sub_str = "海"
    print(str_ori.index(sub_str))
    print(str_ori.find(sub_str))
    find(sub_str, str_ori)
    find02(sub_str, str_ori)

    # 测试
    print(kmp('decdagee', 'age'))


if __name__ == '__main__':
    main()
