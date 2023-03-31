def largest_rectangle_area(arr):
    if arr is None or len(arr) == 0:
        return 0

    n = len(arr)
    stack = [0] * n
    cnt = -1
    left = [0] * n
    right = [0] * n
    for i in range(n):
        while cnt > -1 and arr[stack[cnt]] >= arr[i]:
            cnt -= 1
        if cnt == -1:
            left[i] = 0
        else:
            left[i] = stack[cnt] + 1
        cnt += 1
        stack[cnt] = i

    cnt, ans = -1, 0
    for i in range(n - 1, -1, -1):
        while cnt > -1 and arr[stack[cnt]] >= arr[i]:
            cnt -= 1
        if cnt == -1:
            right[i] = n - 1
        else:
            right[i] = stack[cnt] - 1
        cnt += 1
        stack[cnt] = i
        ans = max(ans, arr[i] * (right[i] - left[i] + 1))

    return ans


def largest_rectangle_area2(heights) -> int:
    heights = [0] + heights + [0]
    stack, max_area = [], 0

    for hi_index, height in enumerate(heights):

        while stack and height < heights[stack[-1]]:
            popped_index = stack.pop()
            lo_index = stack[-1] + 1

            area = heights[popped_index] * (hi_index - lo_index)
            max_area = max(max_area, area)

        stack.append(hi_index)

    return max_area


def largest_rectangle_area3(heights) -> int:
    stack = []
    res = 0
    heights = heights + [0]
    for right in range(len(heights)):
        while stack and heights[right] < heights[stack[-1]]:
            cur = stack.pop()
            if stack:
                left = stack[-1]
            else:
                left = -1
            res = max(res, (right - left - 1) * heights[cur])
        stack.append(right)
    return res


def main():
    arr = [2, 1, 5, 6, 2, 3]
    print(largest_rectangle_area(arr))
    print(largest_rectangle_area2(arr))
    print(largest_rectangle_area3(arr))


if __name__ == '__main__':
    main()
