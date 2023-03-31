def search0(arr, target):
    left = 0
    right = len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            right = mid
        elif arr[mid] > target:
            right = mid - 1
        elif arr[mid] < target:
            left = mid + 1

    if arr[left] == target:
        return left
    else:
        return -1


'''
while(l<=h){
            mid = (l+h)/2;
            if(nums[mid] == target){
                while(mid != 0 && nums[mid]==nums[mid-1]){
                    mid--;
                }
                return mid;
            }
            else if(nums[mid]>target){
                h = mid-1;
            }
            else{
                l = mid+1;
            }
        }
        return -1; //查找失败

'''


def search(arr, target):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (right + left) // 2
        if arr[mid] == target:
            while mid != 0 and arr[mid] == arr[mid - 1]:
                mid -= 1

            return mid

        elif arr[mid] > target:
            right = mid - 1
        else:
            left = mid + 1

    return -1


def main():
    arr = [2, 4, 5, 5, 5, 7, 8, 12, 15]
    target = 2
    print(search(arr, target))


if __name__ == '__main__':
    main()
