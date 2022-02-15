def validate(arr):
    for i in range(len(arr)-1):
        if arr[i] > arr[i+1]:
            return False
    return True


def bubbleSort(arr):
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
    return arr


def mergeSort(arr):
    if len(arr) == 1:
        return arr
    m = int(len(arr)/2)
    left = mergeSort(arr[:m])
    right = mergeSort(arr[m:])

    #merge
    i = j = 0
    while(i+j < len(arr)):
        if j == len(right) or (i < len(left) and left[i] < right[j]):
            arr[i+j] = left[i]
            i += 1
        else:
            arr[i+j] = right[j]
            j += 1

    return arr


def heapSort(arr):

    def heapify(arr, arrlen, node):   # node是要维护的节点
        l = 2*node
        r = l+1
        largest = l if (l <= arrlen-1 and arr[l]>arr[node]) else node
        largest = r if (r <= arrlen-1 and arr[r]>arr[largest]) else largest
        if largest != node:
            arr[node], arr[largest] = arr[largest], arr[node]
            heapify(arr, arrlen, largest)
    
    # 自底向上构建最大堆
    i = len(arr)-1
    while(i >= 0):
        heapify(arr, len(arr), i)
        i -= 1
    
    # 开始提取最大值
    arrlen = len(arr)-1  # 代表当前数组长度，arr[arrlen:]已排好序，不需再考虑
    while(arrlen >= 0):
        arr[0], arr[arrlen] = arr[arrlen], arr[0]     # 此时arr[0]为最大值，交换到数组尾
        heapify(arr, arrlen, 0)  # 维护根节点的最大堆
        arrlen -= 1
    return arr
        

def quickSort(arr):
    def partition(arr, l, r):
        if l >= r:
            return 0
        left = l
        right = r
        pivot = arr[l]
        while l < r:
            while l < r and arr[r] >= pivot:
                r -= 1
            arr[l], arr[r] = arr[r], arr[l]
            while l < r and arr[l] <= pivot:
                l += 1
            arr[l], arr[r] = arr[r], arr[l]
        partition(arr, left, l-1)
        partition(arr, l+1, right)
    
    partition(arr, 0, len(arr)-1)
    return arr


def get_k_min(arr, k):
    def partition(arr, l, r, k):
        if l >= r:
            return 0
        left = l
        right = r
        pivot = arr[l]
        while l < r:
            while l < r and arr[r] >= pivot:
                r -= 1
            arr[l], arr[r] = arr[r], arr[l]
            while l < r and arr[l] <= pivot:
                l += 1
            arr[l], arr[r] = arr[r], arr[l]
        partition(arr, left, l-1, k)
        if k > l+1:
            partition(arr, l+1, right, k)


    if k==0 or k>len(arr):
        return []
    partition(arr, 0, len(arr)-1, k)
    return arr[:k]


def get_kth_max(arr, k):
    def partition(arr, l, r):
        if l >= r:
            return l
        pivot = arr[l]
        while l < r:
            while l < r and arr[r] <= pivot:
                r -= 1
            arr[l], arr[r] = arr[r], arr[l]
            while l < r and arr[l] >= pivot:
                l += 1
            arr[l], arr[r] = arr[r], arr[l]
        return l

   
    left, right = 0, len(arr)-1
    if k<=0 or k>len(arr):
        return -100
    while(left <= right):
        pivot = partition(arr, left, right)
        if pivot == k-1:
            return arr[pivot]
        elif pivot < k-1:
            left = pivot + 1
        else:
            right = pivot - 1
    return -100


if __name__ == '__main__':
    # [-49, 0, 1, 1, 15, 31, 32, 32, 43, 52, 55, 82, 91, 123, 9292]
    a = [52, 32, 91, 123, 1, 31, 43, 32, 55, 82, 1, 9292, -49, 0, 15]
    # print(heapSort(a))
    print(get_kth_max(a,2))
    # print(validate(a))

