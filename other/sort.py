from typing import List

def validate(arr):
    for i in range(len(arr)-1):
        if arr[i] > arr[i+1]:
            return False
    return True


def bubbleSort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-1, i, -1):
            if arr[j] < arr[j-1]:
                arr[j-1], arr[j] = arr[j], arr[j-1]
    return arr


def selectSort(arr):
    for i in range(len(arr)):
        minIndex = i
        for j in range(i, len(arr)):
            if arr[j] < arr[minIndex]:
                minIndex = j
        arr[i], arr[minIndex] = arr[minIndex], arr[i]
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


def quickSort2(arr: List[int]) -> List[int]:
    if not arr:
        return []
    pivot = arr[int(len(arr)/2)]
    left = []
    right = []
    middle = []
    for i in arr:
        if i<pivot:
            left.append(i)
        elif i>pivot:
            right.append(i)
        else:
            middle.append(i)
    return quickSort2(left) + middle + quickSort2(right)


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


def merge_to_A(A, m, B, n):
    '''
        给出一个有序数组A和有序数组B,将数组B合并到数组A中，变成一个有序的升序数组
        len(A) = m+n, len(B) = n
        方法一：新建C数组，先在C中排好序，再复制回A
        方法二：在A数组内部进行操作
            定义指针i,j,p：i,j指针指向原末尾，p指针指向合并后的A末尾
            比较A[i]和B[j]，其中大的，丢到A[p]中，然后p和该指针左移一格。
    '''
    i = m-1
    j = n-1
    p = n+m-1
    while i>=0 and j>=0:
        if A[i] > B[j]:
            A[p] = A[i]
            i -= 1
        else:
            A[p] = B[j]
            j -= 1
        p -= 1
    
    if j>=0:    # 如果B数组还有剩余，则直接全部移到A数组中。
        A[:p+1] = B[:j+1]


if __name__ == '__main__':
    # [-49, 0, 1, 1, 15, 31, 32, 32, 43, 52, 55, 82, 91, 123, 9292]
    a = [52, 32, 91, 123, 1, 31, 43, 32, 55, 82, 1, 9292, -49, 0, 15]
    # print(heapSort(a))
    print(bubbleSort(a))
    # print(validate(a))

