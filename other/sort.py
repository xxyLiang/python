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

    def heapify(arr, arrlen, i):
        l = 2*i
        r = l+1
        largest = l if (l <= arrlen-1 and arr[l]>arr[i]) else i
        largest = r if (r <= arrlen-1 and arr[r]>arr[largest]) else largest
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, arrlen, largest)
    
    i = int(len(arr))-1
    while(i >= 0):
        heapify(arr, len(arr), i)
        i -= 1
    i = len(arr)-1
    while(i >= 0):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
        i -= 1
    return arr
        

def quickSort(arr, l, r):
    if l >= r:
        return arr
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
    quickSort(arr, left, l-1)
    quickSort(arr, l+1, right)
    return arr



if __name__ == '__main__':
    a = [52, 32, 91, 123, 1, 31, 43, 32, 55, 82, 9292, -49, 0, 15]
    a = quickSort(a, 0, len(a)-1)
    print(a)
    print(validate(a))

