def threeSum(num):
    res = []
    n = len(num)
    if(not num or n<3): 
        return [] 
    num.sort()
    for i in range(n): 
        if(num[i]>0):      # num[i]>0，则右边所有数都大于0，不可能成立
            return res
        if(i>0 and num[i]==num[i-1]):    # i的数与上一个数一致，直接跳过
            continue

        L=i+1
        R=n-1

        while(L<R):
            if num[i]+num[L]+num[R]>0:
                R-=1
            elif num[i]+num[L]+num[R]<0:
                L+=1
            else:
                res.append([num[i], num[L], num[R]])
                while(L+1<R and num[L]==num[L+1]):      # 这里是跳到所有重复数的最后一个
                    L+=1
                while(L<R-1 and num[R]==num[R-1]):
                    R-=1
                L+=1    # 在上面去重后，num[L+1] 必定大于num[L]，因此sum必定大于0，所以R也同时-1
                R-=1
    return res


def twoSum(numbers, target: int):
    a = {}
    for i in range(len(numbers)):
        temp = target-numbers[i]
        if temp in a:
            return [a[temp]+1, i+1]         # 这里是返回index+1，视情况返回值
            # return [temp, numbers[i]].sort()
        else:
            a[numbers[i]] = i
    return []
