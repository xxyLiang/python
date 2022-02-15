def kmp(s, sub, pos=1):
    '''
        返回字串sub在主串s中第pos个字符之后的位置（index, start from 0），若不存在返回-1
    '''
    # get next[]
    next = []
    i = 0
    next.append(-1)
    j = -1
    while(i < len(sub)):
        if j == -1 or sub[i] == sub[j]:
            i += 1
            j += 1
            next.append(j)
        else:
            j = next[j]
    # kmp
    i = pos-1 if pos > 0 else 0
    j = 0
    while(i<len(s) and j<len(sub)):
        if(j == -1 or s[i] == sub[j]):
            i += 1
            j += 1
        else:
            j = next[j]
    if j >= len(sub):
        return i-len(sub)
    else:
        return -1


def max_reverse(s):
    res = 1
    for i in range(len(s)):
        j = 0
        # 先默认回文串长度为奇数
        while(i-j-1>=0 and i+j+1<len(s)):
            if s[i-j-1] == s[i+j+1]:
                j += 1
            else:
                break
        res = max(res, 2*j+1)

        # 如果s[i] == s[i+1]，考虑回文串长度为偶数
        if i+1<len(s) and s[i] == s[i+1]:
            j = 0
            while(i-j-1>=0 and i+2+j<len(s)):
                if s[i-j-1] == s[i+2+j]:
                    j += 1
                else:
                    break
            res = max(res, 2*j+2)
        
    return res

def maxUniqueLength(arr) -> int:
        # write code here
        longest_length,i = 0,-1
        arr_length = len(arr)
        l_str = {}      # 出现过的字符及其最后出现的位置
        for j in range(arr_length):         # i左指针，j右指针
            if arr[j] in l_str and l_str[arr[j]] > i:       # 当前字符在l_str中出现，且最后出现在i~j范围内，才算重复
                i = l_str[arr[j]]
            l_str[arr[j]] = j                               # 记录当前字符最后出现位置
            longest_length = max(longest_length, j-i)
        return max(longest_length, j-i)


def maxSameString(s):       # 返回最长重复子字符串
    last = ''
    leng = maxlen = start = 0       # 由于要返回字符串，还需记录字符串开始位置
    for i in range(len(s)):
        if s[i] == last:            # 如果相同，则继续增加
            leng += 1
        else:
            if leng > maxlen:       # 如果刷新了长度，更新maxlen，更新最长字符串开始位置为i-leng
                maxlen = leng 
                start = i-leng             
            last = s[i]
            leng = 1
    return s[start:start+maxlen]



if __name__ == '__main__':
    print(max_reverse('ddbcdddccabbdcdbdbccdaabcdaaaabaadcbadaabdcbaccdabdbccacdddcdcaacdadbacbbccabdabdcddbaacbadacdadbaccbcaadddddddabddbabdaaacdddcd'))