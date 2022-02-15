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


if __name__ == '__main__':
    print(max_reverse('ddbcdddccabbdcdbdbccdaabcdaaaabaadcbadaabdcbaccdabdbccacdddcdcaacdadbacbbccabdabdcddbaacbadacdadbaccbcaadddddddabddbabdaaacdddcd'))