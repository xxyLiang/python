import sys
import time

words = []
with open('./material/NTUSD_positive_simplified.txt', 'r', encoding='utf-16') as f:
    for w in f.readlines():
        flag = 0
        if '.' in w:
            continue
        for i in words:
            if i in w:
                flag = 1
        if flag == 0:
            words.append(w.strip('\n'))

with open('./material/pos.txt', 'w', encoding='utf-16') as f:
    for w in words:
        f.write(w + '\n')
