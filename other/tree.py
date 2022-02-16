from typing import List

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def levelOrder(self , root: TreeNode) -> List[List[int]]:
    '''
        按层遍历二叉树，返回[ [第一层], [第二层], ...]
    '''
    qune, res = [], []
    if not root:    # 根节点为空直接返回
        return res
    qune.append(root)   # 队列保存每一层所有未访问的结点
    while len(qune)!=0:
        n = len(qune)       # 如果不需要每层生成一个一维数组的话，则不需记录当前层的结点数
        temp = []
        # 遍历每一层
        for i in range(n):
            node = qune.pop(0)
            temp.append(node.val)
            # 左右节点按顺序加到队尾
            if node.left:
                qune.append(node.left)
            if node.right:
                qune.append(node.right)
        res.append(temp)
    return res

