from typing import List

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def levelOrder(root: TreeNode) -> List[List[int]]:
    '''
        按层遍历二叉树，返回[ [第一层], [第二层], ...]
    '''
    que, res = [], []
    if not root:    # 根节点为空直接返回
        return res
    que.append(root)   # 队列保存每一层所有未访问的结点
    while que:
        nodes_num_of_level = len(que)       # 如果不需要每层生成一个一维数组的话，则不需记录当前层的结点数
        temp = []
        
        for i in range(nodes_num_of_level): # 遍历每一层
            node = que.pop(0)
            temp.append(node.val)
            # 左右节点按顺序加到队尾
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)

        res.append(temp)
    return res


def maxDepth(root: TreeNode) -> int:
    '''
        返回二叉树的最大深度，和上面思想一样
    '''
    depth = 0
    if not root:
        return depth
    que = []
    que.append(root)
    while que:
        nodes_num_of_level = len(que)
        for i in range(nodes_num_of_level):
            node = que.pop(0)
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)
        depth += 1
    return depth


def preOrderTraversal(root: TreeNode) -> List[int]:
    '''
        非递归方法的前序遍历
        思路：用栈，先将根节点入栈，出栈的同时，先将右子树入栈，再将左子树入栈，循环执行即可
    '''
    res = []
    stack = []
    if not root:
        return res
    stack.append(root)
    while stack:
        p = stack.pop(-1)
        res.append(p.val)
        if(p.right):
            stack.append(p.right)
        if(p.left):
            stack.append(p.left)
    return res


def postOrderTraversal(root: TreeNode) -> List[int]:
    '''
    思路：
        后序遍历的顺序为左、右、根，反过来就是根、右、左，类似先序遍历的根、左、右，
        因此采用与先序遍历的思路：先将根节点入栈，出栈的同时，先将左子树入栈，再将右子树入栈（从而出栈时先右后左）
        最后输出前将数组反过来即可
    '''
    stack = []
    res = []
    if not root:
        return res
    stack.append(root)
    while stack:
        p = stack.pop(-1)
        res.append(p.val)
        if(p.left):
            stack.append(p.left)
        if(p.right):
            stack.append(p.right)
    res.reverse()
    return res


def inOrderTraversal(root: TreeNode) -> List[int]:
    '''
        非递归方法的中序遍历
        思路：
            1.先一直将左子树加入栈中，直到左结点为空。
            2.pop栈中最后一个结点，访问
            3.对它的右子树重新递归，回到步骤1
    '''
    stack = []      # 栈
    res = []
    p = root
    while True:
        while p:           # 首先将所有左结点加入到栈中
            stack.append(p)
            p = p.left
        if not stack:
            break
        p = stack.pop(-1)   # 取最后一个结点，由于上面能退出while p循环，此结点必定没有左孩子，或已遍历完左子树
        res.append(p.val)   # 取数
        p = p.right         # 遍历右子树，相当于递归了
    return res


def traversal_recursion(root: TreeNode) -> List[int]:
    '''
        递归方法的前序/中序/后序遍历，有堆栈溢出（递归过深）的风险
    '''
    def preorder(root, res):
        if root:
            res.append(root.val)
            preorder(root.left, res)
            preorder(root.right, res)

    def inorder(root, res):
        if root:
            inorder(root.left, res)
            res.append(root.val)
            inorder(root.right, res)

    def postorder(root, res):
        if root:
            postorder(root.left, res)
            postorder(root.right, res)
            res.append(root.val)
    
    res = []
    inorder(root, res)
    return res