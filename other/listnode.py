class ListNode:
    def __init__(self, val) -> None:
        self.val = val
        self.next = None


def k_reverse(head, k):
    # k个一组反转链表
    node = head

    for i in range(k):
        if node is None:
            return head
        node = node.next
    
    res = reverse(head, node)
    head.next = k_reverse(node, k)      # 翻转后，head实际成为组内的最后一个结点

    return res
     

def reverse(left, right):   # 反转left~right-1链表，返回链头
    prev = right
    while(left!=right):
        temp = left.next
        left.next = prev
        prev = left
        left = temp
    return prev


def hasCycle_1(head: ListNode) -> bool:
    # 判断链表中是否有环，有true无false
    visited = []    # 使用数组记录已遍历的结点
    while(head):
        if head in visited:
            return True
        visited.append(head)
        head = head.next
    return False

def hasCycle_2(head: ListNode) -> bool:
    # 使用快慢指针，如果存在环，在slow遍历完链表前，fast一定会套圈追上slow
    slow = head
    fast = head
    while fast and fast.next:       # 只需要判断fast是否到达链尾，slow不可能比fast还快
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

def reverseBetween(head: ListNode, m: int, n: int) -> ListNode:
    # 将一个节点数为 size 链表 m 位置到 n 位置之间的区间反转
    
    # l,m 分别是区间的前一个和后一个，即要反转l+1~r-1区间
    l = r = None
    node = head
    for i in range(n):
        if i==m-2:
            l = node
        node = node.next
    r = node
    
    node = l.next if l else head    # 如果m为1则l为None
    prev = r
    while(node!=r):
        temp = node.next
        node.next = prev
        prev = node 
        node = temp
    if l:   # m>1
        l.next = prev
        return head
    else:   # m==1
        return prev


def printnode(head):
    res = []
    while(head):
        res.append(head.val)
        head = head.next
    print(res)


nodes = [ListNode(i) for i in range(1, 12)]
for i in range(len(nodes)-1):
    nodes[i].next = nodes[i+1]

head = nodes[0]

printnode(k_reverse(head, 3))
