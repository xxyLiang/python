class ListNode:
    def __init__(self, val) -> None:
        self.val = val
        self.next = None


def k_reverse(head, k):
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
