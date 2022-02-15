class Node:
    def __init__(self, val) -> None:
        self.val = val
        self.next = None


def k_reverse(headnode, k):
    node = headnode

    for i in range(k):
        if node is None:
            return headnode
        node = node.next
    
    res = reverse(headnode, node)
    headnode.next = k_reverse(node, k)
    return res
     

def reverse(left, right):
    prev = right
    while(left!=right):
        temp = left.next
        left.next = prev
        prev = left
        left = temp
    return prev


def printnode(head):
    while(head):
        print(head.val)
        head = head.next


nodes = [Node(i) for i in range(1, 12)]
for i in range(len(nodes)-1):
    nodes[i].next = nodes[i+1]

printnode(k_reverse(nodes[0], 3))
