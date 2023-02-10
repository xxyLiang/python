

class TreeNode:
    def __init__(self, val, w, l=None, r=None) -> None:
        self.val = val
        self.weight = w
        self.left = l
        self.right = r


def origin_text_alpha_freq(filename: str) -> dict:
    alpha_freq_dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for i in f.read():
            if i in alpha_freq_dict:
                alpha_freq_dict[i] += 1
            else:
                alpha_freq_dict[i] = 1
    return alpha_freq_dict


def build_huffman_tree(alpha_freq_dict: dict) -> TreeNode:
    nodes = []
    for i in alpha_freq_dict:
        nodes.append(TreeNode(i, alpha_freq_dict[i]))
    nodes.sort(key=lambda x: (x.weight, x.val))
    
    while len(nodes)>1:
        lNode = nodes.pop(0)
        rNode = nodes.pop(0)
        root = TreeNode('&&', lNode.weight + rNode.weight, lNode, rNode)
        if len(nodes) == 0:
            return root

        inserted = False
        for i in range(len(nodes)):     # 将新组合的root插入到node数组中
            if (root.weight, root.val) <= (nodes[i].weight, nodes[i].val):
                nodes.insert(i, root)
                inserted = True
                break
        if not inserted:
            nodes.append(root)  


def traverse_huffman_tree(root: TreeNode) -> dict:

    def traverse(root: TreeNode, prefix: str, alpha_code_dict: dict):
        if root:
            if root.val != '&&':
                alpha_code_dict[root.val] = prefix
            else:
                traverse(root.left, prefix+'0', alpha_code_dict)
                traverse(root.right, prefix+'1', alpha_code_dict)

    alpha_code_dict = {}
    traverse(root, '', alpha_code_dict)
    return alpha_code_dict


def translate_text2code(textFileName: str, codeFileName: str) -> None:
    alpha_freq_dict = origin_text_alpha_freq(textFileName)
    huffmanTree_root = build_huffman_tree(alpha_freq_dict)
    alpha_code_dict = traverse_huffman_tree(huffmanTree_root)

    code = ''
    with open(textFileName, 'r', encoding='utf-8') as rf:
        for i in rf.read():
            code += alpha_code_dict[i]

    with open(codeFileName, 'w', encoding='utf-8') as wf:
        wf.write(code)
        wf.write('\n')
        for i in alpha_freq_dict:
            wf.write('%s%d\t' % (i, alpha_freq_dict[i]))
        
    print("ok")


def translate_code2text(codeFileName: str, textFileName: str) -> None:
    with open(codeFileName, 'r', encoding='utf-8') as rf:
        code = rf.readline().strip('\n')
        alpha_freq_dict = {}
        alpha_info = ''
        for i in rf.readlines():
            alpha_info += i
        for i in alpha_info.split('\t')[:-1]:
            alpha_freq_dict[i[0]] = int(i[1:])
    
    huffmanTree_root = build_huffman_tree(alpha_freq_dict)
    txt = ''

    p = huffmanTree_root
    for i in code:
        p = p.left if i == '0' else p.right
        if p.val != '&&':
            txt += p.val
            p = huffmanTree_root
    
    with open(textFileName, 'w', encoding='utf-8') as wf:
        wf.write(txt)
    
    print('ok')

        

translate_text2code('originalText.txt', '2.txt')
translate_code2text('2.txt', 'out.txt')
