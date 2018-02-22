# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        if root is None:
            return None
        if root.left is None and root.right is None:
            return None
        _out = list()
        _out.append(root)
        for node in _out:
            if node.left is not None:
                _out.append(node.left)
            if node.right is not None:
                _out.append(node.right)
        return _out