#-*- coding:utf-8 -*-
import copy

# -*- coding:utf-8 -*-
class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        if pHead is None: return None
        root = copy.deepcopy(pHead)
        return root


