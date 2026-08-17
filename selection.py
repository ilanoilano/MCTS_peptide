#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS Selection 模块 - PUCT 选择器
"""

import math
import random
from typing import List, Dict, Optional

from backpropagation import MCTSNode


class PUCTSelector:
    """PUCT 选择器"""
    
    def __init__(self, c_puct: float = 1.414):
        """
        初始化 PUCT 选择器
        
        Args:
            c_puct: 探索常数（默认 sqrt(2) ≈ 1.414）
        """
        self.c_puct = c_puct
    
    def select(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        使用 PUCT 公式选择最佳子节点
        
        PUCT = Q + U
        Q = average_score ( exploitation )
        U = c_puct * sqrt(parent_visits) / (1 + child_visits) ( exploration )
        
        Args:
            node: 父节点
        
        Returns:
            最佳子节点，或 None（如果没有子节点）
        """
        if not node.children:
            return None
        
        best_score = float('-inf')
        best_child = None
        
        parent_visits = node.visit_count
        
        for amino_acid, child in node.children.items():
            # Q 值：平均得分（exploitation）
            if child.visit_count == 0:
                q_value = 0.0
            else:
                q_value = child.average_score
            
            # U 值：探索奖励（exploration）
            if parent_visits == 0:
                u_value = float('inf')
            else:
                u_value = self.c_puct * math.sqrt(parent_visits) / (1 + child.visit_count)
            
            # PUCT 分数
            puct_score = q_value + u_value
            
            if puct_score > best_score:
                best_score = puct_score
                best_child = child
        
        return best_child
    
    def select_with_random(self, node: MCTSNode, epsilon: float = 0.1) -> Optional[MCTSNode]:
        """
        带随机探索的 PUCT 选择
        
        以 epsilon 概率随机选择，以 1-epsilon 概率使用 PUCT
        
        Args:
            node: 父节点
            epsilon: 随机探索概率
        
        Returns:
            选择的子节点
        """
        if not node.children:
            return None
        
        # epsilon-贪婪策略
        if random.random() < epsilon:
            return random.choice(list(node.children.values()))
        else:
            return self.select(node)
