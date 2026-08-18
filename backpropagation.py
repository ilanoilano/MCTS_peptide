#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS 回溯模块 (Backpropagation)

功能：将 Simulation 得到的分数从叶节点传回根节点，更新路径上所有节点的统计信息。

核心逻辑：
- 路径上的每个节点都分享同一个分数（因为分数是整个路径的评估结果）
- 更新 visit_count 和 total_score
- 计算 average_score = total_score / visit_count
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class MCTSNode:
    """
    MCTS 树节点
    
    Attributes:
        sequence: 当前节点对应的部分序列（含 '_' 表示未确定位置）
        visit_count: 被访问次数
        total_score: 累积得分（所有访问的分数总和）
        children: 子节点字典 {氨基酸: 子节点}
        parent: 父节点引用（可选）
        is_terminal: 是否是叶节点（序列完整）
        position: 当前节点代表的序列位置（用于调试）
    """
    sequence: str
    visit_count: int = 0
    total_score: float = 0.0
    children: Dict[str, 'MCTSNode'] = field(default_factory=dict)
    parent: Optional['MCTSNode'] = None
    is_terminal: bool = False
    position: int = -1  # 当前节点在序列中的位置（-1 表示根节点）
    
    @property
    def average_score(self) -> float:
        """计算平均得分"""
        if self.visit_count == 0:
            return 0.0
        return self.total_score / self.visit_count
    
    def __repr__(self) -> str:
        return (f"MCTSNode(seq='{self.sequence}', visits={self.visit_count}, "
                f"avg_score={self.average_score:.4f}, terminal={self.is_terminal})")


class BackpropagationEngine:
    """MCTS 回溯引擎"""
    
    def __init__(self, verbose: bool = False):
        """
        初始化回溯引擎
        
        Args:
            verbose: 是否打印详细日志
        """
        self.verbose = verbose
    
    def backpropagate(self, path: List[MCTSNode], reward: float) -> None:
        """
        回溯更新路径上所有节点的统计信息
        
        Args:
            path: 从根节点到叶节点的完整路径 [根, 节点1, 节点2, ..., 叶节点]
            reward: Simulation 返回的归一化分数 [0, 1]
        
        注意：
        - 路径上的每个节点都分享同一个 reward
        - 因为 reward 是整个路径的评估结果
        """
        if not path:
            raise ValueError("路径不能为空")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Backpropagation: 更新 {len(path)} 个节点")
            print(f"Reward: {reward:.4f}")
            print(f"{'='*60}")
        
        for i, node in enumerate(path):
            # 更新节点统计
            node.visit_count += 1
            node.total_score += reward
            
            if self.verbose:
                print(f"  [{i}] {node.sequence}")
                print(f"      visits: {node.visit_count-1} -> {node.visit_count}")
                print(f"      total_score: {node.total_score-reward:.4f} -> {node.total_score:.4f}")
                print(f"      avg_score: {node.average_score:.4f}")
        
        if self.verbose:
            print(f"{'='*60}\n")
    
    def backpropagate_with_parent(self, leaf_node: MCTSNode, reward: float, 
                                   max_depth: int = 100) -> None:
        """
        使用 parent 指针回溯（替代方法）
        
        Args:
            leaf_node: 叶节点
            reward: Simulation 返回的归一化分数
            max_depth: 最大回溯深度（防止无限循环）
        
        注意：这个方法不需要传入完整路径，通过 parent 指针向上回溯
        """
        current = leaf_node
        depth = 0
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Backpropagation (via parent): reward={reward:.4f}")
            print(f"{'='*60}")
        
        while current is not None and depth < max_depth:
            # 更新当前节点
            current.visit_count += 1
            current.total_score += reward
            
            if self.verbose:
                print(f"  [{depth}] {current.sequence}")
                print(f"      visits: {current.visit_count}, avg: {current.average_score:.4f}")
            
            # 向上回溯
            current = current.parent
            depth += 1
        
        if self.verbose:
            print(f"{'='*60}\n")
    
    def get_node_stats(self, node: MCTSNode) -> Dict[str, Any]:
        """
        获取节点统计信息
        
        Args:
            node: MCTS 节点
        
        Returns:
            统计信息字典
        """
        return {
            'sequence': node.sequence,
            'visit_count': node.visit_count,
            'total_score': node.total_score,
            'average_score': node.average_score,
            'is_terminal': node.is_terminal,
            'num_children': len(node.children),
            'position': node.position
        }
    
    def print_tree_stats(self, root: MCTSNode, max_depth: int = 3, 
                         current_depth: int = 0) -> None:
        """
        打印树结构统计信息（用于调试）
        
        Args:
            root: 根节点
            max_depth: 最大打印深度
            current_depth: 当前深度（递归用）
        """
        if current_depth > max_depth:
            return
        
        indent = "  " * current_depth
        print(f"{indent}{root}")
        
        for aa, child in root.children.items():
            self.print_tree_stats(child, max_depth, current_depth + 1)


def create_root_node(template: str = None) -> MCTSNode:
    """
    创建 MCTS 根节点
    
    根节点代表空序列（所有可变位置为 '_'）
    
    Args:
        template: 序列模板（默认从 config 读取）
    
    Returns:
        根节点
    """
    if template is None:
        import config
        template = config.PEPTIDE_TEMPLATE
    
    # 将模板中的 'x' 替换为 '_'
    root_sequence = template.replace('x', '_').replace('X', '_')
    
    return MCTSNode(
        sequence=root_sequence,
        visit_count=0,
        total_score=0.0,
        children={},
        parent=None,
        is_terminal=False,
        position=-1
    )


def main():
    """测试回溯功能"""
    import config
    
    print("="*60)
    print("Backpropagation 模块测试")
    print("="*60)
    
    # 创建回溯引擎
    engine = BackpropagationEngine(verbose=True)
    
    # 创建根节点
    root = create_root_node()
    print(f"\n根节点: {root}")
    
    # 模拟创建一条路径：根 -> A -> C -> A
    node_a = MCTSNode(
        sequence="AC_____C______CG",
        parent=root,
        position=2
    )
    root.children['A'] = node_a
    
    node_c = MCTSNode(
        sequence="AC_____C______CG",
        parent=node_a,
        position=3
    )
    node_a.children['C'] = node_c
    
    node_a2 = MCTSNode(
        sequence="ACA____C______CG",
        parent=node_c,
        position=4
    )
    node_c.children['A'] = node_a2
    
    # 构建路径
    path = [root, node_a, node_c, node_a2]
    
    # 模拟分数
    reward = 0.75
    
    # 执行回溯
    print("\n执行回溯...")
    engine.backpropagate(path, reward)
    
    # 验证结果
    print("\n验证结果:")
    for i, node in enumerate(path):
        print(f"  节点 {i}: visits={node.visit_count}, avg={node.average_score:.4f}")
    
    # 测试使用 parent 指针回溯
    print("\n" + "="*60)
    print("测试使用 parent 指针回溯")
    print("="*60)
    
    # 创建新的叶节点
    leaf = MCTSNode(
        sequence="ACG____C______CG",
        parent=node_a2,
        position=5
    )
    node_a2.children['G'] = leaf
    
    reward2 = 0.85
    engine.backpropagate_with_parent(leaf, reward2)
    
    print("\n最终树结构:")
    engine.print_tree_stats(root, max_depth=3)
    
    print("\n" + "="*60)
    print("测试通过!")
    print("="*60)


if __name__ == "__main__":
    main()
