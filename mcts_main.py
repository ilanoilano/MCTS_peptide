#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS 主循环 - 完整的蒙特卡洛树搜索实现

将 Selection → Expansion → Simulation → Backpropagation 四个模块串联
"""

import sys
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

import config
from backpropagation import MCTSNode, BackpropagationEngine, create_root_node

# 导入其他模块
try:
    from selection import PUCTSelector
except ImportError:
    raise ImportError("selection.py 未找到，请确保文件存在")

try:
    from expansion import ExpansionEngine, PeptideState
except ImportError:
    raise ImportError("expansion.py 未找到，请确保文件存在")

# 导入 simulation 模块（直接函数调用）
import sim_1
import sim_2
import sim_3
import sim_4


@dataclass
class MCTSConfig:
    """MCTS 配置"""
    max_iterations: int = 1000
    exploration_constant: float = 1.414  # sqrt(2)
    target_name: str = "1LYZ"
    verbose: bool = True
    save_interval: int = 100  # 每多少次迭代保存一次


class MCTSEngine:
    """MCTS 引擎 - 主循环"""
    
    def __init__(self, mcts_config: MCTSConfig = None):
        """
        初始化 MCTS 引擎
        
        Args:
            mcts_config: MCTS 配置
        """
        self.config = mcts_config or MCTSConfig()
        
        # 初始化四个模块
        self.backprop_engine = BackpropagationEngine(verbose=self.config.verbose)
        
        # 选择器
        self.selector = PUCTSelector(c_puct=self.config.exploration_constant)
        
        # 扩展器
        self.expansion_engine = ExpansionEngine(
            template=config.PEPTIDE_TEMPLATE,
            fixed_positions=config.FIXED_POSITIONS,
            variable_amino_acids=config.VARIABLE_AA_OPTIONS,
        )
        
        # 创建根节点
        self.root = create_root_node()
        
        # 统计
        self.iteration_count = 0
        self.best_score = float('-inf')
        self.best_sequence = None
    
    def select(self, node: MCTSNode) -> List[MCTSNode]:
        """
        第 1 步：选择（Selection）
        
        从根节点开始，用 PUCT 公式递归选择子节点，走到一个叶节点
        
        Args:
            node: 起始节点（通常是根节点）
        
        Returns:
            路径 [根, 节点1, ..., 叶节点]
        """
        path = [node]
        current = node
        
        # 一直走到叶节点（没有子节点或子节点未完全扩展）
        while current.children and len(current.children) > 0:
            # 使用 PUCT 选择最佳子节点
            # 获取允许的氨基酸
            next_pos = self._get_next_position(current.sequence)
            if next_pos is not None:
                allowed_aas = config.VARIABLE_AA_OPTIONS.get(next_pos, config.ALLOWED_AMINO_ACIDS)
                # 过滤掉已尝试的
                available_aas = [aa for aa in allowed_aas if aa not in current.children]
                if available_aas:
                    # 返回 (氨基酸, 子节点)，子节点为 None 表示需要创建
                    selected_aa, best_child = self.selector.select(current, available_aas)
                    if best_child is None and selected_aa:
                        # 需要创建新节点
                        best_child = self._create_child(current, selected_aa, next_pos)
                elif current.children:
                    # 所有氨基酸都已尝试，选择最佳子节点
                    best_child = max(current.children.values(), key=lambda n: n.average_score)
                else:
                    best_child = None
            else:
                best_child = None
            
            if best_child is None:
                break
            
            path.append(best_child)
            current = best_child
            
            # 检查是否是终止节点
            if current.is_terminal:
                break
        
        return path
    
    def expand(self, node: MCTSNode) -> MCTSNode:
        """
        第 2 步：扩展（Expansion）
        
        如果叶节点不是终止节点，选一个未尝试的氨基酸，创建一个新子节点
        
        Args:
            node: 叶节点
        
        Returns:
            新子节点（或原节点如果已是终止节点）
        """
        # 检查是否已终止
        if node.is_terminal:
            return node
        
        # 找到下一个可变位置
        next_pos = self._get_next_position(node.sequence)
        if next_pos is None:
            # 序列已完成
            node.is_terminal = True
            return node
        
        # 获取允许的氨基酸
        allowed_aas = config.VARIABLE_AA_OPTIONS.get(next_pos, config.ALLOWED_AMINO_ACIDS)
        
        # 过滤掉已尝试的
        untried_aas = [aa for aa in allowed_aas if aa not in node.children]
        
        if not untried_aas:
            # 所有氨基酸都已尝试
            if node.children:
                # 选择访问次数最多的子节点
                return max(node.children.values(), key=lambda n: n.visit_count)
            else:
                node.is_terminal = True
                return node
        
        # 随机选择氨基酸（后续可以加入先验概率）
        chosen_aa = random.choice(untried_aas)
        
        # 创建新序列
        new_sequence = self._fill_position(node.sequence, next_pos, chosen_aa)
        
        # 创建新节点
        new_node = MCTSNode(
            sequence=new_sequence,
            parent=node,
            position=next_pos
        )
        
        # 检查是否完成
        if '_' not in new_sequence and 'x' not in new_sequence:
            new_node.is_terminal = True
        
        # 添加到父节点
        node.children[chosen_aa] = new_node
        
        return new_node
    
    def simulate(self, node: MCTSNode) -> float:
        """
        第 3 步：模拟（Simulation）
        
        调用 sim_1 → sim_2 → sim_3 → sim_4 完成完整流程：
        1. sim_1: 补全序列
        2. sim_2: 生成 3D 构象（PDB）
        3. sim_3: 转换为 PDBQT
        4. sim_4: Vina 对接打分
        
        使用直接函数调用（非子进程）
        
        Args:
            node: 当前节点
        
        Returns:
            归一化分数 [0, 1]
        """
        partial_sequence = node.sequence
        
        try:
            # Step 1: sim_1 - 补全序列（填充所有未确定位置）
            full_sequence = sim_1.fill_placeholder_positions(
                partial_sequence,
                fill_all=True
            )
            
            # Step 2: sim_2 - 生成 3D 构象
            pdb_path = sim_2.generate_conformation(
                sequence=full_sequence,
                verbose=False
            )
            
            # Step 3: sim_3 - 转换为 PDBQT
            pdbqt_path = sim_3.pdb_to_pdbqt(
                pdb_path=pdb_path,
                verbose=False
            )
            
            # Step 4: sim_4 - Vina 对接
            vina_output = sim_4.run_vina_docking(
                ligand_pdbqt=pdbqt_path,
                target_name=self.config.target_name,
                exhaustiveness=1,
                num_modes=1,
                verbose=False
            )
            
            # 获取最佳结合能
            best_result = vina_output.get_best_result()
            if best_result is None:
                raise RuntimeError("Vina 对接没有返回结果")
            binding_energy = best_result.binding_energy
            
            # 归一化到 [0, 1]（Vina 结合能通常为 -15 到 0）
            # -15 -> 1.0 (最好), 0 -> 0.0 (最差)
            score = max(0.0, min(1.0, (-binding_energy) / 15.0))
            
            # 更新最佳
            if score > self.best_score:
                self.best_score = score
                self.best_sequence = full_sequence
            
            return score
            
        except Exception as e:
            if self.config.verbose:
                print(f"模拟失败: {e}")
            return 0.0
    
    def backpropagate(self, path: List[MCTSNode], reward: float) -> None:
        """
        第 4 步：回溯（Backpropagation）
        
        把分数沿路径回传，更新路径上所有节点的统计信息
        
        Args:
            path: 路径 [根, 节点1, ..., 叶节点]
            reward: 分数 [0, 1]
        """
        self.backprop_engine.backpropagate(path, reward)
    
    def run_iteration(self) -> Tuple[List[MCTSNode], float]:
        """
        运行一次 MCTS 迭代
        
        Returns:
            (路径, 分数)
        """
        # 第 1 步：选择
        path = self.select(self.root)
        leaf = path[-1]
        
        # 第 2 步：扩展
        new_node = self.expand(leaf)
        if new_node != leaf:
            path.append(new_node)
        
        # 第 3 步：模拟
        reward = self.simulate(path[-1])
        
        # 第 4 步：回溯
        self.backpropagate(path, reward)
        
        self.iteration_count += 1
        
        return path, reward
    
    def search(self, num_iterations: int = None) -> Dict:
        """
        运行 MCTS 搜索
        
        Args:
            num_iterations: 迭代次数（默认使用配置）
        
        Returns:
            搜索结果
        """
        if num_iterations is None:
            num_iterations = self.config.max_iterations
        
        print(f"\n{'='*60}")
        print(f"MCTS 搜索开始")
        print(f"{'='*60}")
        print(f"迭代次数: {num_iterations}")
        print(f"探索常数: {self.config.exploration_constant}")
        print(f"靶点: {self.config.target_name}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for i in range(num_iterations):
            path, reward = self.run_iteration()
            
            # 打印进度
            if self.config.verbose and (i + 1) % self.config.save_interval == 0:
                elapsed = time.time() - start_time
                print(f"迭代 {i+1}/{num_iterations}, "
                      f"得分: {reward:.4f}, "
                      f"最佳: {self.best_score:.4f}, "
                      f"时间: {elapsed:.1f}s")
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"MCTS 搜索完成")
        print(f"{'='*60}")
        print(f"总迭代: {self.iteration_count}")
        print(f"总时间: {elapsed:.1f}s")
        print(f"平均每次: {elapsed/self.iteration_count:.3f}s")
        print(f"最佳得分: {self.best_score:.4f}")
        print(f"最佳序列: {self.best_sequence}")
        print(f"{'='*60}\n")
        
        # 提取最优路径
        best_path = self._extract_best_path()
        
        return {
            'best_sequence': self.best_sequence,
            'best_score': self.best_score,
            'iterations': self.iteration_count,
            'time': elapsed,
            'best_path': best_path,
            'root': self.root
        }
    
    def _extract_best_path(self) -> List[str]:
        """提取最优路径（访问次数最多或平均分最高）"""
        path = []
        current = self.root
        
        while current.children:
            # 选择访问次数最多的子节点
            best_child = max(current.children.values(), 
                           key=lambda n: n.visit_count)
            path.append(best_child.sequence)
            current = best_child
            
            if current.is_terminal:
                break
        
        return path
    
    def _create_child(self, parent: MCTSNode, amino_acid: str, position: int) -> MCTSNode:
        """创建新子节点"""
        new_sequence = self._fill_position(parent.sequence, position, amino_acid)
        
        child = MCTSNode(
            sequence=new_sequence,
            parent=parent,
            position=position
        )
        
        # 检查是否完成
        if '_' not in new_sequence and 'x' not in new_sequence:
            child.is_terminal = True
        
        parent.children[amino_acid] = child
        return child
    
    def _get_next_position(self, sequence: str) -> Optional[int]:
        """获取下一个可变位置"""
        for i, aa in enumerate(sequence):
            if aa == '_' or aa == 'x':
                return i
        return None
    
    def _fill_position(self, sequence: str, position: int, amino_acid: str) -> str:
        """在指定位置填充氨基酸"""
        seq_list = list(sequence)
        seq_list[position] = amino_acid
        return ''.join(seq_list)
    
    def _fill_placeholder_positions(self, sequence: str) -> str:
        """填充所有占位符（随机）"""
        seq_list = list(sequence)
        for i, aa in enumerate(seq_list):
            if aa == '_' or aa == 'x':
                allowed = config.VARIABLE_AA_OPTIONS.get(i, config.ALLOWED_AMINO_ACIDS)
                seq_list[i] = random.choice(allowed)
        return ''.join(seq_list)
    
    def _validate_sequence(self, sequence: str) -> bool:
        """验证序列有效性"""
        # 检查长度
        if len(sequence) != len(config.PEPTIDE_TEMPLATE):
            return False
        
        # 检查固定位置
        for pos, expected_aa in config.FIXED_POSITIONS.items():
            if sequence[pos] != expected_aa:
                return False
        
        # 检查有效氨基酸
        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        for aa in sequence:
            if aa not in valid_aas:
                return False
        
        return True
    



def main():
    """测试 MCTS 主循环"""
    print("="*60)
    print("MCTS 主循环测试")
    print("="*60)
    
    # 创建配置
    mcts_config = MCTSConfig(
        max_iterations=100,
        exploration_constant=1.414,
        target_name="1LYZ",
        verbose=True,
        save_interval=10
    )
    
    # 创建 MCTS 引擎
    engine = MCTSEngine(mcts_config)
    
    # 运行搜索
    result = engine.search()
    
    print("\n最终结果:")
    print(f"  最佳序列: {result['best_sequence']}")
    print(f"  最佳得分: {result['best_score']:.4f}")
    print(f"  迭代次数: {result['iterations']}")
    print(f"  总时间: {result['time']:.1f}s")


if __name__ == "__main__":
    main()
