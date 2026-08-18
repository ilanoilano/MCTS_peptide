#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: MCTS-Directed 环肽设计 - 主运行脚本

功能：执行完整的蒙特卡洛树搜索，优化环肽序列
输入：Phase 1 准备的受体、口袋、Vina 配置
输出：最优环肽序列及 Top-N 候选
"""

import sys
import json
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import config
from backpropagation import MCTSNode, BackpropagationEngine, create_root_node
from selection import PUCTSelector

# 导入 simulation 模块
import sim_1
import sim_2
import sim_3
import sim_4


@dataclass
class Phase2Config:
    """Phase 2 配置"""
    target_name: str = "1LYZ"
    max_iterations: int = 1000
    exploration_constant: float = 1.414
    top_n_candidates: int = 10
    
    # Vina 参数
    vina_exhaustiveness: int = 1
    vina_num_modes: int = 1
    vina_timeout: int = 300
    
    # 输出控制
    save_interval: int = 100
    verbose: bool = True
    output_dir: Path = None
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = Path(f"/mnt/d/code/AA/results/{self.target_name}/phase2")


class MCTSEngine:
    """MCTS 引擎 - 完整的四步循环"""
    
    def __init__(self, phase2_config: Phase2Config):
        self.config = phase2_config
        
        # 初始化模块
        self.backprop_engine = BackpropagationEngine(verbose=self.config.verbose)
        self.selector = PUCTSelector(c_puct=self.config.exploration_constant)
        
        # 创建根节点
        self.root = create_root_node()
        
        # 统计
        self.iteration_count = 0
        self.best_score = float('-inf')
        self.best_sequence = None
        self.start_time = None
        
        # 确保输出目录存在
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def select(self, node: MCTSNode) -> List[MCTSNode]:
        """
        第 1 步：选择（Selection）
        从根节点开始，用 PUCT 公式递归选择子节点，走到叶节点
        """
        path = [node]
        current = node
        
        while current.children:
            # 找到下一个可变位置
            next_pos = self._get_next_position(current.sequence)
            if next_pos is None:
                # 序列已完成
                current.is_terminal = True
                break
            
            # 获取允许的氨基酸
            allowed_aas = config.VARIABLE_AA_OPTIONS.get(next_pos, config.ALLOWED_AMINO_ACIDS)
            
            # 过滤掉已尝试的
            available_aas = [aa for aa in allowed_aas if aa not in current.children]
            
            if available_aas:
                # 从未尝试的氨基酸中随机选择
                selected_aa = random.choice(available_aas)
                # 创建新节点
                best_child = self._create_child(current, selected_aa, next_pos)
                path.append(best_child)
                break
            else:
                # 所有氨基酸都已尝试，使用 PUCT 选择最佳子节点
                best_child = self.selector.select(current)
                if best_child is None:
                    break
                path.append(best_child)
                current = best_child
            
            if current.is_terminal:
                break
        
        return path
    
    def expand(self, node: MCTSNode) -> MCTSNode:
        """
        第 2 步：扩展（Expansion）
        如果叶节点不是终止节点，选一个未尝试的氨基酸，创建一个新子节点
        """
        if node.is_terminal:
            return node
        
        # 找到下一个可变位置
        next_pos = self._get_next_position(node.sequence)
        if next_pos is None:
            node.is_terminal = True
            return node
        
        # 获取允许的氨基酸
        allowed_aas = config.VARIABLE_AA_OPTIONS.get(next_pos, config.ALLOWED_AMINO_ACIDS)
        
        # 过滤掉已尝试的
        untried_aas = [aa for aa in allowed_aas if aa not in node.children]
        
        if not untried_aas:
            # 所有氨基酸都已尝试
            if node.children:
                return max(node.children.values(), key=lambda n: n.visit_count)
            else:
                node.is_terminal = True
                return node
        
        # 随机选择氨基酸
        chosen_aa = random.choice(untried_aas)
        
        # 创建新节点
        new_node = self._create_child(node, chosen_aa, next_pos)
        
        return new_node
    
    def simulate(self, node: MCTSNode) -> float:
        """
        第 3 步：模拟（Simulation）
        调用 sim_1 → sim_2 → sim_3 → sim_4 完成完整流程
        """
        partial_sequence = node.sequence
        
        try:
            # Step 3.1: sim_1 - 补全序列
            full_sequence = sim_1.fill_placeholder_positions(
                partial_sequence,
                fill_all=True
            )
            
            # Step 3.2: sim_2 - 生成 3D 构象
            pdb_path = sim_2.generate_conformation(
                sequence=full_sequence,
                verbose=False
            )
            
            # Step 3.3: sim_3 - 转换为 PDBQT
            pdbqt_path = sim_3.pdb_to_pdbqt(
                pdb_path=pdb_path,
                verbose=False
            )
            
            # Step 3.4: sim_4 - Vina 对接
            vina_output = sim_4.run_vina_docking(
                ligand_pdbqt=pdbqt_path,
                target_name=self.config.target_name,
                exhaustiveness=self.config.vina_exhaustiveness,
                num_modes=self.config.vina_num_modes,
                timeout=self.config.vina_timeout,
                verbose=False
            )
            
            # Step 3.5: 获取最佳结合能并归一化
            best_result = vina_output.get_best()
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
                if self.config.verbose:
                    print(f"  [New Best] {full_sequence}: {binding_energy:.2f} kcal/mol (score={score:.4f})")
            
            return score
            
        except Exception as e:
            if self.config.verbose:
                print(f"  [Simulation Failed] {e}")
            return 0.0
    
    def backpropagate(self, path: List[MCTSNode], reward: float) -> None:
        """
        第 4 步：回溯（Backpropagation）
        把分数沿路径回传，更新路径上所有节点的统计信息
        """
        self.backprop_engine.backpropagate(path, reward)
    
    def run_iteration(self) -> Tuple[List[MCTSNode], float]:
        """运行一次 MCTS 迭代"""
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
    
    def search(self) -> Dict:
        """运行 MCTS 搜索"""
        print(f"\n{'='*70}")
        print(f"Phase 2: MCTS-Directed 环肽设计")
        print(f"{'='*70}")
        print(f"靶点: {self.config.target_name}")
        print(f"肽模板: {config.PEPTIDE_TEMPLATE}")
        print(f"交联剂: {config.CROSSLINKER}")
        print(f"迭代次数: {self.config.max_iterations}")
        print(f"探索常数: {self.config.exploration_constant}")
        print(f"Vina exhaustiveness: {self.config.vina_exhaustiveness}")
        print(f"{'='*70}\n")
        
        self.start_time = time.time()
        
        for i in range(self.config.max_iterations):
            path, reward = self.run_iteration()
            
            # 打印进度
            if self.config.verbose and (i + 1) % self.config.save_interval == 0:
                elapsed = time.time() - self.start_time
                iter_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"Iter {i+1}/{self.config.max_iterations} | "
                      f"Reward: {reward:.4f} | "
                      f"Best: {self.best_score:.4f} | "
                      f"Speed: {iter_per_sec:.2f} iter/s")
            
            # 定期保存检查点
            if (i + 1) % (self.config.save_interval * 5) == 0:
                self._save_checkpoint()
        
        elapsed = time.time() - self.start_time
        
        print(f"\n{'='*70}")
        print(f"MCTS 搜索完成")
        print(f"{'='*70}")
        print(f"总迭代: {self.iteration_count}")
        print(f"总时间: {elapsed:.1f}s")
        print(f"平均速度: {self.iteration_count/elapsed:.2f} iter/s")
        print(f"最佳得分: {self.best_score:.4f}")
        print(f"最佳序列: {self.best_sequence}")
        print(f"{'='*70}\n")
        
        # 提取候选
        candidates = self._extract_candidates()
        
        return {
            'best_sequence': self.best_sequence,
            'best_score': self.best_score,
            'iterations': self.iteration_count,
            'time': elapsed,
            'candidates': candidates,
            'root': self.root
        }
    
    def _extract_candidates(self) -> List[Dict]:
        """提取 Top-N 候选序列"""
        candidates = []
        
        # 收集所有终止节点
        terminal_nodes = []
        
        def collect_terminal(node):
            if node.is_terminal:
                terminal_nodes.append(node)
            for child in node.children.values():
                collect_terminal(child)
        
        collect_terminal(self.root)
        
        # 按平均分数排序
        terminal_nodes.sort(key=lambda n: n.average_score, reverse=True)
        
        # 取 Top-N
        for i, node in enumerate(terminal_nodes[:self.config.top_n_candidates]):
            candidates.append({
                'rank': i + 1,
                'sequence': node.sequence,
                'average_score': node.average_score,
                'visit_count': node.visit_count,
                'total_score': node.total_score
            })
        
        return candidates
    
    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint_file = self.config.output_dir / f"checkpoint_{self.iteration_count}.json"
        checkpoint_data = {
            'iteration': self.iteration_count,
            'best_sequence': self.best_sequence,
            'best_score': self.best_score,
            'timestamp': datetime.now().isoformat()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
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


def save_results(result: Dict, output_dir: Path, target_name: str):
    """保存搜索结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存 JSON 结果
    result_file = output_dir / f"phase2_result_{target_name}_{timestamp}.json"
    
    # 转换不可序列化的对象
    serializable_result = {
        'best_sequence': result['best_sequence'],
        'best_score': result['best_score'],
        'iterations': result['iterations'],
        'time': result['time'],
        'candidates': result['candidates']
    }
    
    with open(result_file, 'w') as f:
        json.dump(serializable_result, f, indent=2)
    
    print(f"结果已保存: {result_file}")
    
    # 保存候选序列为文本文件
    candidates_file = output_dir / f"candidates_{target_name}_{timestamp}.txt"
    with open(candidates_file, 'w') as f:
        f.write(f"# Phase 2 Results for {target_name}\n")
        f.write(f"# Generated: {timestamp}\n")
        f.write(f"# Best Sequence: {result['best_sequence']}\n")
        f.write(f"# Best Score: {result['best_score']:.4f}\n")
        f.write("#\n")
        f.write("# Rank | Sequence | Avg Score | Visits\n")
        f.write("-" * 70 + "\n")
        
        for cand in result['candidates']:
            f.write(f"{cand['rank']:4d} | {cand['sequence']} | "
                   f"{cand['average_score']:9.4f} | {cand['visit_count']:6d}\n")
    
    print(f"候选列表已保存: {candidates_file}")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='Phase 2: MCTS-Directed 环肽设计')
    parser.add_argument('-t', '--target', type=str, default='1LYZ',
                        help='靶点名称 (默认: 1LYZ)')
    parser.add_argument('-n', '--iterations', type=int, default=1000,
                        help='MCTS 迭代次数 (默认: 1000)')
    parser.add_argument('-c', '--exploration', type=float, default=1.414,
                        help='PUCT 探索常数 (默认: 1.414)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='提取 Top-N 候选 (默认: 10)')
    parser.add_argument('-e', '--exhaustiveness', type=int, default=1,
                        help='Vina exhaustiveness (默认: 1)')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Vina 超时时间 (默认: 300s)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='静默模式')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='保存检查点间隔 (默认: 100)')
    
    args = parser.parse_args()
    
    # 创建配置
    config = Phase2Config(
        target_name=args.target,
        max_iterations=args.iterations,
        exploration_constant=args.exploration,
        top_n_candidates=args.top_n,
        vina_exhaustiveness=args.exhaustiveness,
        vina_timeout=args.timeout,
        verbose=not args.quiet,
        save_interval=args.save_interval
    )
    
    # 运行 MCTS
    engine = MCTSEngine(config)
    result = engine.search()
    
    # 保存结果
    save_results(result, config.output_dir, config.target_name)
    
    # 打印候选列表
    print("\n" + "="*70)
    print("Top Candidates:")
    print("="*70)
    for cand in result['candidates'][:5]:
        print(f"  {cand['rank']}. {cand['sequence']} "
              f"(score={cand['average_score']:.4f}, visits={cand['visit_count']})")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
