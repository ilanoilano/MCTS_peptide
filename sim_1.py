#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sim_1: 序列补全模块
功能：将部分序列（含 x/_）补全为完整序列
输入：部分序列（如 ACxxxxxxCxxxxxxCG）
输出：完整序列（如 ACAAAAAACAAAAAACG）
"""

import random
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))

import config


def fill_placeholder_positions(sequence: str, 
                                mcts_depth: int = 0,
                                fill_all: bool = False,
                                fixed_positions: dict = None, 
                                variable_amino_acids: dict = None) -> str:
    """
    填充序列中的占位符（_ 或 x）为有效氨基酸
    
    关键特性：MCTS 深度指的是【第几个未知氨基酸】，不是序列位置
    - 第 0 到 mcts_depth-1 个未知氨基酸：已确定（保持当前值）
    - 第 mcts_depth 个未知氨基酸：正在搜索（保持占位符）
    - 第 mcts_depth+1 个及之后的未知氨基酸：随机填充
    
    Args:
        sequence: 部分填充的序列（可能含 _ 或 x）
        mcts_depth: MCTS 当前搜索深度（0-based），指第几个未知氨基酸
        fixed_positions: 固定位置映射（默认使用 config.FIXED_POSITIONS）
        variable_amino_acids: 可变位置允许的氨基酸（默认使用 config.VARIABLE_AMINO_ACIDS）
    
    Returns:
        完整序列（无占位符）
    """
    if fixed_positions is None:
        fixed_positions = config.FIXED_POSITIONS
    if variable_amino_acids is None:
        variable_amino_acids = config.VARIABLE_AMINO_ACIDS
    
    seq_list = list(sequence)
    
    # 确保固定位置正确
    for pos, aa in fixed_positions.items():
        if pos < len(seq_list):
            seq_list[pos] = aa
    
    # 收集所有可变位置（按顺序）
    variable_positions = [i for i, aa in enumerate(seq_list) 
                         if aa == '_' or aa == 'x' or aa == 'X']
    
    # 按 MCTS 深度处理可变位置
    for var_idx, seq_pos in enumerate(variable_positions):
        if fill_all:
            # 填充所有未知位置（用于 Simulation）
            allowed_aas = variable_amino_acids.get(seq_pos, config.ALLOWED_AMINO_ACIDS)
            seq_list[seq_pos] = random.choice(allowed_aas)
        elif var_idx < mcts_depth:
            # 第 0 到 mcts_depth-1 个未知氨基酸：已确定，保持当前值
            pass
        elif var_idx == mcts_depth:
            # 第 mcts_depth 个未知氨基酸：正在搜索，保持占位符
            pass
        else:
            # 第 mcts_depth+1 个及之后的未知氨基酸：随机填充
            allowed_aas = variable_amino_acids.get(seq_pos, config.ALLOWED_AMINO_ACIDS)
            seq_list[seq_pos] = random.choice(allowed_aas)
    
    return ''.join(seq_list)


def validate_sequence(sequence: str, template: str = None) -> bool:
    """
    验证序列是否符合模板要求
    
    Args:
        sequence: 完整序列
        template: 模板（默认使用 config.PEPTIDE_TEMPLATE）
    
    Returns:
        是否有效
    """
    if template is None:
        template = config.PEPTIDE_TEMPLATE
    
    # 检查长度
    if len(sequence) != len(template):
        return False
    
    # 检查固定位置
    for i, template_aa in enumerate(template):
        if template_aa != 'x' and template_aa != 'X':
            if sequence[i] != template_aa:
                return False
    
    # 检查是否包含无效字符
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    for aa in sequence:
        if aa not in valid_aas:
            return False
    
    return True


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='序列补全工具 (Sim_1)')
    parser.add_argument('-s', '--sequence', type=str, 
                        default=config.PEPTIDE_TEMPLATE,
                        help=f'输入序列（默认: {config.PEPTIDE_TEMPLATE}）')
    parser.add_argument('-n', '--num-samples', type=int, default=1,
                        help='生成样本数量（默认: 1）')
    parser.add_argument('-d', '--depth', type=int, default=0,
                        help='MCTS 当前深度（默认: 0）')
    parser.add_argument('--fill-all', action='store_true',
                        help='填充所有未知位置（用于 Simulation）')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='静默模式（只输出序列）')
    parser.add_argument('--validate', action='store_true',
                        help='验证生成的序列')
    
    args = parser.parse_args()
    
    # 静默模式跳过所有打印
    if not args.quiet:
        print(f"{'='*60}")
        print(f"Sim_1: 序列补全")
        print(f"{'='*60}")
        print(f"输入序列: {args.sequence}")
        print(f"模板: {config.PEPTIDE_TEMPLATE}")
        print(f"固定位置: {config.FIXED_POSITIONS}")
        # 计算可变位置
        variable_positions = [i for i, aa in enumerate(args.sequence) if aa in 'xX_']
        print(f"可变位置: {variable_positions} (共 {len(variable_positions)} 个)")
        print(f"MCTS 深度: {args.depth} (指第 {args.depth} 个未知氨基酸)")
        if args.depth < len(variable_positions):
            searching_pos = variable_positions[args.depth]
            print(f"  - 第 0-{args.depth-1} 个未知氨基酸: 已确定")
            print(f"  - 第 {args.depth} 个未知氨基酸: 正在搜索 (序列位置 {searching_pos})")
            print(f"  - 第 {args.depth+1}- 个未知氨基酸: 随机填充")
        print()
    
    for i in range(args.num_samples):
        filled_seq = fill_placeholder_positions(
            args.sequence, 
            mcts_depth=args.depth,
            fill_all=args.fill_all
        )
        
        if args.quiet:
            # 静默模式只输出序列
            print(filled_seq)
        elif args.validate:
            is_valid = validate_sequence(filled_seq)
            status = "✓" if is_valid else "✗"
            print(f"{status} 样本 {i+1}: {filled_seq}")
        else:
            print(f"  样本 {i+1}: {filled_seq}")
    
    if not args.quiet:
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
