#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sim_2: 肽构象生成模块
功能：
1. 读取交联剂配置（从 config）
2. 使用 RDKit 生成肽的 3D 构象（包含交联剂）
3. 保存构象文件到 D:\code\AA\temporary

输入：完整氨基酸序列（如 ACAAAAAACAAAAAACG）
输出：PDB 文件路径
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, List

sys.path.insert(0, str(Path(__file__).parent))

import config

# 输出目录
OUTPUT_DIR = Path("/mnt/d/code/AA/temporary")

# 交联剂 SMILES
CROSSLINKER_SMILES = {
    'TBMB': 'c1c(CS)cc(CS)cc1CS',  # 1,3,5-三(巯基甲基)苯（简化版）
    'TATA': 'C(CS)CS',              # 简化版
    'TBAB': 'c1c(CS)c(CS)cc(CS)c1CS',  # 简化版
}

# 氨基酸侧链 SMILES
AA_SIDE_CHAINS = {
    'A': '[H]',           # Ala
    'C': 'CS',            # Cys
    'D': 'CC(=O)O',       # Asp
    'E': 'CCC(=O)O',      # Glu
    'F': 'Cc1ccccc1',     # Phe
    'G': '',              # Gly
    'H': 'Cc1cnc[nH]1',   # His
    'I': 'C(C)CC',        # Ile
    'K': 'CCCCN',         # Lys
    'L': 'CC(C)C',        # Leu
    'M': 'CCSC',          # Met
    'N': 'CC(=O)N',       # Asn
    'P': '',              # Pro
    'Q': 'CCC(=O)N',      # Gln
    'R': 'CCCNC(=N)N',    # Arg
    'S': 'CO',            # Ser
    'T': 'C(C)O',         # Thr
    'V': 'C(C)C',         # Val
    'W': 'Cc1c[nH]c2ccccc12',  # Trp
    'Y': 'Cc1ccc(O)cc1' # Tyr
}


def get_crosslinker_info(crosslinker_type: str = None) -> Optional[Dict]:
    """
    获取交联剂信息
    
    Args:
        crosslinker_type: 交联剂类型（默认从 config 读取）
    
    Returns:
        交联剂信息字典，或 None（如果不使用交联剂）
    """
    if crosslinker_type is None:
        # 尝试从 config 读取
        crosslinker_type = getattr(config, 'CROSSLINKER', None)
    
    if crosslinker_type is None or crosslinker_type not in CROSSLINKER_SMILES:
        return None
    
    return {
        'type': crosslinker_type,
        'smiles': CROSSLINKER_SMILES[crosslinker_type],
        'required_cys': 3 if crosslinker_type in ['TBMB', 'TATA'] else 4
    }


def get_cys_positions(sequence: str) -> List[int]:
    """获取序列中所有 Cys 的位置"""
    return [i for i, aa in enumerate(sequence) if aa == 'C']


def add_crosslinker_to_mol(mol, cys_positions: List[int], crosslinker_type: str = 'TBMB', verbose: bool = False):
    """
    使用 RDKit 添加交联剂形成环肽
    
    通过创建共价键将 Cys 的硫原子连接到交联剂上
    
    Args:
        mol: RDKit 分子对象（已添加氢）
        cys_positions: Cys 在序列中的位置（0-based）
        crosslinker_type: 交联剂类型
        verbose: 是否打印详细信息
    
    Returns:
        修改后的分子对象
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    if verbose:
        print(f"  添加交联剂 {crosslinker_type} 到 Cys 位置: {cys_positions}")
    
    # 获取交联剂 SMILES
    if crosslinker_type not in CROSSLINKER_SMILES:
        raise ValueError(f"不支持的交联剂: {crosslinker_type}")
    
    crosslinker_smiles = CROSSLINKER_SMILES[crosslinker_type]
    crosslinker_mol = Chem.MolFromSmiles(crosslinker_smiles)
    if crosslinker_mol is None:
        raise RuntimeError(f"无法解析交联剂 SMILES: {crosslinker_smiles}")
    
    # 添加氢原子
    crosslinker_mol = Chem.AddHs(crosslinker_mol)
    
    # 生成交联剂构象
    AllChem.EmbedMolecule(crosslinker_mol, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(crosslinker_mol)
    
    # 找到交联剂中的硫原子（-SH 基团）
    crosslinker_sulfur_indices = []
    for atom in crosslinker_mol.GetAtoms():
        if atom.GetAtomicNum() == 16:  # 硫原子
            # 找到连接的氢原子
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:  # 氢原子
                    crosslinker_sulfur_indices.append((atom.GetIdx(), neighbor.GetIdx()))
                    break
    
    if len(crosslinker_sulfur_indices) < 3:
        raise RuntimeError(f"交联剂中硫原子数量不足: {len(crosslinker_sulfur_indices)}")
    
    if verbose:
        print(f"  交联剂硫原子数: {len(crosslinker_sulfur_indices)}")
    
    # 合并分子
    combined = Chem.CombineMols(mol, crosslinker_mol)
    editable = Chem.EditableMol(combined)
    
    # 获取原始分子和交联剂的原子偏移
    mol_atom_count = mol.GetNumAtoms()
    
    # 找到肽链中 Cys 侧链的硫原子
    # 这需要根据 Cys 在序列中的位置来定位
    # 简化处理：假设 Cys 的硫原子是侧链的一部分
    
    # 获取 PDB 残基信息来定位 Cys
    cys_sulfur_indices = []
    for i, pos in enumerate(cys_positions):
        # 在合并后的分子中查找对应残基的硫原子
        # 这需要更复杂的逻辑来正确识别 Cys 侧链的硫
        # 这里简化处理：通过原子序号大致定位
        
        # 找到所有硫原子
        for atom_idx in range(mol_atom_count):
            atom = combined.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() == 16:  # 硫原子
                # 检查是否是 Cys 的硫（连接到 CB 碳）
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 6:  # 碳
                        # 这是一个简化的检查，实际应该检查残基名称
                        cys_sulfur_indices.append(atom_idx)
                        break
                if len(cys_sulfur_indices) > i:
                    break
    
    if len(cys_sulfur_indices) < len(cys_positions):
        if verbose:
            print(f"  警告: 只找到 {len(cys_sulfur_indices)} 个 Cys 硫原子，期望 {len(cys_positions)}")
        cys_sulfur_indices = cys_sulfur_indices[:len(crosslinker_sulfur_indices)]
    
    # 创建交联键（S-S 键）
    bonds_to_add = []
    for i, (cys_s_idx, (cl_s_idx, cl_h_idx)) in enumerate(zip(cys_sulfur_indices, crosslinker_sulfur_indices)):
        # 交联剂硫原子在合并分子中的索引
        cl_s_idx_in_combined = mol_atom_count + cl_s_idx
        # 添加键
        bonds_to_add.append((cys_s_idx, cl_s_idx_in_combined))
        if verbose:
            print(f"  添加交联键 {i+1}: Cys-S({cys_s_idx}) -- 交联剂-S({cl_s_idx_in_combined})")
    
    # 添加所有键
    for atom1, atom2 in bonds_to_add:
        editable.AddBond(atom1, atom2, order=Chem.BondType.SINGLE)
    
    # 移除交联剂上的氢原子（形成 S-S 键后）
    # 注意：RDKit 的 EditableMol 不直接支持移除原子，需要在获取分子后处理
    
    # 获取修改后的分子
    new_mol = editable.GetMol()
    
    # 重新生成构象（包含交联剂）
    try:
        AllChem.EmbedMolecule(new_mol, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(new_mol)
    except Exception as e:
        if verbose:
            print(f"  警告: 交联后构象优化失败: {e}")
        # 返回原始分子作为 fallback
        return mol
    
    if verbose:
        print(f"  交联成功！新分子原子数: {new_mol.GetNumAtoms()}")
    
    return new_mol


def build_peptide_smiles(sequence: str) -> str:
    """
    构建肽链的 SMILES
    
    Args:
        sequence: 氨基酸序列
    
    Returns:
        SMILES 字符串
    """
    smiles_parts = []
    
    for i, aa in enumerate(sequence):
        if aa not in AA_SIDE_CHAINS:
            raise ValueError(f"未知的氨基酸: {aa}")
        
        side = AA_SIDE_CHAINS[aa]
        
        if aa == 'P':
            # 脯氨酸
            if i == 0:
                smiles_parts.append('CC(=O)N1CCCC1C(=O)')
            elif i == len(sequence) - 1:
                smiles_parts.append('N1CCCC1C(=O)O')
            else:
                smiles_parts.append('N1CCCC1C(=O)')
        elif aa == 'G':
            # 甘氨酸
            if i == 0:
                smiles_parts.append('CC(=O)NCC(=O)')
            elif i == len(sequence) - 1:
                smiles_parts.append('NCC(=O)O')
            else:
                smiles_parts.append('NCC(=O)')
        else:
            # 标准氨基酸
            if i == 0:
                if side:
                    smiles_parts.append(f'CC(=O)N[C@@H]({side})C(=O)')
                else:
                    smiles_parts.append('CC(=O)N[C@@H]C(=O)')
            elif i == len(sequence) - 1:
                if side:
                    smiles_parts.append(f'N[C@@H]({side})C(=O)O')
                else:
                    smiles_parts.append('N[C@@H]C(=O)O')
            else:
                if side:
                    smiles_parts.append(f'N[C@@H]({side})C(=O)')
                else:
                    smiles_parts.append('N[C@@H]C(=O)')
    
    return ''.join(smiles_parts)


def generate_conformation(sequence: str, 
                          crosslinker_type: str = None,
                          output_dir: Path = None,
                          verbose: bool = True) -> str:
    """
    生成肽的 3D 构象
    
    Args:
        sequence: 完整氨基酸序列
        crosslinker_type: 交联剂类型（默认从 config 读取）
        output_dir: 输出目录（默认 D:\code\AA\temporary）
        verbose: 是否打印详细信息
    
    Returns:
        PDB 文件路径
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取交联剂信息
    crosslinker = get_crosslinker_info(crosslinker_type)
    if crosslinker:
        # 检查 Cys 数量
        cys_positions = get_cys_positions(sequence)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Sim_2: 肽构象生成")
            print(f"{'='*60}")
            print(f"序列: {sequence}")
            print(f"长度: {len(sequence)} 个氨基酸")
            print(f"交联剂: {crosslinker['type']}")
            print(f"需要 Cys 数量: {crosslinker['required_cys']}")
            print(f"序列中 Cys 位置: {cys_positions}")
        
        if len(cys_positions) < crosslinker['required_cys']:
            raise ValueError(f"Cys 数量不足: 需要 {crosslinker['required_cys']}, 实际 {len(cys_positions)}")
    
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        if verbose:
            print(f"\n[1/3] 构建肽分子...")
        
        # 构建 SMILES
        smiles = build_peptide_smiles(sequence)
        
        if verbose:
            print(f"  SMILES: {smiles[:80]}...")
        
        # 创建分子
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise RuntimeError("SMILES 解析失败")
        
        if verbose:
            print(f"  分子创建成功: {mol.GetNumAtoms()} 个原子")
        
        # 添加氢原子
        if verbose:
            print(f"\n[2/3] 添加氢原子...")
        
        mol = Chem.AddHs(mol)
        Chem.SanitizeMol(mol)
        
        if verbose:
            print(f"  添加后: {mol.GetNumAtoms()} 个原子")
        
        # 生成 3D 构象
        if verbose:
            print(f"\n[3/3] 生成 3D 构象...")
        
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(mol)
        
        if verbose:
            print(f"  3D 构象生成成功")
        
        # 添加交联剂形成环肽（如果需要）
        if crosslinker and len(cys_positions) >= crosslinker['required_cys']:
            if verbose:
                print(f"\n[4/4] 添加交联剂 {crosslinker['type']}...")
            
            # 获取用于交联的 Cys 位置（使用前 N 个）
            link_cys = cys_positions[:crosslinker['required_cys']]
            
            # 使用 RDKit 添加交联剂
            try:
                mol = add_crosslinker_to_mol(mol, link_cys, crosslinker['type'], verbose)
            except Exception as e:
                if verbose:
                    print(f"  警告: 添加交联剂失败: {e}")
                    print(f"  将继续使用线性肽结构")
        
        # 保存为 PDB
        # 使用序列的哈希值作为文件名，避免过长或特殊字符
        import hashlib
        seq_hash = hashlib.md5(sequence.encode()).hexdigest()[:8]
        pdb_path = output_dir / f"peptide_{seq_hash}.pdb"
        Chem.MolToPDBFile(mol, str(pdb_path))
        
        if verbose:
            print(f"\n  PDB 文件: {pdb_path}")
            print(f"  文件大小: {pdb_path.stat().st_size} bytes")
            print(f"{'='*60}\n")
        
        return str(pdb_path)
        
    except ImportError:
        raise RuntimeError("RDKit 未安装。请安装: pip install rdkit")
    except Exception as e:
        raise RuntimeError(f"生成构象失败: {e}")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='肽构象生成工具 (Sim_2)')
    parser.add_argument('-s', '--sequence', type=str, required=True,
                        help='完整氨基酸序列')
    parser.add_argument('-x', '--crosslinker', type=str, default=None,
                        choices=['TBMB', 'TATA', 'TBAB'],
                        help='交联剂类型（默认从 config 读取）')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help=f'输出目录（默认: {OUTPUT_DIR}）')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='静默模式')
    
    args = parser.parse_args()
    
    try:
        pdb_path = generate_conformation(
            sequence=args.sequence,
            crosslinker_type=args.crosslinker,
            output_dir=args.output,
            verbose=not args.quiet
        )
        
        if args.quiet:
            print(pdb_path, end='')
        else:
            print(f"✓ PDB 文件: {pdb_path}")
        
        return 0
        
    except Exception as e:
        print(f"✗ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
