#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sim_2: 肽构象生成模块
功能：使用 RDKit 生成肽的 3D 构象
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, List

sys.path.insert(0, str(Path(__file__).parent))

import config

OUTPUT_DIR = Path("/mnt/d/code/AA/temporary")

# 氨基酸侧链 SMILES（简化版）
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
    'Y': 'Cc1ccc(O)cc1'   # Tyr
}


def get_cys_positions(sequence: str) -> List[int]:
    """获取序列中所有 Cys 的位置"""
    return [i for i, aa in enumerate(sequence) if aa == 'C']


def generate_conformation(sequence: str, 
                          output_dir: Path = None,
                          verbose: bool = True) -> str:
    """
    生成肽的 3D 构象
    
    Args:
        sequence: 完整氨基酸序列
        output_dir: 输出目录
        verbose: 是否打印详细信息
    
    Returns:
        PDB 文件路径
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Sim_2: 肽构象生成")
        print(f"{'='*60}")
        print(f"序列: {sequence}")
        print(f"长度: {len(sequence)} 个氨基酸")
    
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        if verbose:
            print(f"\n[1/2] 构建肽分子...")
        
        # 使用 RDKit 的肽构建功能
        mol = build_peptide_from_sequence(sequence)
        
        if mol is None:
            raise RuntimeError("肽分子构建失败")
        
        if verbose:
            print(f"  分子创建成功: {mol.GetNumAtoms()} 个原子")
        
        # 添加氢原子
        if verbose:
            print(f"\n[2/2] 生成 3D 构象...")
        
        mol = Chem.AddHs(mol)
        
        # 尝试多种方法生成构象
        success = False
        
        # 方法1: ETKDGv3
        try:
            from rdkit.Chem import rdDistGeom
            params = rdDistGeom.ETKDGv3()
            params.randomSeed = 42
            result = rdDistGeom.EmbedMolecule(mol, params)
            if result == 0:
                success = True
        except Exception as e:
            if verbose:
                print(f"  ETKDGv3 失败: {e}")
        
        # 方法2: 标准 EmbedMolecule
        if not success:
            result = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=100)
            if result == 0:
                success = True
        
        # 方法3: 随机坐标
        if not success:
            result = AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=100, randomSeed=42)
            if result == 0:
                success = True
        
        if not success:
            raise RuntimeError("所有构象生成方法都失败")
        
        # 优化构象
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        
        # 验证坐标
        conf = mol.GetConformer()
        atom0_pos = conf.GetAtomPosition(0)
        if atom0_pos.x == 0 and atom0_pos.y == 0 and atom0_pos.z == 0:
            # 尝试第二个构象
            if conf.GetId() > 0:
                conf = mol.GetConformer(1)
                atom0_pos = conf.GetAtomPosition(0)
        
        # 保存为 PDB
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


def build_peptide_from_sequence(sequence: str):
    """
    使用 RDKit 从序列构建肽分子
    使用 MolFromSmiles 构建每个氨基酸，然后手动连接
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    if len(sequence) == 0:
        return None
    
    # 构建 SMILES：N-乙酰基肽酰胺
    # 格式: CC(=O)-[氨基酸1]-[氨基酸2]-...-[氨基酸n]-N
    smiles_parts = []
    
    # N-端乙酰基
    smiles_parts.append("CC(=O)")
    
    for i, aa in enumerate(sequence):
        if aa not in AA_SIDE_CHAINS:
            raise ValueError(f"未知的氨基酸: {aa}")
        
        side = AA_SIDE_CHAINS[aa]
        
        if aa == 'P':
            # 脯氨酸
            if i == len(sequence) - 1:
                smiles_parts.append("N1CCCC1C(=O)N")
            else:
                smiles_parts.append("N1CCCC1C(=O)")
        elif aa == 'G':
            # 甘氨酸
            if i == len(sequence) - 1:
                smiles_parts.append("NCC(=O)N")
            else:
                smiles_parts.append("NCC(=O)")
        else:
            # 标准氨基酸
            if side:
                if i == len(sequence) - 1:
                    smiles_parts.append(f"N[C@H]({side})C(=O)N")
                else:
                    smiles_parts.append(f"N[C@H]({side})C(=O)")
            else:
                if i == len(sequence) - 1:
                    smiles_parts.append("N[C@H]C(=O)N")
                else:
                    smiles_parts.append("N[C@H]C(=O)")
    
    smiles = "".join(smiles_parts)
    
    # 解析 SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # 尝试简化版本（无手性）
        smiles_simple = smiles.replace("[C@H]", "C")
        mol = Chem.MolFromSmiles(smiles_simple)
    
    return mol


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='肽构象生成工具 (Sim_2)')
    parser.add_argument('-s', '--sequence', type=str, required=True,
                        help='完整氨基酸序列')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help=f'输出目录（默认: {OUTPUT_DIR}）')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='静默模式')
    
    args = parser.parse_args()
    
    try:
        pdb_path = generate_conformation(
            sequence=args.sequence,
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
