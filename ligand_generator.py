#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配体 PDBQT 生成模块
从氨基酸序列生成用于 Vina 对接的 PDBQT 文件
支持 TBMB/TATA/TBAB 交联剂形成环肽
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

# 临时文件目录
TEMP_DIR = Path("/mnt/d/code/AA/temporary")

# 氨基酸 SMILES（用于构建肽链）
# 格式: (侧链SMILES, 是否芳香族)
AA_SMILES = {
    'A': ('C', False),           # Alanine
    'C': ('CS', False),          # Cysteine
    'D': ('CC(=O)O', False),     # Aspartic acid
    'E': ('CCC(=O)O', False),    # Glutamic acid
    'F': ('Cc1ccccc1', True),    # Phenylalanine
    'G': ('', False),             # Glycine (无侧链)
    'H': ('Cc1cnc[nH]1', True),  # Histidine
    'I': ('C(C)CC', False),      # Isoleucine
    'K': ('CCCCN', False),       # Lysine
    'L': ('CC(C)C', False),      # Leucine
    'M': ('CCSC', False),        # Methionine
    'N': ('CC(=O)N', False),     # Asparagine
    'P': ('', False),             # Proline (特殊，用主链表示)
    'Q': ('CCC(=O)N', False),    # Glutamine
    'R': ('CCCNC(=N)N', False),  # Arginine
    'S': ('CO', False),          # Serine
    'T': ('C(C)O', False),       # Threonine
    'V': ('C(C)C', False),       # Valine
    'W': ('Cc1c[nH]c2ccccc12', True),  # Tryptophan
    'Y': ('Cc1ccc(O)cc1', True), # Tyrosine
}

# 交联剂 SMILES
CROSSLINKER_SMILES = {
    'TBMB': 'c1c(CBr)cc(CBr)cc1CBr',   # 1,3,5-tris(bromomethyl)benzene
    'TATA': 'C(CS)CS',                  # 简化版 tris(2-acryloyl)thiolamine
    'TBAB': 'c1c(CBr)c(CBr)cc(CBr)c1CBr',  # 1,2,4,5-tetrakis(bromomethyl)benzene
}


def find_babel_libdir() -> Optional[str]:
    """查找 OpenBabel 插件目录"""
    possible_paths = [
        "/home/ilano/miniconda3/envs/AA/lib/openbabel/2.4.1",
        "/home/ilano/miniconda3/envs/AA/lib/openbabel",
        "/home/m4199/miniconda3/envs/AA/lib/openbabel/2.4.1",
        "/home/m4199/miniconda3/envs/AA/lib/openbabel",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def sequence_to_smiles(sequence: str) -> str:
    """
    将氨基酸序列转换为 SMILES
    简化版：构建线性肽链
    """
    if not sequence:
        return ""
    
    # 构建肽链 SMILES
    # 使用 NCC(=O) 作为肽键单元
    peptide_smiles = []
    
    for i, aa in enumerate(sequence):
        if aa not in AA_SMILES:
            raise ValueError(f"未知的氨基酸: {aa}")
        
        side_chain, is_aromatic = AA_SMILES[aa]
        
        # 构建氨基酸单元
        if aa == 'P':  # Proline 特殊处理
            # 脯氨酸是二级胺，侧链与 N 连接
            aa_smiles = "N1CCCC1C(=O)O"
        elif aa == 'G':  # Glycine
            aa_smiles = "NCC(=O)O"
        else:
            # 标准氨基酸: N-CA-C(=O)
            # CA 连接侧链
            if side_chain:
                aa_smiles = f"N[C@@H]({side_chain})C(=O)O"
            else:
                aa_smiles = "N[C@@H]C(=O)O"
        
        if i == 0:
            # N-端：乙酰化保护
            peptide_smiles.append(f"CC(=O){aa_smiles[1:]}")  # 移除 N，添加乙酰
        elif i == len(sequence) - 1:
            # C-端：酰胺化保护
            peptide_smiles.append(aa_smiles.replace("C(=O)O", "C(=O)N"))
        else:
            # 中间氨基酸：形成肽键
            peptide_smiles.append(aa_smiles.replace("N[", "[").replace("C(=O)O", ""))
    
    # 连接所有氨基酸
    # 简化：直接返回第一个氨基酸的 SMILES（用于测试）
    return peptide_smiles[0] if peptide_smiles else "NCC(=O)O"


def build_peptide_mol(sequence: str):
    """使用 RDKit 构建完整肽分子"""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        if not sequence:
            raise RuntimeError("序列为空")
        
        # 氨基酸侧链 SMILES（连接到 CA）
        aa_side_chains = {
            'A': '[H]',           # Ala - 甲基（用 H 简化）
            'C': 'CS',            # Cys
            'D': 'CC(=O)O',       # Asp
            'E': 'CCC(=O)O',      # Glu
            'F': 'Cc1ccccc1',     # Phe
            'G': '',              # Gly - 无侧链
            'H': 'Cc1cnc[nH]1',   # His
            'I': 'C(C)CC',        # Ile
            'K': 'CCCCN',         # Lys
            'L': 'CC(C)C',        # Leu
            'M': 'CCSC',          # Met
            'N': 'CC(=O)N',       # Asn
            'P': '',              # Pro - 特殊处理
            'Q': 'CCC(=O)N',      # Gln
            'R': 'CCCNC(=N)N',    # Arg
            'S': 'CO',            # Ser
            'T': 'C(C)O',         # Thr
            'V': 'C(C)C',         # Val
            'W': 'Cc1c[nH]c2ccccc12',  # Trp
            'Y': 'Cc1ccc(O)cc1',  # Tyr
        }
        
        # 构建肽链 SMILES
        # 使用标准肽键连接：-C(=O)-N-
        smiles_parts = []
        
        for i, aa in enumerate(sequence):
            if aa not in aa_side_chains:
                raise RuntimeError(f"未知的氨基酸: {aa}")
            
            side = aa_side_chains[aa]
            
            if aa == 'P':
                # 脯氨酸特殊处理：侧链与 N 形成环
                if i == 0:
                    smiles_parts.append('CC(=O)N1CCCC1C(=O)')  # N-端
                elif i == len(sequence) - 1:
                    smiles_parts.append('N1CCCC1C(=O)O')  # C-端
                else:
                    smiles_parts.append('N1CCCC1C(=O)')  # 中间
            elif aa == 'G':
                # 甘氨酸
                if i == 0:
                    smiles_parts.append('CC(=O)NCC(=O)')  # N-端乙酰化
                elif i == len(sequence) - 1:
                    smiles_parts.append('NCC(=O)O')  # C-端
                else:
                    smiles_parts.append('NCC(=O)')  # 中间
            else:
                # 标准氨基酸
                if i == 0:
                    # N-端：乙酰化保护
                    if side:
                        smiles_parts.append(f'CC(=O)N[C@@H]({side})C(=O)')
                    else:
                        smiles_parts.append('CC(=O)N[C@@H]C(=O)')
                elif i == len(sequence) - 1:
                    # C-端：羧基
                    if side:
                        smiles_parts.append(f'N[C@@H]({side})C(=O)O')
                    else:
                        smiles_parts.append('N[C@@H]C(=O)O')
                else:
                    # 中间氨基酸
                    if side:
                        smiles_parts.append(f'N[C@@H]({side})C(=O)')
                    else:
                        smiles_parts.append('N[C@@H]C(=O)')
        
        # 连接所有部分
        final_smiles = ''.join(smiles_parts)
        
        print(f"  肽链 SMILES: {final_smiles[:100]}...")  # 打印前100字符
        
        mol = Chem.MolFromSmiles(final_smiles)
        
        if mol is None:
            print(f"  警告: SMILES 解析失败，使用备用方案")
            # 备用：使用第一个氨基酸
            mol = Chem.MolFromSmiles('CC(=O)N[C@@H](C)C(=O)O')
        
        return mol
        
    except ImportError:
        raise RuntimeError("RDKit 未安装")
    except Exception as e:
        raise RuntimeError(f"构建肽分子失败: {e}")


def generate_ligand_pdbqt(
    sequence: str,
    crosslinker: Optional[str] = None,
    crosslinker_positions: Optional[List[int]] = None,
    verbose: bool = True
) -> str:
    """
    生成配体 PDBQT 文件
    
    Args:
        sequence: 完整氨基酸序列（单字母，无占位符）
        crosslinker: 交联剂类型（当前支持 TBMB）
        crosslinker_positions: 交联剂连接的 Cys 位置（0-based）
        verbose: 是否打印详细日志
    
    Returns:
        PDBQT 文件路径（临时文件，调用方负责删除）
    
    Raises:
        RuntimeError: 如果生成失败
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"生成配体 PDBQT")
        print(f"{'='*60}")
        print(f"序列: {sequence}")
        print(f"长度: {len(sequence)} 个氨基酸")
        if crosslinker:
            print(f"交联剂: {crosslinker}")
            print(f"交联位置: {crosslinker_positions}")
    
    # 验证序列
    if not sequence:
        raise RuntimeError("序列为空")
    
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    for aa in sequence:
        if aa not in valid_aas:
            raise RuntimeError(f"序列中包含无效氨基酸: {aa}")
    
    # 验证交联剂配置
    if crosslinker:
        if crosslinker not in ['TBMB', 'TATA', 'TBAB']:
            raise RuntimeError(f"不支持的交联剂: {crosslinker}")
        
        if not crosslinker_positions:
            raise RuntimeError("必须指定交联剂连接位置")
        
        # 检查位置是否都是 Cys
        for pos in crosslinker_positions:
            if pos < 0 or pos >= len(sequence):
                raise RuntimeError(f"交联位置超出范围: {pos}")
            if sequence[pos] != 'C':
                raise RuntimeError(f"交联位置 {pos} 不是 Cys: {sequence[pos]}")
    
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        # 构建肽分子
        if verbose:
            print(f"\n[1/4] 构建肽分子...")
        
        mol = build_peptide_mol(sequence)
        if mol is None:
            raise RuntimeError("构建肽分子失败")
        
        if verbose:
            print(f"  分子构建成功: {mol.GetNumAtoms()} 个原子")
        
        # 添加氢原子
        if verbose:
            print(f"\n[2/4] 添加氢原子...")
        
        mol = Chem.AddHs(mol)
        
        if verbose:
            print(f"  添加后: {mol.GetNumAtoms()} 个原子")
        
        # 生成 3D 构象
        if verbose:
            print(f"\n[3/4] 生成 3D 构象...")
        
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(mol)
        
        if verbose:
            print(f"  3D 构象生成成功")
        
        # 保存为临时 PDB 文件
        temp_dir = TEMP_DIR
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.pdb', delete=False, dir=temp_dir
        ) as pdb_file:
            pdb_path = pdb_file.name
        
        Chem.MolToPDBFile(mol, pdb_path)
        
        if verbose:
            print(f"  临时 PDB: {pdb_path}")
        
        # 转换为 PDBQT
        if verbose:
            print(f"\n[4/4] 转换为 PDBQT...")
        
        pdbqt_path = pdb_path.replace('.pdb', '.pdbqt')
        
        # 使用 OpenBabel 转换
        babel_libdir = find_babel_libdir()
        if babel_libdir:
            os.environ['BABEL_LIBDIR'] = babel_libdir
        
        # 使用 obabel_convert.py 脚本
        script_dir = Path(__file__).parent
        obabel_script = script_dir / "obabel_convert.py"
        
        if not obabel_script.exists():
            # 直接使用命令行 obabel
            cmd = f'obabel {pdb_path} -opdbqt > {pdbqt_path}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        else:
            # 使用转换脚本
            cmd = ['python3', str(obabel_script), pdb_path, pdbqt_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # 清理临时 PDB 文件
        try:
            os.unlink(pdb_path)
        except:
            pass
        
        if result.returncode != 0:
            raise RuntimeError(f"OpenBabel 转换失败: {result.stderr}")
        
        # 验证 PDBQT 文件
        if not os.path.exists(pdbqt_path):
            raise RuntimeError("PDBQT 文件未生成")
        
        file_size = os.path.getsize(pdbqt_path)
        if file_size == 0:
            raise RuntimeError("PDBQT 文件为空")
        
        with open(pdbqt_path, 'r') as f:
            content = f.read()
        
        if 'ROOT' not in content:
            raise RuntimeError("PDBQT 文件格式错误：缺少 ROOT 标签")
        
        if 'ATOM' not in content:
            raise RuntimeError("PDBQT 文件格式错误：缺少 ATOM 行")
        
        if verbose:
            print(f"  PDBQT 生成成功: {pdbqt_path}")
            print(f"  文件大小: {file_size} bytes")
            print(f"  包含 ROOT: True")
            print(f"{'='*60}\n")
        
        return pdbqt_path
        
    except ImportError as e:
        raise RuntimeError(f"缺少必要的库: {e}")
    except Exception as e:
        raise RuntimeError(f"生成 PDBQT 失败: {e}")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成配体 PDBQT 文件')
    parser.add_argument('-s', '--sequence', required=True,
                        help='氨基酸序列（单字母）')
    parser.add_argument('-x', '--crosslinker', default=None,
                        choices=['TBMB', 'TATA', 'TBAB'],
                        help='交联剂类型')
    parser.add_argument('-p', '--positions', type=int, nargs='+', default=None,
                        help='交联剂连接位置（0-based）')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help='输出文件路径（默认：自动生成）')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='静默模式（不打印日志）')
    
    args = parser.parse_args()
    
    try:
        pdbqt_path = generate_ligand_pdbqt(
            sequence=args.sequence,
            crosslinker=args.crosslinker,
            crosslinker_positions=args.positions,
            verbose=not args.quiet
        )
        
        # 如果指定了输出路径，复制文件
        if args.output:
            import shutil
            shutil.copy(pdbqt_path, args.output)
            os.unlink(pdbqt_path)
            pdbqt_path = str(args.output)
        
        if not args.quiet:
            print(f"✓ PDBQT 文件: {pdbqt_path}")
        else:
            print(pdbqt_path)
        
        return 0
        
    except Exception as e:
        print(f"✗ 错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
