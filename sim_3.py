#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sim_3: PDB 转 PDBQT 模块
功能：使用 OpenBabel 将 PDB 文件转换为 PDBQT 格式

输入：PDB 文件路径
输出：PDBQT 文件路径
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

# 输出目录
OUTPUT_DIR = Path("/mnt/d/code/AA/temporary")


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


def pdb_to_pdbqt(pdb_path: Path, output_dir: Path = None, verbose: bool = True) -> str:
    """
    将 PDB 文件转换为 PDBQT 格式
    
    Args:
        pdb_path: 输入 PDB 文件路径
        output_dir: 输出目录（默认与输入文件相同目录）
        verbose: 是否打印详细信息
    
    Returns:
        PDBQT 文件路径
    """
    pdb_path = Path(pdb_path)
    
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB 文件不存在: {pdb_path}")
    
    if output_dir is None:
        output_dir = pdb_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 输出文件路径
    pdbqt_path = output_dir / f"{pdb_path.stem}.pdbqt"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Sim_3: PDB 转 PDBQT")
        print(f"{'='*60}")
        print(f"输入 PDB: {pdb_path}")
        print(f"输出 PDBQT: {pdbqt_path}")
        
        # 设置 OpenBabel 插件路径
        babel_libdir = find_babel_libdir()
        if babel_libdir:
            print(f"BABEL_LIBDIR: {babel_libdir}")
    
    # 直接使用命令行 obabel（更稳定）
    # 注意：需要在 shell 中设置 BABEL_LIBDIR
    babel_libdir = find_babel_libdir()
    env_cmd = f"export BABEL_LIBDIR={babel_libdir}; " if babel_libdir else ""
    
    cmd = f'{env_cmd}obabel {pdb_path} -opdbqt > {pdbqt_path}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    
    if verbose:
        print(f"\nOpenBabel 返回码: {result.returncode}")
        if result.stdout:
            print(f"输出: {result.stdout.strip()}")
        if result.stderr:
            print(f"错误: {result.stderr[:200]}")
    
    if result.returncode != 0:
        raise RuntimeError(f"OpenBabel 转换失败: {result.stderr}")
    
    # 验证输出文件
    if not pdbqt_path.exists():
        raise RuntimeError("PDBQT 文件未生成")
    
    file_size = pdbqt_path.stat().st_size
    if file_size == 0:
        raise RuntimeError("PDBQT 文件为空")
    
    # 后处理：修复原子类型（Vina 1.1.2 兼容性）
    with open(pdbqt_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line in lines:
        # 将 HD 替换为 H（Vina 1.1.2 要求）
        # HD 通常在行尾，前面有空格
        if 'HD' in line and (line.startswith('ATOM') or line.startswith('HETATM')):
            # 检查最后几个字符
            if line.rstrip().endswith('HD'):
                # 替换行尾的 HD 为 H
                line = line.rstrip()[:-2] + 'H \n'
        fixed_lines.append(line)
    
    with open(pdbqt_path, 'w') as f:
        f.writelines(fixed_lines)
    
    # 验证格式
    with open(pdbqt_path, 'r') as f:
        content = f.read()
    
    if 'ROOT' not in content:
        raise RuntimeError("PDBQT 文件格式错误：缺少 ROOT 标签")
    
    if 'ATOM' not in content:
        raise RuntimeError("PDBQT 文件格式错误：缺少 ATOM 行")
    
    if verbose:
        print(f"\n✓ PDBQT 生成成功")
        print(f"  文件大小: {file_size} bytes")
        print(f"  包含 ROOT: True")
        print(f"  包含 ATOM: True")
        print(f"{'='*60}\n")
    
    return str(pdbqt_path)


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PDB 转 PDBQT 工具 (Sim_3)')
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='输入 PDB 文件路径')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help='输出目录（默认与输入文件相同）')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='静默模式')
    
    args = parser.parse_args()
    
    try:
        pdbqt_path = pdb_to_pdbqt(
            pdb_path=args.input,
            output_dir=args.output,
            verbose=not args.quiet
        )
        
        if args.quiet:
            print(pdbqt_path, end='')
        else:
            print(f"✓ PDBQT 文件: {pdbqt_path}")
        
        return 0
        
    except Exception as e:
        print(f"✗ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
