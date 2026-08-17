# -*- coding: utf-8 -*-
"""
Vina 受体准备模块
功能：将 PDB 转换为 Vina 可识别的 PDBQT 格式，生成对接配置
输出：
  - results/[target_name]/vina/vina-receptor.pdbqt
  - results/[target_name]/vina/vina_config.txt
"""

import os
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent))

from config import get_target_dirs, TOOLS, VINA_BOX_SIZE


# Vina 原子类型映射（基于 AutoDock4 力场）
# 格式: (元素符号, 原子类型)
# 注意：Vina 1.2.0+ 使用 H 而非 HD 表示氢原子
VINA_ATOM_TYPES = {
    "C": "C",   # 碳（脂肪族）
    "CA": "A",  # 碳（芳香族）
    "N": "N",   # 氮（酰胺）
    "NA": "NA", # 氮（受体）
    "NS": "NA", # 氮（磺酰胺）
    "O": "O",   # 氧（羰基）
    "OA": "OA", # 氧（羟基/酚）
    "OS": "OA", # 氧（醚/酯）
    "S": "S",   # 硫
    "SA": "S",  # 硫（受体）
    "P": "P",   # 磷
    "F": "F",   # 氟
    "Cl": "Cl", # 氯
    "CL": "Cl",
    "Br": "Br", # 溴
    "BR": "Br",
    "I": "I",   # 碘
    "H": "H",   # 氢（Vina 1.2.0+ 使用 H）
    "HD": "H",  # 氢（极性，向后兼容）
    "HS": "H",  # 氢（非极性）
    "MG": "Mg", # 镁
    "ZN": "Zn", # 锌
    "CA": "Ca", # 钙
    "FE": "Fe", # 铁
    "MN": "Mn", # 锰
}


def parse_pdb_line(line: str) -> dict:
    """
    解析 PDB ATOM/HETATM 行
    
    字段位置（1-based indexing）：
    1-6:   记录名 (ATOM/HETATM)
    7-11:  原子序号
    13-16: 原子名称
    18-20: 残基名
    22:    链标识
    23-26: 残基序号
    31-38: X 坐标
    39-46: Y 坐标
    47-54: Z 坐标
    55-60: 占据率
    61-66: 温度因子
    77-78: 元素符号
    """
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return None
    
    record = line[0:6].strip()
    atom_serial = line[6:11].strip()
    atom_name = line[12:16].strip()
    res_name = line[17:20].strip()
    chain_id = line[21].strip() if len(line) > 21 else ""
    res_seq = line[22:26].strip()
    
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except (ValueError, IndexError):
        return None
    
    occupancy = line[54:60].strip() if len(line) >= 60 else "1.00"
    temp_factor = line[60:66].strip() if len(line) >= 66 else "0.00"
    
    # 元素符号（列 77-78）
    if len(line) >= 78:
        element = line[76:78].strip()
    else:
        # 从原子名推断
        element = atom_name[0] if atom_name else ""
        if len(atom_name) > 1 and atom_name[0].isdigit():
            element = atom_name[1]
        elif len(atom_name) > 1 and atom_name[:2] in ["CL", "BR", "FE", "ZN", "CA", "MG", "MN", "NA"]:
            element = atom_name[:2]
    
    # 确定 Vina 原子类型
    atom_type = determine_atom_type(atom_name, res_name, element)
    
    # 计算 Gasteiger 部分电荷（简化版）
    charge = calculate_gasteiger_charge(atom_name, res_name, element)
    
    return {
        "record": record,
        "atom_serial": int(atom_serial) if atom_serial.isdigit() else 0,
        "atom_name": atom_name,
        "res_name": res_name,
        "chain_id": chain_id,
        "res_seq": res_seq,
        "x": x,
        "y": y,
        "z": z,
        "occupancy": float(occupancy) if occupancy else 1.00,
        "temp_factor": float(temp_factor) if temp_factor else 0.00,
        "charge": charge,
        "atom_type": atom_type,
    }


def determine_atom_type(atom_name: str, res_name: str, element: str) -> str:
    """确定 Vina 原子类型"""
    element = element.upper()
    atom_name = atom_name.upper()
    res_name = res_name.upper()
    
    # 金属离子
    if element in ["ZN", "MG", "CA", "MN", "FE", "NA", "K"]:
        return element
    
    # 卤素
    if element in ["F", "CL", "BR", "I"]:
        return element if element != "CL" else "Cl"
    if element == "CL" or atom_name.startswith("CL"):
        return "Cl"
    if element == "BR" or atom_name.startswith("BR"):
        return "Br"
    
    # 碳
    if element == "C":
        # 芳香族碳检测（简化：基于常见芳香残基）
        aromatic_res = ["PHE", "TYR", "TRP", "HIS", "HID", "HIE", "HIP"]
        aromatic_atoms = ["CG", "CD", "CE", "CZ", "CH", "CD1", "CD2", "CE1", "CE2", "CZ1", "CZ2", "CZ3", "CE3", "CH2", "ND1", "NE2"]
        
        if res_name in aromatic_res and any(atom_name.startswith(a) for a in aromatic_atoms):
            return "A"
        return "C"
    
    # 氮
    if element == "N":
        # 质子化氮（正电荷）
        if res_name in ["LYS", "LYN"] and atom_name in ["NZ", "NZ1"]:
            return "N"
        if res_name in ["ARG"] and atom_name in ["NE", "NH1", "NH2"]:
            return "N"
        if res_name in ["HIS", "HID", "HIE", "HIP"]:
            return "NA"
        return "N"
    
    # 氧
    if element == "O":
        # 羟基氧（受体/供体）
        if res_name in ["SER", "THR", "TYR"] and atom_name in ["OG", "OG1", "OH"]:
            return "OA"
        if res_name == "HOH" or atom_name.startswith("OW"):
            return "OA"
        return "O"
    
    # 硫
    if element == "S":
        if res_name == "MET" and atom_name == "SD":
            return "S"
        if res_name == "CYS" and atom_name == "SG":
            return "S"
        return "S"
    
    # 磷
    if element == "P":
        return "P"
    
    # 氢（Vina 1.2.0+ 使用 H 而非 HD）
    if element == "H":
        return "H"
    
    # 默认
    return element


def calculate_gasteiger_charge(atom_name: str, res_name: str, element: str) -> float:
    """
    计算 Gasteiger 部分电荷（简化版）
    实际应使用 Open Babel 的电荷计算
    """
    # 简化：基于残基和原子的常见电荷
    res_name = res_name.upper()
    atom_name = atom_name.upper()
    element = element.upper()
    
    # 带电残基
    if res_name == "ARG":
        if atom_name in ["CZ"]:
            return 0.64
        if atom_name in ["NE"]:
            return -0.33
        if atom_name in ["NH1", "NH2"]:
            return 0.36
    
    if res_name == "LYS":
        if atom_name == "NZ":
            return -0.30
        if atom_name.startswith("HZ"):
            return 0.33
    
    if res_name == "ASP":
        if atom_name in ["CG"]:
            return 0.62
        if atom_name in ["OD1", "OD2"]:
            return -0.76
    
    if res_name == "GLU":
        if atom_name in ["CD"]:
            return 0.62
        if atom_name in ["OE1", "OE2"]:
            return -0.76
    
    if res_name in ["HIP", "HIS"]:
        if atom_name in ["CG", "CD2", "CE1"]:
            return 0.20
        if atom_name == "ND1":
            return -0.30
        if atom_name == "NE2":
            return 0.36
    
    # 默认中性
    return 0.0


def format_pdbqt_atom(atom: dict) -> str:
    """
    格式化 PDBQT ATOM 行
    
    列定义：
    1-6:   记录名
    7-11:  原子序号（右对齐）
    12:    空格
    13-16: 原子名称（左对齐）
    17:    空格
    18-20: 残基名（右对齐）
    21:    空格
    22:    链标识
    23-26: 残基序号（右对齐）
    27-30: 空格
    31-38: X 坐标（右对齐，3位小数）
    39-46: Y 坐标（右对齐，3位小数）
    47-54: Z 坐标（右对齐，3位小数）
    55-60: 占据率
    61-66: 温度因子
    67-76: 电荷（右对齐，有符号）
    77-78: 原子类型
    """
    # Vina 1.1.2 格式要求：
    # 1. 原子名称严格4字符左对齐（13-16列）
    # 2. 氢原子类型使用 HD（极性氢）
    
    atom_name = atom['atom_name']
    atom_type = atom['atom_type']
    
    # 从原子名推断元素
    element = atom_name[0] if atom_name else ""
    if len(atom_name) > 1 and atom_name[0].isdigit():
        element = atom_name[1]
    elif len(atom_name) > 1 and atom_name[:2] in ["CL", "BR", "FE", "ZN", "CA", "MG", "MN", "NA"]:
        element = atom_name[:2]
    
    # 修复原子名称格式：确保严格4字符左对齐（13-16列）
    # Vina 1.1.2 要求原子名称必须占满4列
    if len(atom_name) == 1:
        atom_name = atom_name + "   "  # N -> "N   " (1+3=4)
    elif len(atom_name) == 2:
        atom_name = atom_name + "  "   # CA -> "CA  " (2+2=4)
    elif len(atom_name) == 3:
        atom_name = atom_name + " "    # H3 -> "H3 " (3+1=4)
    # len=4 保持不变
    elif len(atom_name) > 4:
        atom_name = atom_name[:4]      # 截断到4字符
    
    # Vina 1.1.2：氢原子使用 H（单字符）
    # 注意：Vina 1.1.2 使用单字符原子类型，HD 会导致解析错误
    if element.upper() == 'H':
        atom_type = 'H'  # 使用单字符 H
    
    line = f"{atom['record']:<6}"  # 1-6
    line += f"{atom['atom_serial']:>5}"  # 7-11
    line += " "  # 12
    line += f"{atom_name:<4}"  # 13-16（左对齐4字符）
    line += " "  # 17
    line += f"{atom['res_name']:>3}"  # 18-20
    line += " "  # 21
    line += f"{atom['chain_id']:<1}"  # 22
    line += f"{atom['res_seq']:>4}"  # 23-26
    line += "    "  # 27-30
    line += f"{atom['x']:>8.3f}"  # 31-38
    line += f"{atom['y']:>8.3f}"  # 39-46
    line += f"{atom['z']:>8.3f}"  # 47-54
    line += f"{atom['occupancy']:>6.2f}"  # 55-60
    line += f"{atom['temp_factor']:>6.2f}"  # 61-66
    line += f"{atom['charge']:>10.3f}"  # 67-76
    # Vina 1.1.2: 原子类型需要右对齐，且后面有空格到行尾
    line += f"{atom_type:>2}"  # 77-78（右对齐）
    line += "\n"  # 行尾换行
    
    return line


def pdb_to_pdbqt(input_pdb: Path, output_pdbqt: Path, remove_water: bool = True):
    """
    将 PDB 转换为 PDBQT 格式
    
    Args:
        input_pdb: 输入 PDB 文件
        output_pdbqt: 输出 PDBQT 文件
        remove_water: 是否移除水分子
    """
    atoms = []
    
    with open(input_pdb, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # 检查是否为水分子
                res_name = line[17:20].strip()
                if remove_water and res_name in ["HOH", "WAT", "H2O", "TIP", "TIP3"]:
                    continue
                
                atom = parse_pdb_line(line)
                if atom:
                    atoms.append(atom)
            elif line.startswith("TER") or line.startswith("END"):
                # Vina 1.1.2: 跳过 TER 和 END 标签
                pass
    
    # 重新编号原子
    for i, atom in enumerate(atoms, 1):
        atom["atom_serial"] = i
    
    # 写入 PDBQT
    # 注意：对于刚性受体，不使用 ROOT/ENDROOT 标签
    # 这些标签只用于柔性残基
    output_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_pdbqt, 'w') as f:
        f.write("REMARK   4      COMPLIES WITH FORMAT V. 2.0\n")
        f.write("REMARK  Generated by MCTS-Peptide Pipeline\n")
        f.write("REMARK  AutoDock4 atom types assigned\n")
        # 刚性受体不需要 ROOT/ENDROOT
        
        for atom in atoms:
            f.write(format_pdbqt_atom(atom))
        
        # Vina 1.1.2: 不需要 TER 和 END 标签
    
    print(f"  ✓ PDBQT: {output_pdbqt} ({len(atoms)} atoms)")


def create_vina_config(pocket_center: Tuple[float, float, float], 
                       box_size: Tuple[float, float, float],
                       output_config: Path,
                       receptor_pdbqt: Path = None):
    """
    创建 Vina 对接配置文件
    
    Args:
        pocket_center: 口袋中心 (x, y, z)
        box_size: 盒子尺寸 (x, y, z)
        output_config: 输出配置文件路径
        receptor_pdbqt: 受体 PDBQT 路径（可选，用于记录）
    """
    config_content = f"""# Vina 对接配置
# Auto-generated by MCTS-Peptide Pipeline

# 受体
{"receptor = " + str(receptor_pdbqt) if receptor_pdbqt else "# receptor = path/to/receptor.pdbqt"}

# 搜索空间（对接盒子）
center_x = {pocket_center[0]:.3f}
center_y = {pocket_center[1]:.3f}
center_z = {pocket_center[2]:.3f}

size_x = {box_size[0]:.1f}
size_y = {box_size[1]:.1f}
size_z = {box_size[2]:.1f}

# 可选参数
# exhaustiveness = 32
# num_modes = 9
# energy_range = 4
"""
    
    output_config.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_config, 'w') as f:
        f.write(config_content)
    
    print(f"  ✓ Vina 配置: {output_config}")


def prepare_for_vina(target_name: str):
    """
    主入口：准备 Vina 受体文件
    
    Args:
        target_name: 靶点名称
    """
    import json
    
    print(f"\n{'='*60}")
    print(f"准备 Vina 受体: {target_name}")
    print(f"{'='*60}")
    
    dirs = get_target_dirs(target_name)
    
    # 输入文件
    cleaned_pdb = dirs["cleaned"] / "cleaned.pdb"
    pocket_json = dirs["pocket"] / "pocket.json"
    
    if not cleaned_pdb.exists():
        raise FileNotFoundError(f"请先运行 pdb_cleaner.py: {cleaned_pdb}")
    
    if not pocket_json.exists():
        raise FileNotFoundError(f"请先运行 pdb_to_pockets.py: {pocket_json}")
    
    # 读取口袋信息
    with open(pocket_json, 'r') as f:
        pocket_data = json.load(f)
    
    pocket_center = pocket_data["best_pocket"]["center"]
    print(f"  口袋中心: ({pocket_center[0]:.3f}, {pocket_center[1]:.3f}, {pocket_center[2]:.3f})")
    
    # 1. 转换为 PDBQT
    output_pdbqt = dirs["vina"] / "vina-receptor.pdbqt"
    pdb_to_pdbqt(cleaned_pdb, output_pdbqt, remove_water=True)
    
    # 2. 创建 Vina 配置
    output_config = dirs["vina"] / "vina_config.txt"
    create_vina_config(
        pocket_center=tuple(pocket_center),
        box_size=VINA_BOX_SIZE,
        output_config=output_config,
        receptor_pdbqt=output_pdbqt
    )
    
    print(f"\n✓ Vina 受体准备完成")
    return output_pdbqt


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="准备 Vina 受体文件")
    parser.add_argument("target", help="靶点名称（如 1LYZ）")
    
    args = parser.parse_args()
    
    try:
        prepare_for_vina(args.target)
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
