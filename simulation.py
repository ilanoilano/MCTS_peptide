#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS Simulation 模块
结合能评估 + 物理启发式 + 蛋白酶稳定性
"""

import os
import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import config
from expansion import PeptideState


@dataclass
class SimulationResult:
    """模拟结果"""
    binding_energy: float
    molecular_weight: float
    hydrophobicity: float
    net_charge: float
    trypsin_risk: float
    chymotrypsin_risk: float
    total_score: float
    details: Dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class VinaScorer:
    """AutoDock Vina 结合能评分器"""
    
    def __init__(self, vina_path: str = "vina"):
        self.vina_path = vina_path
    
    def score(self, peptide_pdbqt: Path, receptor_pdbqt: Path, vina_config: Path) -> float:
        """
        计算结合能
        
        Raises:
            RuntimeError: Vina不可用、文件不存在、执行失败或解析失败时抛出
        """
        import tempfile
        import shutil
        
        vina_exe = shutil.which(self.vina_path)
        if vina_exe is None:
            if Path(self.vina_path).exists():
                vina_exe = self.vina_path
            else:
                raise RuntimeError(f"Vina 未找到 ({self.vina_path})")
        
        if not peptide_pdbqt.exists():
            raise RuntimeError(f"配体文件不存在: {peptide_pdbqt}")
        if not receptor_pdbqt.exists():
            raise RuntimeError(f"受体文件不存在: {receptor_pdbqt}")
        if not vina_config.exists():
            raise RuntimeError(f"配置文件不存在: {vina_config}")
        
        # 打印调试信息
        print(f"  [Vina] 配体: {peptide_pdbqt} ({peptide_pdbqt.stat().st_size} bytes)")
        print(f"  [Vina] 受体: {receptor_pdbqt} ({receptor_pdbqt.stat().st_size} bytes)")
        print(f"  [Vina] 配置: {vina_config}")
        
        with tempfile.NamedTemporaryFile(suffix='.pdbqt', delete=False) as output_file:
            output_pdbqt = output_file.name
        
        cmd = [
            vina_exe,
            '--receptor', str(receptor_pdbqt),
            '--ligand', str(peptide_pdbqt),
            '--config', str(vina_config),
            '--out', output_pdbqt
        ]
        
        print(f"  [Vina] 命令: {' '.join(cmd)}")
        print(f"  [Vina] 开始对接 (超时: 60秒)...")
        
        try:
            # 缩短超时时间到 60 秒，避免长时间卡死
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            print(f"  [Vina] 返回码: {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"  [Vina] 超时!")
            # 清理临时文件
            try:
                os.unlink(output_pdbqt)
            except:
                pass
            raise RuntimeError("Vina 对接超时 (60秒)")
        except Exception as e:
            print(f"  [Vina] 异常: {e}")
            # 清理临时文件
            try:
                os.unlink(output_pdbqt)
            except:
                pass
            raise RuntimeError(f"Vina 执行失败: {e}")
        
        # 打印 Vina 输出
        if result.stdout:
            print(f"  [Vina] stdout:\n{result.stdout[:500]}")
        if result.stderr:
            print(f"  [Vina] stderr:\n{result.stderr[:500]}")
        
        if result.returncode != 0:
            error_msg = result.stderr[:500] if result.stderr else "无错误输出"
            # 清理临时文件
            try:
                os.unlink(output_pdbqt)
            except:
                pass
            raise RuntimeError(f"Vina 返回错误码 {result.returncode}: {error_msg}")
        
        # 解析结合能
        if result.stdout:
            for line in result.stdout.split('\n'):
                if 'REMARK VINA RESULT:' in line:
                    try:
                        parts = line.split()
                        energy = float(parts[3])
                        print(f"  [Vina] 结合能: {energy} kcal/mol")
                        # 清理临时文件
                        try:
                            os.unlink(output_pdbqt)
                        except:
                            pass
                        return energy
                    except (ValueError, IndexError) as e:
                        raise RuntimeError(f"无法解析 Vina 输出行: {line}, 错误: {e}")
        
        # 清理临时文件
        try:
            os.unlink(output_pdbqt)
        except:
            pass
        
        raise RuntimeError("无法从 Vina 输出中解析结合能")


class PhysicsScorer:
    """物理启发式评分器"""
    
    HYDROPATHY = config.HYDROPATHY
    CHARGE = config.AA_CHARGE
    MOLECULAR_WEIGHT = config.AA_MOLECULAR_WEIGHT
    
    def calculate_hydrophobicity(self, sequence: str) -> float:
        values = [self.HYDROPATHY.get(aa, 0) for aa in sequence]
        return sum(values) / max(len(values), 1)
    
    def calculate_net_charge(self, sequence: str) -> float:
        return sum(self.CHARGE.get(aa, 0) for aa in sequence)
    
    def calculate_molecular_weight(self, sequence: str) -> float:
        residue_weight = sum(self.MOLECULAR_WEIGHT.get(aa, 110) for aa in sequence)
        water_weight = 18.015 * (len(sequence) - 1)
        return residue_weight - water_weight
    
    def score(self, sequence: str) -> Dict[str, float]:
        mw = self.calculate_molecular_weight(sequence)
        hydro = self.calculate_hydrophobicity(sequence)
        charge = self.calculate_net_charge(sequence)
        mw_min, mw_max = config.MOLECULAR_WEIGHT_RANGE
        mw_score = 1.0 if mw_min <= mw <= mw_max else 0.0
        
        return {
            'molecular_weight': mw,
            'hydrophobicity': hydro,
            'net_charge': charge,
            'mw_in_range': mw_score
        }


class ProteaseScorer:
    """蛋白酶切割位点预测器"""
    
    TRYPSIN_PATTERN = re.compile(r'(?<=[KR])(?!P)')
    CHYMOTRYPSIN_PATTERN = re.compile(r'(?<=[FWY])(?!P)')
    
    def predict_trypsin_risk(self, sequence: str) -> float:
        matches = len(self.TRYPSIN_PATTERN.findall(sequence))
        return min(matches / 5.0, 1.0)
    
    def predict_chymotrypsin_risk(self, sequence: str) -> float:
        matches = len(self.CHYMOTRYPSIN_PATTERN.findall(sequence))
        return min(matches / 5.0, 1.0)
    
    def score(self, sequence: str) -> Dict[str, float]:
        trypsin_risk = self.predict_trypsin_risk(sequence)
        chymo_risk = self.predict_chymotrypsin_risk(sequence)
        stability = 1.0 - (trypsin_risk + chymo_risk) / 2.0
        
        return {
            'trypsin_risk': trypsin_risk,
            'chymotrypsin_risk': chymo_risk,
            'stability_score': stability
        }


class SimulationEngine:
    """MCTS 模拟引擎"""
    
    def __init__(self, target_name: str, weights: Optional[Dict[str, float]] = None):
        self.target_name = target_name
        self.vina_scorer = VinaScorer(config.TOOLS.get("vina", "vina"))
        self.physics_scorer = PhysicsScorer()
        self.protease_scorer = ProteaseScorer()
        self.weights = weights or config.SCORING_WEIGHTS
    
    def simulate(self, state: PeptideState) -> SimulationResult:
        """模拟评估"""
        sequence = state.sequence
        
        physics = self.physics_scorer.score(sequence)
        stability = self.protease_scorer.score(sequence)
        binding_energy = self._run_vina_docking(state)
        
        total_score = self._calculate_total_score(
            binding_energy=binding_energy,
            physics=physics,
            stability=stability
        )
        
        return SimulationResult(
            binding_energy=binding_energy,
            molecular_weight=physics['molecular_weight'],
            hydrophobicity=physics['hydrophobicity'],
            net_charge=physics['net_charge'],
            trypsin_risk=stability['trypsin_risk'],
            chymotrypsin_risk=stability['chymotrypsin_risk'],
            total_score=total_score,
            details={'physics': physics, 'stability': stability}
        )
    
    def _run_vina_docking(self, state: PeptideState) -> float:
        """运行 Vina 分子对接"""
        import tempfile
        from pathlib import Path
        
        peptide_pdbqt_path = self._generate_peptide_pdbqt(state)
        if peptide_pdbqt_path is None:
            raise RuntimeError("生成肽 PDBQT 文件失败")
        
        print(f"  [Docking] PDBQT 生成成功: {peptide_pdbqt_path}")
        
        dirs = config.get_target_dirs(self.target_name)
        receptor_pdbqt = dirs["vina"] / "vina-receptor.pdbqt"
        vina_config = dirs["vina"] / "vina_config.txt"
        
        if not receptor_pdbqt.exists():
            # 清理临时文件
            try:
                os.unlink(peptide_pdbqt_path)
            except:
                pass
            raise RuntimeError(f"Vina 受体文件不存在: {receptor_pdbqt}")
        if not vina_config.exists():
            # 清理临时文件
            try:
                os.unlink(peptide_pdbqt_path)
            except:
                pass
            raise RuntimeError(f"Vina 配置文件不存在: {vina_config}")
        
        try:
            print(f"  [Docking] 开始 Vina 对接...")
            energy = self.vina_scorer.score(
                peptide_pdbqt=Path(peptide_pdbqt_path),
                receptor_pdbqt=receptor_pdbqt,
                vina_config=vina_config
            )
            print(f"  [Docking] Vina 对接完成，能量: {energy}")
            return energy
        finally:
            # 确保清理临时 PDBQT 文件
            try:
                if os.path.exists(peptide_pdbqt_path):
                    os.unlink(peptide_pdbqt_path)
                    print(f"  [Docking] 清理临时文件: {peptide_pdbqt_path}")
            except Exception as e:
                print(f"  [Docking] 清理临时文件失败: {e}")
    
    def _generate_peptide_pdbqt(self, state: PeptideState) -> Optional[str]:
        """从肽序列生成 PDBQT 文件（使用独立 Python 进程隔离 OpenBabel）"""
        import os
        import random
        
        sequence = state.sequence
        
        # 【关键修复】如果序列包含 _ 或 x，先用随机氨基酸填充
        if '_' in sequence or 'x' in sequence:
            sequence = self._fill_placeholder_positions(sequence, state)
        
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            mol = self._build_peptide_mol(sequence, state)
            if mol is None:
                return None
            
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
            AllChem.MMFFOptimizeMolecule(mol)
            
            # 保存为临时 PDB 文件
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as pdb_file:
                pdb_path = pdb_file.name
            Chem.MolToPDBFile(mol, pdb_path)
            
            # 创建临时 PDBQT 文件
            with tempfile.NamedTemporaryFile(suffix='.pdbqt', delete=False) as pdbqt_file:
                pdbqt_path = pdbqt_file.name
            
            # 使用独立 Python 进程执行 OpenBabel 转换（避免与 RDKit 冲突）
            # 找到 obabel_convert.py 脚本
            script_dir = Path(__file__).parent
            obabel_script = script_dir / "obabel_convert.py"
            
            if not obabel_script.exists():
                raise RuntimeError(f"找不到 obabel_convert.py 脚本: {obabel_script}")
            
            # 在独立进程中运行转换脚本（使用 python3，避免 Python 2）
            cmd = [
                "python3",
                str(obabel_script),
                pdb_path,
                pdbqt_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # 调试输出
            print(f"  OpenBabel 返回码: {result.returncode}")
            if result.stdout:
                print(f"  OpenBabel 输出: {result.stdout.strip()}")
            if result.stderr:
                print(f"  OpenBabel 错误: {result.stderr[:200]}")
            
            # 清理临时 PDB 文件
            try:
                os.unlink(pdb_path)
            except:
                pass
            
            if result.returncode != 0:
                raise RuntimeError(f"OpenBabel 转换失败: {result.stderr}")
            
            # 检查 PDBQT 文件是否生成
            if not os.path.exists(pdbqt_path):
                raise RuntimeError("OpenBabel 未生成 PDBQT 文件")
            
            file_size = os.path.getsize(pdbqt_path)
            print(f"  PDBQT 文件大小: {file_size} bytes")
            
            if file_size == 0:
                raise RuntimeError("OpenBabel 生成了空的 PDBQT 文件")
            
            # 验证 PDBQT 格式：必须包含 ROOT 标签
            with open(pdbqt_path, 'r') as f:
                content = f.read()
            
            if 'ROOT' not in content:
                print(f"  [WARNING] PDBQT 文件缺少 ROOT 标签，内容:\n{content[:500]}")
                raise RuntimeError("PDBQT 文件格式错误：缺少 ROOT 标签")
            
            if 'ATOM' not in content:
                print(f"  [WARNING] PDBQT 文件缺少 ATOM 行，内容:\n{content[:500]}")
                raise RuntimeError("PDBQT 文件格式错误：缺少 ATOM 行")
            
            print(f"  [OK] PDBQT 文件验证通过 (包含 ROOT 和 ATOM)")
            
            return pdbqt_path
                
        except ImportError as e:
            raise RuntimeError(f"缺少必要的库: {e}。请安装: pip install rdkit")
        except Exception as e:
            raise RuntimeError(f"生成肽 PDBQT 失败: {e}")
    
    def _fill_placeholder_positions(self, sequence: str, state: PeptideState) -> str:
        """
        填充序列中的占位符（_ 或 x）为有效氨基酸
        
        关键修复：只填充 MCTS 未到达的位置
        - MCTS 已确定的位置保持原样
        - 只随机填充剩余未知位置
        
        Args:
            sequence: 部分填充的序列（可能含 _ 或 x）
            state: PeptideState 对象，包含当前 MCTS 节点信息
        """
        seq_list = list(sequence)
        
        # 获取当前序列中已确定的氨基酸位置
        # 已确定的位置：不是 _ 也不是 x
        determined_positions = set()
        for i, aa in enumerate(seq_list):
            if aa != '_' and aa != 'x':
                determined_positions.add(i)
        
        # 只填充未确定的位置
        for i, aa in enumerate(seq_list):
            if aa == '_' or aa == 'x':
                # 获取该位置允许的氨基酸
                allowed_aas = config.VARIABLE_AMINO_ACIDS.get(i, config.ALLOWED_AMINO_ACIDS)
                # 随机选择一个
                seq_list[i] = random.choice(allowed_aas)
        
        return ''.join(seq_list)
    
    def _build_peptide_mol(self, sequence: str, state: PeptideState):
        """构建肽分子对象 - 使用 SMILES 连接方式"""
        try:
            from rdkit import Chem
            
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
            
        except Exception as e:
            raise RuntimeError(f"构建肽分子失败: {e}")
    
    def _calculate_total_score(self, binding_energy: float, physics: Dict, stability: Dict) -> float:
        """计算综合评分"""
        w = self.weights
        binding_score = max(0, (-binding_energy - 5) / 10.0)
        mw_score = physics.get('mw_in_range', 0.0)
        stability_score = stability.get('stability_score', 0.0)
        
        total = (
            w.get('vina_score', 0.5) * binding_score +
            w.get('molecular_weight', 0.1) * mw_score +
            w.get('protease_stability', 0.2) * stability_score +
            w.get('hydrophobicity', 0.1) * max(0, physics['hydrophobicity'] / 5.0) +
            w.get('charge_balance', 0.1) * (1.0 - abs(physics['net_charge']) / 10.0)
        )
        
        return total
