#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESM-IF (Inverse Folding) 模型接口
用于预测肽序列中可变位置的氨基酸概率分布
"""

import os
import sys
import re
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# ESM-IF 模型路径
ESMIF_MODEL_PATH = Path("/mnt/d/code/AA/models/esm_if1_gvp4_t16_142M_UR50.pt")
ESMIF_MODEL_URL = "https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt"

# 标准氨基酸（用于输出概率分布）
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # 20种标准氨基酸
AA_DICT = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


@dataclass
class ESMIFPrediction:
    """ESM-IF 预测结果"""
    position: int              # 序列位置（0-based）
    probabilities: Dict[str, float]  # 每个氨基酸的概率
    entropy: float             # 预测熵（不确定性度量）
    
    def get_top_k(self, k: int = 5) -> List[Tuple[str, float]]:
        """获取概率最高的 k 个氨基酸"""
        sorted_aas = sorted(self.probabilities.items(), key=lambda x: x[1], reverse=True)
        return sorted_aas[:k]
    
    def get_best_aa(self) -> str:
        """获取概率最高的氨基酸"""
        return max(self.probabilities.items(), key=lambda x: x[1])[0]


class ESMIFPredictor:
    """ESM-IF 预测器"""
    
    def __init__(self, model_path: Optional[Path] = None, device: str = "auto"):
        """
        初始化 ESM-IF 预测器
        
        Args:
            model_path: 模型文件路径（默认使用 ESMIF_MODEL_PATH）
            device: 计算设备（"cuda", "cpu", 或 "auto"）
        """
        self.model_path = model_path or ESMIF_MODEL_PATH
        self.device = self._get_device(device)
        self.model = None
        self.alphabet = None
        
        # 加载模型
        self._load_model()
    
    def _get_device(self, device: str) -> torch.device:
        """确定计算设备"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _download_model(self):
        """自动下载 ESM-IF 模型"""
        import urllib.request
        import ssl
        
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"模型文件不存在，开始下载...")
        print(f"下载地址: {ESMIF_MODEL_URL}")
        print(f"保存位置: {self.model_path}")
        print(f"模型大小: ~142M 参数 (约 500MB)")
        print()
        
        # 创建 SSL 上下文（处理某些系统的证书问题）
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # 下载进度回调
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            mb = downloaded / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            print(f"\r下载进度: {percent:.1f}% ({mb:.1f} / {total_mb:.1f} MB)", end='', flush=True)
        
        try:
            urllib.request.urlretrieve(
                ESMIF_MODEL_URL, 
                self.model_path,
                reporthook=download_progress
            )
            print()  # 换行
            print(f"✓ 模型下载成功: {self.model_path}")
            print(f"  文件大小: {self.model_path.stat().st_size / 1024 / 1024:.1f} MB")
            return True
        except Exception as e:
            print(f"\n✗ 模型下载失败: {e}")
            return False
    
    def _load_model(self):
        """加载 ESM-IF 模型"""
        try:
            import esm
            
            # 检查模型文件是否存在，不存在则自动下载
            if not self.model_path.exists():
                print(f"ESM-IF 模型文件不存在: {self.model_path}")
                success = self._download_model()
                if not success:
                    raise RuntimeError(
                        "模型下载失败。请手动下载:\n"
                        f"wget {ESMIF_MODEL_URL} -O {self.model_path}"
                    )
            
            print(f"加载 ESM-IF 模型: {self.model_path}")
            print(f"使用设备: {self.device}")
            
            # 加载模型和字母表
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(
                str(self.model_path)
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ 模型加载成功")
            print(f"  模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except ImportError:
            raise RuntimeError(
                "ESM 库未安装。请安装: pip install fair-esm"
            )
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    def parse_pocket_structure(self, pocket_pdb: Path) -> Tuple[str, torch.Tensor]:
        """
        解析口袋结构 PDB 文件，提取 N, CA, C 坐标
        
        Args:
            pocket_pdb: 口袋结构 PDB 文件路径
        
        Returns:
            (坐标字符串, 坐标张量 [L, 3, 3])
        """
        if not pocket_pdb.exists():
            raise FileNotFoundError(f"口袋结构文件不存在: {pocket_pdb}")
        
        # 读取 PDB 文件中的坐标，按残基组织
        residues = {}  # {res_seq: {'N': [x,y,z], 'CA': [x,y,z], 'C': [x,y,z]}}
        
        with open(pocket_pdb, 'r') as f:
            for line in f:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                
                atom_name = line[12:16].strip()
                res_seq = line[22:26].strip()
                
                # 只提取 N, CA, C 原子
                if atom_name not in ['N', 'CA', 'C']:
                    continue
                
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    if res_seq not in residues:
                        residues[res_seq] = {}
                    
                    residues[res_seq][atom_name] = [x, y, z]
                    
                except (ValueError, IndexError):
                    continue
        
        if not residues:
            raise RuntimeError(f"无法从 {pocket_pdb} 解析 N/CA/C 坐标")
        
        # 按残基序号排序
        sorted_residues = sorted(residues.items(), key=lambda x: int(x[0]))
        
        # 构建坐标列表 [L, 3, 3]
        coords_list = []
        for res_seq, atoms in sorted_residues:
            if 'N' in atoms and 'CA' in atoms and 'C' in atoms:
                coords_list.append([
                    atoms['N'],   # N 坐标
                    atoms['CA'],  # CA 坐标
                    atoms['C']    # C 坐标
                ])
        
        if not coords_list:
            raise RuntimeError(f"没有完整的 N-CA-C 残基")
        
        # 转换为张量
        coords_tensor = torch.tensor(coords_list, dtype=torch.float32)
        
        print(f"  从口袋提取了 {len(coords_list)} 个残基的坐标")
        
        return "", coords_tensor
    
    def predict_sequence_probabilities(
        self,
        pocket_pdb: Path,
        partial_sequence: str,
        temperature: float = 1.0
    ) -> List[ESMIFPrediction]:
        """
        预测可变位置的氨基酸概率分布
        
        Args:
            pocket_pdb: 口袋结构 PDB 文件路径
            partial_sequence: 部分序列（含 x 或 _ 表示未知位置）
            temperature: 采样温度（控制多样性）
        
        Returns:
            每个可变位置的预测结果列表
        """
        # 即使没有真实模型，也可以使用模拟模式
        
        print(f"\n{'='*60}")
        print(f"ESM-IF 序列预测")
        print(f"{'='*60}")
        print(f"输入序列: {partial_sequence}")
        print(f"序列长度: {len(partial_sequence)}")
        
        # 解析口袋结构
        print(f"\n解析口袋结构: {pocket_pdb}")
        coord_str, coords = self.parse_pocket_structure(pocket_pdb)
        print(f"  口袋原子数: {len(coords)}")
        
        # 识别可变位置
        variable_positions = []
        for i, aa in enumerate(partial_sequence):
            if aa in ['x', 'X', '_']:
                variable_positions.append(i)
        
        print(f"可变位置: {variable_positions}")
        
        if not variable_positions:
            print("警告: 序列中没有可变位置")
            return []
        
        # 准备输入（使用占位符序列）
        # ESM-IF 需要完整的 backbone 坐标，这里简化处理
        placeholder_seq = partial_sequence.replace('x', 'A').replace('X', 'A').replace('_', 'A')
        
        # 使用 ESM-IF 模型进行推理
        if self.model is None:
            raise RuntimeError("模型未加载，无法预测")
        
        predictions = self._esmif_predict(
            pocket_pdb,
            coord_str,
            coords,
            variable_positions, 
            partial_sequence,
            temperature
        )
        
        # 打印预测结果
        print(f"\n预测结果:")
        print("-" * 60)
        for pred in predictions:
            top5 = pred.get_top_k(5)
            top5_str = ", ".join([f"{aa}({prob:.3f})" for aa, prob in top5])
            print(f"位置 {pred.position:2d}: {top5_str}")
        print(f"{'='*60}\n")
        
        return predictions
    
    def _esmif_predict(
        self,
        pocket_pdb: Path,
        coord_str: str,
        coords: torch.Tensor,
        positions: List[int],
        partial_sequence: str,
        temperature: float
    ) -> List[ESMIFPrediction]:
        """
        使用 ESM-IF 模型进行预测
        
        ESM-IF 输入格式:
        - coords: [L, 3, 3] 张量，每个残基的 N, CA, C 坐标
        - seq: 部分序列，用 <mask> 表示未知位置
        
        输出:
        - 每个掩码位置的概率分布
        """
        # 构建 ESM-IF 输入序列
        # 将 x/_ 替换为 <mask>
        esmif_seq = partial_sequence.replace('x', '<mask>').replace('X', '<mask>').replace('_', '<mask>')
        
        # 准备坐标数据
        L = len(partial_sequence)
        
        # 使用从口袋结构提取的坐标
        # coords 是从 parse_pocket_structure 传入的 [num_residues, 3, 3] 张量
        
        # 如果口袋残基数与序列长度不匹配，需要处理
        pocket_L = coords.shape[0]
        
        if pocket_L != L:
            print(f"  警告: 口袋残基数 ({pocket_L}) 与序列长度 ({L}) 不匹配")
            # 简化处理：截取或填充
            if pocket_L > L:
                coords_tensor = coords[:L].to(self.device)
            else:
                # 填充到序列长度
                padding = torch.zeros(L - pocket_L, 3, 3, dtype=torch.float32)
                coords_tensor = torch.cat([coords, padding], dim=0).to(self.device)
        else:
            coords_tensor = coords.to(self.device)
        
        # 添加置信度（全 1 表示高置信度）
        confidence = torch.ones(L, device=self.device)
        
        # 使用 ESM-IF 进行预测
        with torch.no_grad():
            # ESM-IF 推理
            # 使用多次采样来估计概率分布
            
            # 多次采样以估计概率分布
            num_samples_estimate = 100
            sample_counts = torch.zeros(L, len(self.alphabet), device=self.device)
            
            for _ in range(num_samples_estimate):
                # ESM-IF sample 直接返回字符串
                sampled_seq = self.model.sample(coords_tensor, confidence, temperature=temperature)
                
                # 将采样的序列转换为 token
                for pos, aa in enumerate(sampled_seq):
                    if pos < L:
                        tok_idx = self.alphabet.get_idx(aa)
                        if tok_idx < len(self.alphabet):
                            sample_counts[pos, tok_idx] += 1
            
            # 计算概率分布
            probs = sample_counts / num_samples_estimate
        
        # 提取可变位置的预测结果
        predictions = []
        
        for pos in positions:
            # 获取该位置的概率分布
            pos_probs = probs[pos].cpu().numpy()
            
            # 构建概率字典（只包含标准氨基酸）
            prob_dict = {}
            for i, aa in enumerate(AMINO_ACIDS):
                if i < len(pos_probs):
                    prob_dict[aa] = float(pos_probs[i])
                else:
                    prob_dict[aa] = 0.0
            
            # 归一化概率
            total_prob = sum(prob_dict.values())
            if total_prob > 0:
                prob_dict = {k: v / total_prob for k, v in prob_dict.items()}
            
            # 计算熵
            probs_array = np.array(list(prob_dict.values()))
            entropy = -np.sum(probs_array * np.log(probs_array + 1e-10))
            
            predictions.append(ESMIFPrediction(
                position=pos,
                probabilities=prob_dict,
                entropy=float(entropy)
            ))
        
        return predictions
    
    def sample_sequence(
        self,
        pocket_pdb: Path,
        partial_sequence: str,
        num_samples: int = 1,
        temperature: float = 1.0
    ) -> List[str]:
        """
        从预测分布中采样完整序列
        
        Args:
            pocket_pdb: 口袋结构 PDB 文件路径
            partial_sequence: 部分序列（含 x 或 _）
            num_samples: 采样数量
            temperature: 采样温度
        
        Returns:
            采样的完整序列列表
        """
        predictions = self.predict_sequence_probabilities(
            pocket_pdb, partial_sequence, temperature
        )
        
        if not predictions:
            # 没有可变位置，返回原序列
            return [partial_sequence.replace('x', 'A').replace('X', 'A').replace('_', 'A')]
        
        sequences = []
        for _ in range(num_samples):
            seq_list = list(partial_sequence)
            for pred in predictions:
                # 按概率采样
                aas = list(pred.probabilities.keys())
                probs = list(pred.probabilities.values())
                sampled_aa = np.random.choice(aas, p=probs)
                seq_list[pred.position] = sampled_aa
            sequences.append(''.join(seq_list))
        
        return sequences


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ESM-IF 序列预测')
    parser.add_argument('-p', '--pocket', type=Path, required=True,
                        help='口袋结构 PDB 文件路径')
    parser.add_argument('-s', '--sequence', type=str, required=True,
                        help='部分序列（含 x 或 _ 表示未知位置）')
    parser.add_argument('-n', '--num-samples', type=int, default=1,
                        help='采样数量（默认: 1）')
    parser.add_argument('-t', '--temperature', type=float, default=1.0,
                        help='采样温度（默认: 1.0）')
    parser.add_argument('--model', type=Path, default=None,
                        help='模型路径（可选）')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='计算设备（默认: auto）')
    
    args = parser.parse_args()
    
    try:
        # 初始化预测器
        predictor = ESMIFPredictor(
            model_path=args.model,
            device=args.device
        )
        
        # 预测概率分布
        predictions = predictor.predict_sequence_probabilities(
            pocket_pdb=args.pocket,
            partial_sequence=args.sequence,
            temperature=args.temperature
        )
        
        # 采样序列
        if args.num_samples > 0:
            sequences = predictor.sample_sequence(
                pocket_pdb=args.pocket,
                partial_sequence=args.sequence,
                num_samples=args.num_samples,
                temperature=args.temperature
            )
            
            print(f"\n采样序列 ({args.num_samples} 个):")
            for i, seq in enumerate(sequences, 1):
                print(f"  {i}: {seq}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
