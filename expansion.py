# -*- coding: utf-8 -*-
"""
MCTS Expansion 模块
序列延伸 + 拓扑异构体选择
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import config
from selection import MCTSNode


@dataclass
class PeptideState:
    """肽状态 - 包含序列和拓扑信息"""
    sequence: str
    disulfide_bonds: List[Tuple[int, int]]
    crosslinker: Optional[str] = None
    crosslinker_positions: Optional[List[int]] = None
    
    def is_valid(self) -> bool:
        """检查状态是否有效"""
        # 检查二硫键配对
        for i, j in self.disulfide_bonds:
            if i >= len(self.sequence) or j >= len(self.sequence):
                return False
            if self.sequence[i] != 'C' or self.sequence[j] != 'C':
                return False
        
        # 检查交联剂配置
        if self.crosslinker:
            if not TopologyManager.validate_crosslinker(
                self.sequence, self.crosslinker, self.crosslinker_positions or []
            ):
                return False
        
        return True
    
    def get_ring_count(self) -> int:
        """获取环数"""
        if self.crosslinker and self.crosslinker_positions:
            crosslinker_rings = {'TBMB': 1, 'TATA': 1, 'TBAB': 2}
            return crosslinker_rings.get(self.crosslinker, 1)
        return len(self.disulfide_bonds)
    
    def get_all_cys_positions(self) -> List[int]:
        """获取序列中所有Cys的位置"""
        return [i for i, aa in enumerate(self.sequence) if aa == 'C']


class ExpansionEngine:
    """MCTS 扩展引擎"""
    
    def __init__(
        self,
        template: str,
        fixed_positions: Dict[int, str],
        variable_amino_acids: Dict[int, List[str]],
        disulfide_bonds: Optional[List[Tuple[int, int]]] = None,
        crosslinker: Optional[str] = None,
        crosslinker_positions: Optional[List[int]] = None
    ):
        self.template = template
        self.fixed_positions = fixed_positions
        self.variable_amino_acids = variable_amino_acids
        self.disulfide_bonds = disulfide_bonds or []
        self.crosslinker = crosslinker
        self.crosslinker_positions = crosslinker_positions
    
    def expand(self, node: MCTSNode, amino_acid: str, prior_probs: Optional[Dict[str, float]] = None) -> MCTSNode:
        """扩展节点"""
        next_pos = self._get_next_position(node.sequence)
        
        if next_pos is None:
            node.is_terminal = True
            return node
        
        allowed_aas = self.variable_amino_acids.get(next_pos, list(config.AMINO_ACIDS))
        
        if amino_acid not in allowed_aas:
            raise ValueError(f"位置 {next_pos} 不允许氨基酸 {amino_acid}")
        
        new_sequence = self._fill_position(node.sequence, next_pos, amino_acid)
        
        child = MCTSNode(
            sequence=new_sequence,
            position=next_pos,
            parent=node,
            prior_probs=prior_probs or {}
        )
        
        if self._is_complete(new_sequence):
            child.is_terminal = True
        
        node.children[amino_acid] = child
        return child
    
    def get_expandable_positions(self, sequence: str) -> List[int]:
        """获取可扩展的位置列表"""
        expandable = []
        for i in range(len(self.template)):
            if i in self.fixed_positions:
                continue
            if self.template[i].lower() == 'x':
                if i >= len(sequence) or sequence[i] == '_' or sequence[i] == 'x':
                    expandable.append(i)
        return expandable
    
    def get_allowed_amino_acids(self, position: int) -> List[str]:
        """获取指定位置允许的氨基酸"""
        return self.variable_amino_acids.get(position, list(config.AMINO_ACIDS))
    
    def create_final_state(self, sequence: str) -> PeptideState:
        """从完整序列创建最终肽状态"""
        # 获取所有Cys位置
        all_cys_positions = [i for i, aa in enumerate(sequence) if aa == 'C']
        
        # 确定交联剂连接位置
        crosslinker_positions = None
        if self.crosslinker:
            required_cys = TopologyManager.get_required_cys_count(self.crosslinker)
            if len(all_cys_positions) >= required_cys:
                crosslinker_positions = all_cys_positions[:required_cys]
        
        return PeptideState(
            sequence=sequence,
            disulfide_bonds=self.disulfide_bonds.copy(),
            crosslinker=self.crosslinker,
            crosslinker_positions=crosslinker_positions
        )
    
    def _get_next_position(self, sequence: str) -> Optional[int]:
        """获取下一个要填充的位置"""
        for i in range(len(self.template)):
            if i in self.fixed_positions:
                continue
            if self.template[i].lower() == 'x':
                if i >= len(sequence) or sequence[i] == '_' or sequence[i] == 'x':
                    return i
        return None
    
    def _fill_position(self, sequence: str, position: int, amino_acid: str) -> str:
        """在指定位置填充氨基酸"""
        seq_list = list(sequence) if len(sequence) == len(self.template) else list(self.template)
        
        for pos, aa in self.fixed_positions.items():
            seq_list[pos] = aa
        
        for i in range(len(self.template)):
            if self.template[i].lower() == 'x':
                if i == position:
                    seq_list[i] = amino_acid
                elif i < len(sequence) and sequence[i] != '_' and sequence[i] != 'x':
                    seq_list[i] = sequence[i]
                else:
                    seq_list[i] = '_'
        
        return ''.join(seq_list)
    
    def _is_complete(self, sequence: str) -> bool:
        """检查序列是否完整"""
        if len(sequence) != len(self.template):
            return False
        for i in range(len(self.template)):
            if sequence[i] == '_' or sequence[i] == 'x':
                return False
        return True


class TopologyManager:
    """拓扑管理器"""
    
    CROSSLINKERS = {
        'TBMB': {'name': '1,3,5-tris(bromomethyl)benzene', 'positions': 3},
        'TATA': {'name': 'tris(2-acryloyl)thiolamine', 'positions': 3},
        'TBAB': {'name': '1,2,4,5-tetrakis(bromomethyl)benzene', 'positions': 4},
    }
    
    @classmethod
    def validate_crosslinker(cls, sequence: str, crosslinker: str, positions: List[int]) -> bool:
        """验证交联剂配置"""
        if crosslinker not in cls.CROSSLINKERS:
            return False
        
        info = cls.CROSSLINKERS[crosslinker]
        cys_positions = [i for i, aa in enumerate(sequence) if aa == 'C']
        
        if len(cys_positions) < info['positions']:
            return False
        
        for pos in positions:
            if pos < 0 or pos >= len(sequence):
                return False
            if sequence[pos] != 'C':
                return False
        
        return True
    
    @classmethod
    def get_required_cys_count(cls, crosslinker: str) -> int:
        """获取交联剂需要的Cys数量"""
        if crosslinker not in cls.CROSSLINKERS:
            return 0
        return cls.CROSSLINKERS[crosslinker]['positions']
