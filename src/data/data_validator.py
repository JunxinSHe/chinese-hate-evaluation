import re
from typing import List, Dict, Tuple

class DataValidator:
    """
    数据校验器，负责校验数据集的字段完整性、标签合法性、数据格式正确性
    """
    
    REQUIRED_FIELDS = ['id', 'text', 'label']
    VALID_LABELS = [0, 1]
    VALID_HATE_TYPES = ['gender', 'race', 'region', 'attack', 'other', 'lgbtq', 'non-hate', None]
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate_sample(self, sample: Dict, index: int = -1) -> Tuple[bool, List[str]]:
        """
        校验单个样本的合法性
        :param sample: 待校验的样本字典
        :param index: 样本在数据集中的索引，用于错误提示
        :return: (是否合法, 错误信息列表)
        """
        sample_errors = []
        prefix = f"样本{index if index != -1 else ''}:"
        
        # 检查必填字段
        for field in self.REQUIRED_FIELDS:
            if field not in sample:
                sample_errors.append(f"{prefix}缺失必填字段{field}")
        
        # 检查label合法性
        if 'label' in sample:
            label = sample['label']
            if label not in self.VALID_LABELS:
                sample_errors.append(f"{prefix}标签值{label}不合法，必须是0或1")
        
        # 检查hate_type合法性
        if 'hate_type' in sample:
            hate_type = sample['hate_type']
            if hate_type not in self.VALID_HATE_TYPES:
                self.warnings.append(f"{prefix}仇恨类型{hate_type}不在预定义列表中")
        
        # 检查text字段
        if 'text' in sample:
            text = sample['text']
            if not isinstance(text, str) or len(text.strip()) == 0:
                sample_errors.append(f"{prefix}文本字段为空或不是字符串类型")
            # 检查乱码（简单检查：包含大量不可见字符）
            invisible_chars = len(re.findall(r'[\x00-\x1F\x7F-\x9F]', text))
            if invisible_chars > len(text) * 0.3:
                self.warnings.append(f"{prefix}文本包含大量不可见字符，可能是乱码")
        
        # 检查id字段
        if 'id' in sample:
            sample_id = sample['id']
            if not isinstance(sample_id, (str, int)):
                sample_errors.append(f"{prefix}ID字段类型不合法，必须是字符串或整数")
        
        return len(sample_errors) == 0, sample_errors
    
    def validate_dataset(self, dataset: List[Dict]) -> Tuple[bool, List[str], List[str]]:
        """
        校验整个数据集的合法性
        :param dataset: 待校验的数据集列表
        :return: (是否全部合法, 错误信息列表, 警告信息列表)
        """
        self.errors.clear()
        self.warnings.clear()
        all_valid = True
        id_set = set()
        
        for i, sample in enumerate(dataset):
            is_valid, sample_errors = self.validate_sample(sample, i)
            if not is_valid:
                all_valid = False
                self.errors.extend(sample_errors)
            
            # 检查ID重复
            if 'id' in sample:
                sample_id = sample['id']
                if sample_id in id_set:
                    self.errors.append(f"样本{i}:ID{sample_id}重复")
                id_set.add(sample_id)
        
        return all_valid, self.errors, self.warnings
    
    def filter_invalid_samples(self, dataset: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        过滤掉不合法的样本
        :param dataset: 原始数据集
        :return: (合法样本列表, 不合法样本列表)
        """
        valid_samples = []
        invalid_samples = []
        
        for sample in dataset:
            is_valid, _ = self.validate_sample(sample)
            if is_valid:
                valid_samples.append(sample)
            else:
                invalid_samples.append(sample)
        
        return valid_samples, invalid_samples
