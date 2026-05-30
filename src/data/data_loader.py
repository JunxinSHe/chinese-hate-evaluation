import json
import random
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    数据加载器，负责加载和划分评测数据集
    支持JSONL格式的数据集加载，支持按比例划分训练集、验证集、测试集
    """
    
    def __init__(self, file_path: str, seed: int = 42):
        """
        初始化数据加载器
        :param file_path: 数据集文件路径，JSONL格式
        :param seed: 随机种子，用于数据集划分
        """
        self.file_path = file_path
        self.seed = seed
        self.data: List[Dict] = []
        
    def load_jsonl(self) -> List[Dict]:
        """
        加载JSONL格式的数据集
        :return: 数据集列表，每个元素是一个字典，包含样本的所有字段
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                self.data.append(sample)
        return self.data
    
    def split_data(self, 
                   train_ratio: float = 0.0, 
                   val_ratio: float = 0.0, 
                   test_ratio: float = 1.0,
                   stratify: bool = False) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        划分数据集为训练集、验证集、测试集
        :param train_ratio: 训练集比例，0表示不划分训练集
        :param val_ratio: 验证集比例，0表示不划分验证集
        :param test_ratio: 测试集比例，默认1.0表示全部作为测试集
        :param stratify: 是否按标签分层划分，保持类别分布一致
        :return: (train_data, val_data, test_data)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("训练集、验证集、测试集比例之和必须为1.0")
        
        data = self.data.copy()
        random.seed(self.seed)
        random.shuffle(data)
        
        if stratify and 'label' in data[0]:
            labels = [sample['label'] for sample in data]
            remaining_data, test_data = train_test_split(
                data, test_size=test_ratio, random_state=self.seed, stratify=labels
            )
            if train_ratio > 0 and val_ratio > 0:
                val_size = val_ratio / (train_ratio + val_ratio)
                train_data, val_data = train_test_split(
                    remaining_data, test_size=val_size, random_state=self.seed,
                    stratify=[sample['label'] for sample in remaining_data]
                )
            else:
                train_data = remaining_data if train_ratio > 0 else []
                val_data = [] if val_ratio == 0 else remaining_data
        else:
            total = len(data)
            train_end = int(total * train_ratio)
            val_end = train_end + int(total * val_ratio)
            
            train_data = data[:train_end]
            val_data = data[train_end:val_end]
            test_data = data[val_end:]
        
        return train_data, val_data, test_data
    
    def to_pandas(self, data: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        将数据集转换为Pandas DataFrame格式
        :param data: 要转换的数据集，默认使用加载的全部数据
        :return: DataFrame格式的数据集
        """
        if data is None:
            data = self.data
        return pd.DataFrame(data)
    
    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息
        :return: 统计信息字典，包含样本总数、正负样本比例、仇恨类型分布等
        """
        if not self.data:
            raise ValueError("数据集未加载，请先调用load_jsonl()方法")
        
        total = len(self.data)
        label_counts = {}
        hate_type_counts = {}
        
        for sample in self.data:
            label = sample.get('label', -1)
            label_counts[label] = label_counts.get(label, 0) + 1
            
            hate_type = sample.get('hate_type', 'unknown')
            hate_type_counts[hate_type] = hate_type_counts.get(hate_type, 0) + 1
        
        return {
            "total_samples": total,
            "label_distribution": label_counts,
            "hate_type_distribution": hate_type_counts
        }
