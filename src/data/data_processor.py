import re
import json
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter
from .data_validator import DataValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    数据预处理器，负责数据清洗、标注对齐、格式转换
    支持输出四元组格式：(text, label, hate_type, id)
    支持转换为LLaMA-Factory微调所需的格式
    """
    
    def __init__(self, 
                 remove_duplicates: bool = True,
                 remove_empty_text: bool = True,
                 normalize_text: bool = True,
                 align_hate_types: bool = True,
                 validate_samples: bool = True):
        """
        初始化预处理器
        :param remove_duplicates: 是否去除重复样本
        :param remove_empty_text: 是否去除空文本样本
        :param normalize_text: 是否标准化文本格式
        :param align_hate_types: 是否对齐仇恨类型标签
        :param validate_samples: 是否校验样本合法性，自动过滤不合法样本
        """
        self.remove_duplicates = remove_duplicates
        self.remove_empty_text = remove_empty_text
        self.normalize_text = normalize_text
        self.align_hate_types = align_hate_types
        self.validate_samples = validate_samples
        self.validator = DataValidator() if validate_samples else None
        
        # 仇恨类型映射，用于对齐不同标注体系的仇恨类型
        self.hate_type_mapping = {
            '性别歧视': 'gender',
            '性别': 'gender',
            'gender': 'gender',
            'Sexism': 'gender',
            'sexism': 'gender',
            '种族歧视': 'race',
            '种族': 'race',
            'race': 'race',
            'Racism': 'race',
            'racism': 'race',
            '地域歧视': 'region',
            '地域': 'region',
            'region': 'region',
            'Region': 'region',
            '人身攻击': 'attack',
            '攻击': 'attack',
            'attack': 'attack',
            'LGBTQ': 'lgbtq',
            'lgbtq': 'lgbtq',
            '其他': 'other',
            'other': 'other',
            'others': 'other',
            'Others': 'other',
            'non-hate': 'non-hate',
            '未知': 'other',
            None: 'other'
        }
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本，去除多余空格、特殊字符、HTML标签等
        :param text: 原始文本
        :return: 清洗后的文本
        """
        if not isinstance(text, str):
            return ""
        
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 去除URL链接
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # 去除@提及和#话题
        text = re.sub(r'@\w+|#\w+#', '', text)
        # 去除多余空格和换行
        text = re.sub(r'\s+', ' ', text).strip()
        # 去除特殊控制字符
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        return text
    
    def align_hate_type(self, hate_type: Optional[str]) -> str:
        """
        对齐仇恨类型标签，统一为预定义的标准类型
        :param hate_type: 原始仇恨类型标签
        :return: 标准化后的仇恨类型
        """
        if not hate_type:
            return 'other'
        
        hate_type = str(hate_type).lower().strip()
        return self.hate_type_mapping.get(hate_type, 'other')
    
    def process_sample(self, sample: Dict) -> Optional[Dict]:
        """
        处理单个样本
        :param sample: 原始样本字典
        :return: 处理后的样本字典，返回None表示样本被过滤
        """
        processed = sample.copy()
        
        # 校验样本合法性
        if self.validate_samples:
            is_valid, errors = self.validator.validate_sample(processed)
            if not is_valid:
                logger.debug(f"样本不合法，已过滤: {errors}")
                return None
        
        # 清洗文本
        if self.normalize_text and 'text' in processed:
            processed['text'] = self.clean_text(processed['text'])
        
        # 过滤空文本
        if self.remove_empty_text and processed.get('text', '').strip() == '':
            logger.debug("样本文本为空，已过滤")
            return None
        
        # 对齐仇恨类型
        if self.align_hate_types:
            processed['hate_type'] = self.align_hate_type(processed.get('hate_type', 'other'))
        
        return processed
    
    def process_dataset(self, dataset: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        处理整个数据集
        :param dataset: 原始数据集列表
        :return: (处理后的数据集列表, 处理统计信息)
        """
        processed_dataset = []
        text_set = set()
        duplicate_count = 0
        empty_count = 0
        
        for sample in dataset:
            processed = self.process_sample(sample)
            if processed is None:
                empty_count += 1
                continue
            
            # 去重
            if self.remove_duplicates:
                text = processed['text']
                if text in text_set:
                    duplicate_count += 1
                    continue
                text_set.add(text)
            
            processed_dataset.append(processed)
        
        stats = {
            "original_count": len(dataset),
            "processed_count": len(processed_dataset),
            "duplicate_removed": duplicate_count,
            "empty_removed": empty_count
        }
        
        return processed_dataset, stats
    
    def to_quadruple(self, dataset: List[Dict]) -> List[Tuple[str, int, str, str]]:
        """
        将数据集转换为四元组格式：(text, label, hate_type, id)
        :param dataset: 处理后的数据集
        :return: 四元组列表
        """
        quadruples = []
        filtered_count = 0
        for sample in dataset:
            try:
                text = str(sample['text']).strip()
                label = int(sample['label'])
                hate_type = str(sample.get('hate_type', 'other')).lower()
                sample_id = str(sample['id'])
                quadruples.append((text, label, hate_type, sample_id))
            except (KeyError, ValueError) as e:
                filtered_count += 1
                logger.debug(f"转换四元组时过滤无效样本: {str(e)}")
        
        if filtered_count > 0:
            logger.info(f"四元组转换共过滤{filtered_count}个无效样本")
        return quadruples
    
    def to_llama_factory_format(self, dataset: List[Dict], output_path: str) -&gt; None:
        """
        将数据集转换为LLaMA-Factory微调所需的JSONL格式
        :param dataset: 处理后的数据集
        :param output_path: 输出文件路径
        """
        system_prompt = """你是一个内容安全审核专家。分析文本后严格按以下格式输出：

格式：Target|Argument|Group|Hateful [END]

示例：
文本：黑人就是犯罪分子
输出：黑人|黑人就是犯罪分子|Racism|hate [END]

文本：今天天气真好
输出：non-hate|non-hate|non-hate|non-hate [END]

Group可选值：Racism, Sexism, Region, LGBTQ, others, non-hate
Hateful可选值：hate, non-hate

只输出结果，不要输出其他文字。"""
        filtered_count = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in dataset:
                try:
                    text = str(sample['text']).strip()
                    # 使用完整输出格式，如果没有则降级为基于label的格式
                    if 'full_output' in sample:
                        output_text = sample['full_output']
                    else:
                        label = int(sample['label'])
                        if label == 1:
                            hate_type = sample.get('hate_type', 'others')
                            output_text = f"target|content|{hate_type}|hate [END]"
                        else:
                            output_text = "non-hate|non-hate|non-hate|non-hate [END]"
                    
                    conversation = {
                        "instruction": system_prompt,
                        "input": text,
                        "output": output_text,
                        "system": system_prompt,
                        "tools": [],
                        "history": []
                    }
                    f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
                except (KeyError, ValueError) as e:
                    filtered_count += 1
                    logger.debug(f"转换训练格式时过滤无效样本: {str(e)}")
        
        if filtered_count &gt; 0:
            logger.info(f"训练格式转换共过滤{filtered_count}个无效样本")
    
    def save_quadruples(self, quadruples: List[Tuple[str, int, str, str]], output_path: str) -> None:
        """
        保存四元组到文件，每行一个四元组，用制表符分隔
        :param quadruples: 四元组列表
        :param output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for text, label, hate_type, sample_id in quadruples:
                f.write(f"{sample_id}\t{label}\t{hate_type}\t{text}\n")
