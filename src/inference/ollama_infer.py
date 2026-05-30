
import os
import sys
import json
import argparse
from typing import List, Dict, Tuple
import requests
from tqdm import tqdm

# 添加项目根目录到 Python 搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class OllamaInferencer:
    """
    基于Ollama的批量推理器，支持仇恨识别任务
    包含提示词模板，支持批量推理，结果保存为txt文件
    """
    
    # 系统提示词模板 - 简化版本
    SYSTEM_PROMPT = """你是一个内容安全审核专家。分析文本后严格按以下格式输出：

格式：Target|Argument|Group|Hateful [END]

示例：
文本：黑人就是犯罪分子
输出：黑人|黑人就是犯罪分子|Racism|hate [END]

文本：今天天气真好
输出：non-hate|non-hate|non-hate|non-hate [END]

Group可选值：Racism, Sexism, Region, LGBTQ, others, non-hate
Hateful可选值：hate, non-hate

只输出结果，不要输出其他文字。"""
    
    # 用户提示词模板
    USER_PROMPT_TEMPLATE = "文本：{text}\n输出："
    
    def __init__(self, 
                 model_name: str = "qwen3:8b",
                 api_url: str = "http://localhost:11434/api/generate",
                 timeout: int = 30):
        """
        初始化推理器
        :param model_name: Ollama上部署的模型名称
        :param api_url: Ollama API接口地址
        :param timeout: 请求超时时间，单位秒
        """
        self.model_name = model_name
        self.api_url = api_url
        self.timeout = timeout
    
    def predict_single(self, text: str) -> Tuple[str, int]:
        """
        预测单条文本
        :param text: 待预测的文本内容
        :return: (完整输出, 预测标签0或1)
        """
        prompt = f"{self.SYSTEM_PROMPT}\n\n{self.USER_PROMPT_TEMPLATE.format(text=text)}"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
                "num_predict": 300,
                "stop": ["\n", "文本："]
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            output = result['response'].strip()
            
            # 清理输出 - 只保留 [END] 前的内容
            if '[END]' in output:
                output = output.split('[END]')[0].strip() + ' [END]'
            
            # 解析输出，判断是否仇恨
            pred_label = 0
            output_lower = output.lower()
            
            # 查找 | 分隔的部分
            parts = output_lower.split('|')
            if len(parts) >= 4:
                last_part = parts[-1].strip()
                if 'hate' in last_part and 'non' not in last_part:
                    pred_label = 1
            else:
                # 降级方案
                if 'hate' in output_lower and 'non-hate' not in output_lower:
                    pred_label = 1
            
            # 确保输出格式正确
            if '|' not in output or '[END]' not in output:
                if pred_label == 1:
                    output = f"target|content|Racism|hate [END]"
                else:
                    output = f"non-hate|non-hate|non-hate|non-hate [END]"
            
            return output, pred_label
        except Exception as e:
            print(f"预测失败: {str(e)}，文本: {text[:50]}...")
            return "non-hate|non-hate|non-hate|non-hate [END]", 0
    
    def predict_batch(self, dataset: List[Dict]) -> List[Tuple[str, int, str, int, str]]:
        """
        批量预测数据集
        :param dataset: 数据集列表，每个样本必须包含id、text、label字段
        :return: 预测结果列表，每个元素是(id, 真实标签, 预测完整输出, 预测标签, 文本)
        """
        results = []
        for sample in tqdm(dataset, desc="批量推理中"):
            sample_id = str(sample['id'])
            text = sample['text']
            true_label = int(sample['label'])
            pred_output, pred_label = self.predict_single(text)
            results.append((sample_id, true_label, pred_output, pred_label, text))
        return results
    
    def save_results(self, results: List[Tuple[str, int, str, int, str]], output_path: str) -> None:
        """
        保存预测结果到txt文件，同时保存demo.txt格式和详细格式
        :param results: 预测结果列表
        :param output_path: 输出文件路径
        """
        # 保存 demo.txt 格式的文件
        demo_path = output_path.replace('.txt', '_demo.txt')
        with open(demo_path, 'w', encoding='utf-8') as f:
            for sample_id, _, pred_output, _, _ in results:
                f.write(f"{pred_output}\n")
        print(f"Demo格式结果已保存到: {demo_path}")
        
        # 保存详细格式的文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("id\ttrue_label\tpred_label\tpred_output\ttext\n")
            for sample_id, true_label, pred_output, pred_label, text in results:
                f.write(f"{sample_id}\t{true_label}\t{pred_label}\t{pred_output}\t{text}\n")
        print(f"详细结果已保存到: {output_path}")
    
    def evaluate_results(self, results: List[Tuple[str, int, str, int, str]]) -> Dict[str, float]:
        """
        评估预测结果，计算准确率、精确率、召回率、F1值
        :param results: 预测结果列表
        :return: 评估指标字典
        """
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        
        for _, true_label, _, pred_label, _ in results:
            if true_label == 1 and pred_label == 1:
                true_positive += 1
            elif true_label == 0 and pred_label == 1:
                false_positive += 1
            elif true_label == 0 and pred_label == 0:
                true_negative += 1
            elif true_label == 1 and pred_label == 0:
                false_negative += 1
        
        accuracy = (true_positive + true_negative) / len(results) if len(results) > 0 else 0.0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative
        }
        
        print("评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1值: {f1:.4f}")
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Ollama批量推理脚本')
    parser.add_argument('--dataset', type=str, required=True, help='测试集路径，JSONL格式')
    parser.add_argument('--model', type=str, default='qwen3:8b', help='Ollama模型名称')
    parser.add_argument('--api_url', type=str, default='http://localhost:11434/api/generate', help='Ollama API地址')
    parser.add_argument('--output', type=str, required=True, help='结果输出路径，txt格式')
    parser.add_argument('--evaluate', action='store_true', help='是否评估结果，需要数据集包含label字段')
    
    args = parser.parse_args()
    
    # 加载数据集
    from src.data.data_loader import DataLoader
    loader = DataLoader(args.dataset)
    dataset = loader.load_jsonl()
    print(f"加载测试集样本数: {len(dataset)}")
    
    # 初始化推理器
    inferencer = OllamaInferencer(model_name=args.model, api_url=args.api_url)
    
    # 批量推理
    results = inferencer.predict_batch(dataset)
    
    # 保存结果
    inferencer.save_results(results, args.output)
    
    # 评估结果
    if args.evaluate:
        inferencer.evaluate_results(results)

if __name__ == '__main__':
    main()
