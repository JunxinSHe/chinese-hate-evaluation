import os
import sys
import yaml
import json
import argparse
import shutil
from typing import Dict, Any
from llmtuner.train.tuner import run_exp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class QwenFinetuner:
    """
    基于Qwen模型和LLaMA-Factory的全参数微调器
    支持仇恨识别四元组任务的训练、验证、模型保存全流程
    """
    
    def __init__(self, config_path: str):
        """
        初始化微调器
        :param config_path: 配置文件路径，YAML格式
        """
        self.config = self.load_config(config_path)
        self.exp_args = self.build_experiment_args()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        :param config_path: 配置文件路径
        :return: 配置字典
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def build_experiment_args(self) -> Dict[str, Any]:
        """
        构建LLaMA-Factory实验参数字典
        :return: 参数字典
        """
        model_config = self.config.get('model', {})
        training_config = self.config.get('training', {})
        data_config = self.config.get('data', {})
        output_config = self.config.get('output', {})
        
        args = {
            # 模型配置
            'model_name_or_path': model_config.get('model_name_or_path', 'Qwen/Qwen-7B-Chat'),
            'quantization_bit': model_config.get('quantization_bit', None),
            'quantization_type': model_config.get('quantization_type', 'fp4'),
            'double_quantization': model_config.get('double_quantization', True),
            
            # 训练配置
            'stage': training_config.get('stage', 'sft'),
            'do_train': training_config.get('do_train', True),
            'do_eval': training_config.get('do_eval', True),
            'finetuning_type': training_config.get('finetuning_type', 'lora'),
            'num_train_epochs': training_config.get('num_train_epochs', 3),
            'per_device_train_batch_size': training_config.get('per_device_train_batch_size', 4),
            'per_device_eval_batch_size': training_config.get('per_device_eval_batch_size', 4),
            'gradient_accumulation_steps': training_config.get('gradient_accumulation_steps', 4),
            'learning_rate': training_config.get('learning_rate', 2e-5),
            'max_grad_norm': training_config.get('max_grad_norm', 1.0),
            'lr_scheduler_type': training_config.get('lr_scheduler_type', 'cosine'),
            'warmup_ratio': training_config.get('warmup_ratio', 0.05),
            'weight_decay': training_config.get('weight_decay', 0.01),
            'logging_steps': training_config.get('logging_steps', 10),
            'save_strategy': training_config.get('save_strategy', 'epoch'),
            'eval_strategy': training_config.get('evaluation_strategy', 'epoch'),
            'load_best_model_at_end': training_config.get('load_best_model_at_end', True),
            'metric_for_best_model': training_config.get('metric_for_best_model', 'f1'),
            'greater_is_better': training_config.get('greater_is_better', True),
            
            # 数据配置
            'dataset': data_config.get('dataset', 'json'),
            'dataset_dir': data_config.get('dataset_dir', './results/processed_data'),
            'template': data_config.get('template', 'qwen'),
            'cutoff_len': data_config.get('cutoff_len', 1024),
            'max_samples': data_config.get('max_samples', None),
            'val_size': data_config.get('val_size', 0.1),
            
            # 输出配置
            'output_dir': output_config.get('output_dir', './results/qwen_finetuned'),
            'logging_dir': output_config.get('logging_dir', './logs'),
            'overwrite_output_dir': output_config.get('overwrite_output_dir', True),
            'save_total_limit': output_config.get('save_total_limit', 3),
            
            # 其他配置
            'seed': training_config.get('seed', 42),
            'fp16': training_config.get('fp16', True),
            'bf16': training_config.get('bf16', False),
            'gradient_checkpointing': training_config.get('gradient_checkpointing', True),
            'report_to': training_config.get('report_to', 'none'),
        }
        
        # 移除 None 值
        args = {k: v for k, v in args.items() if v is not None}
        return args
    
    def run(self) -> None:
        """
        执行微调训练和验证
        """
        run_exp(self.exp_args)
    
    def save_model(self, output_path: str) -> None:
        """
        保存训练好的模型到指定路径
        :param output_path: 模型保存路径
        """
        src_dir = self.exp_args.get('output_dir', './results/qwen_finetuned')
        if os.path.exists(src_dir):
            shutil.copytree(src_dir, output_path, dirs_exist_ok=True)
            print(f"模型已保存到: {output_path}")
        else:
            raise FileNotFoundError(f"模型目录不存在: {src_dir}")

def main():
    parser = argparse.ArgumentParser(description='Qwen模型仇恨识别任务微调脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--save_path', type=str, help='模型导出路径，可选')
    args = parser.parse_args()
    
    finetuner = QwenFinetuner(args.config)
    finetuner.run()
    
    if args.save_path:
        finetuner.save_model(args.save_path)

if __name__ == '__main__':
    main()
