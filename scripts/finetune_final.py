
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终版 - 仇恨言论 LoRA 微调脚本（直接使用output字段）
"""
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """简单的文本数据集"""
    
    def __init__(self, texts, tokenizer, max_len=384):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


def main():
    parser = argparse.ArgumentParser(description='仇恨言论模型微调')
    parser.add_argument('--model_path', type=str, 
                        default='./models/Qwen2___5-7B-Instruct-GPTQ-Int4', 
                        help='模型路径')
    parser.add_argument('--data_path', type=str, 
                        default='./datasets/train.json', 
                        help='训练数据')
    parser.add_argument('--output_dir', type=str, 
                        default='./results/finetuned', 
                        help='输出目录')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--max_len', type=int, default=384, help='最大序列长度')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("仇恨言论模型微调（显存优化版）")
    print("=" * 60)
    
    # 1. 加载模型和分词器
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\n正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ 分词器加载成功")
    
    print("\n正在加载 GPTQ 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    print("✅ 模型加载成功")
    
    # 2. 设置 LoRA（关键：在 apply 之前确保模型可训练）
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    print("\n准备模型用于训练...")
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 3. 加载数据集（直接使用output字段）
    print("\n正在加载训练数据...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    texts = []
    for example in raw_data:
        text = example.get('content', example.get('text', ''))
        output = example.get('output', '')
        
        prompt = """你是一个内容安全审核专家。分析文本后严格按以下格式输出：

格式：Target|Argument|Group|Hateful [END]

示例：
文本：黑人就是犯罪分子
输出：黑人 | 黑人就是犯罪分子 | Racism | hate [END]

文本：今天天气真好
输出：non-hate | non-hate | non-hate | non-hate [END]

Group可选值：Racism, Sexism, Region, LGBTQ, others, non-hate
Hateful可选值：hate, non-hate

只输出结果，不要输出其他文字。

文本：""" + text + "\n输出："
        
        full_text = prompt + output
        texts.append(full_text)
    
    print(f"✅ 数据预处理完成，共 {len(texts)} 条")
    print(f"示例数据: {texts[0][:200]}...")
    
    # 创建数据集
    dataset = TextDataset(texts, tokenizer, args.max_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 4. 设置训练参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * len(dataloader)
    
    print(f"\n训练配置:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Steps per Epoch: {len(dataloader)}")
    print(f"  Total Steps: {num_training_steps}")
    
    # 5. 开始训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/(global_step % len(dataloader) + 1):.4f}'
            })
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1}/{args.epochs} 完成, Average Loss: {avg_loss:.4f}")
        
        # 保存模型
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"模型已保存到: {args.output_dir}")
    
    # 6. 最终保存
    print("\n" + "=" * 60)
    print("训练完成！模型已保存")
    print("=" * 60)


if __name__ == '__main__':
    main()

