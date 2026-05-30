
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仇恨言论推理脚本 - 使用微调后的模型（符合项目格式）
"""
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch


class HateSpeechInferencer:
    """仇恨言论推理器"""
    
    def __init__(self, 
                 model_path="./results/finetuned",
                 base_model_path="./models/Qwen2___5-7B-Instruct-GPTQ-Int4"):
        """初始化推理器"""
        print("加载模型...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        self.model = PeftModel.from_pretrained(self.model, model_path)
        
        print("模型加载完成！")
    
    def predict_single(self, text):
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
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.01,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = generated_text[len(prompt):].strip()
        
        if '[END]' in prediction:
            prediction = prediction.split('[END]')[0].strip() + ' [END]'
        
        for clean_str in ['Human:', 'Assistant:', '文本：', '输出：']:
            if clean_str in prediction:
                prediction = prediction[prediction.rfind(clean_str) + len(clean_str):].strip()
        
        output_lower = prediction.lower()
        pred_label = 0
        if '|hate' in output_lower or '| hate' in output_lower:
            if 'non-hate' not in output_lower and 'non - hate' not in output_lower:
                pred_label = 1
        
        is_hate = pred_label == 1
        
        return {
            'text': text,
            'prediction': prediction,
            'is_hate': is_hate,
            'pred_label': pred_label
        }
    
    def predict_batch(self, texts):
        results = []
        for text in tqdm(texts, desc="推理中"):
            results.append(self.predict_single(text))
        return results
    
    def predict_from_file(self, input_file, output_file=None, save_demo=True):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        for sample in data:
            text = sample.get('content', sample.get('text', ''))
            texts.append(text)
        
        results = self.predict_batch(texts)
        
        for i, sample in enumerate(data):
            sample['prediction'] = results[i]['prediction']
            sample['is_hate'] = results[i]['is_hate']
            sample['pred_label'] = results[i]['pred_label']
        
        if output_file:
            os.makedirs(Path(output_file).parent, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print("详细结果已保存到: " + output_file)
            
            if save_demo:
                demo_file = output_file.replace('.json', '_demo.txt')
                with open(demo_file, 'w', encoding='utf-8') as f:
                    for sample in data:
                        f.write(sample['prediction'] + '\n')
                print("Demo格式结果已保存到: " + demo_file)
        
        return data


def main():
    parser = argparse.ArgumentParser(description='仇恨言论推理脚本')
    parser.add_argument('--model_path', type=str, default='./results/finetuned', help='微调模型路径')
    parser.add_argument('--base_model_path', type=str, default='./models/Qwen2___5-7B-Instruct-GPTQ-Int4', help='基础模型路径')
    parser.add_argument('--input_file', type=str, help='输入文件（JSON格式）')
    parser.add_argument('--output_file', type=str, help='输出文件')
    parser.add_argument('--text', type=str, help='单条文本输入')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("仇恨言论识别")
    print("=" * 60)
    
    inferencer = HateSpeechInferencer(
        model_path=args.model_path,
        base_model_path=args.base_model_path
    )
    
    if args.text:
        print("\n输入文本: " + args.text)
        result = inferencer.predict_single(args.text)
        print("\n预测结果: " + result['prediction'])
        print("是否仇恨: " + ("是" if result['is_hate'] else "否"))
    
    elif args.input_file:
        print("\n从文件加载: " + args.input_file)
        results = inferencer.predict_from_file(
            args.input_file, 
            args.output_file or args.input_file.replace('.json', '_result.json')
        )
        
        hate_count = sum(1 for r in results if r['is_hate'])
        print("\n统计: 共 " + str(len(results)) + " 条，其中仇恨言论 " + str(hate_count) + " 条")
        
        print("\n示例:")
        for i, r in enumerate(results[:5]):
            text = r.get('content', r.get('text', ''))[:50]
            status = "仇恨" if r['is_hate'] else "正常"
            print(str(i+1) + ". " + text + "...")
            print("   输出: " + r['prediction'])
    
    else:
        print("\n请使用 --text 输入单条文本，或 --input_file 输入文件")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()

