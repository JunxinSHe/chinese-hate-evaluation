<<<<<<< HEAD
# 中文仇恨识别评测工具

## 项目简介
本项目是一套标准化的中文仇恨识别模型评测体系，包含数据处理模块、模型微调模块、推理评测模块，支持数据集预处理、模型微调、批量推理、性能评测全流程。

## 模块功能

### 1. 数据处理模块 (src/data)
- **DataLoader**: 负责加载JSONL格式的数据集，支持按比例划分训练集、验证集、测试集
- **DataValidator**: 负责数据校验，检查字段完整性、标签合法性、数据格式正确性
- **DataProcessor**: 负责数据清洗、标注对齐、格式转换，支持输出四元组格式：(text, label, hate_type, id)，以及转换为LLaMA-Factory微调所需的格式

### 2. 模型微调模块 (src/model)
- **QwenFinetuner**: 基于Qwen模型和LLaMA-Factory实现全参数微调，支持训练、验证、模型保存全流程，适配仇恨识别四元组任务

### 3. 推理评测模块 (src/inference)
- **OllamaInferencer**: 基于Ollama的批量推理器，包含标准化提示词模板，支持批量推理，结果保存为txt文件，支持自动计算准确率、精确率、召回率、F1值等评测指标

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用示例

### 1. 数据处理示例
```python
from src.data import DataLoader, DataValidator, DataProcessor

# 加载数据集
loader = DataLoader('datasets/test_data.jsonl')
data = loader.load_jsonl()
print(f"加载样本数: {len(data)}")

# 校验数据
validator = DataValidator()
all_valid, errors, warnings = validator.validate_dataset(data)
print(f"数据校验结果: {'通过' if all_valid else '不通过'}")
if errors:
    print(f"错误信息: {errors[:5]}")

# 处理数据
processor = DataProcessor()
processed_data, stats = processor.process_dataset(data)
print(f"处理后样本数: {len(processed_data)}")
print(f"处理统计: {stats}")

# 转换为四元组格式
quadruples = processor.to_quadruple(processed_data)
processor.save_quadruples(quadruples, 'results/quadruples.txt')

# 转换为LLaMA-Factory训练格式
processor.to_llama_factory_format(processed_data, 'datasets/train_data.jsonl')
```

### 2. 模型微调示例
```bash
# 编写配置文件 config/training_config.yaml
python src/model/train.py --config config/training_config.yaml --save_path models/qwen_hate_finetuned
```

### 3. 批量推理示例
```bash
# 部署模型到Ollama后执行
python src/inference/ollama_infer.py \
    --dataset datasets/test_data.jsonl \
    --model qwen_hate_finetuned \
    --output results/inference_result.txt \
    --evaluate
```

## 输出格式说明
1. 四元组格式：每行由制表符分隔，顺序为id\tlabel\thate_type\ttext
2. 推理结果格式：每行由制表符分隔，顺序为id\ttrue_label\tpred_label\ttext
3. LLaMA-Factory训练格式：JSONL格式，包含instruction、input、output字段

## 配置文件示例
参考config目录下的示例配置文件：
- dataset_config.yaml: 数据集配置
- model_config.yaml: 模型配置
- training_config.yaml: 训练参数配置
=======
# chinese-hate-evaluation
>>>>>>> 19cb1a65338cc21bc15801df5a7804d8d416f32e
