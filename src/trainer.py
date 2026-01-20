"""模型微调模块 - 带详细日志"""

import os
import sys
import json
import warnings
import traceback
import gc

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from typing import Optional
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

from .config import get_config


class ModelTrainer:
    """模型微调器"""
    
    def __init__(self):
        self.config = get_config()
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def check_gpu(self):
        """检查GPU状态"""
        print("\n" + "=" * 50)
        print("GPU 信息")
        print("=" * 50)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU: {gpu_name}")
            print(f"✓ 显存: {gpu_memory:.1f} GB")
            print(f"✓ CUDA版本: {torch.version.cuda}")
            torch.cuda.empty_cache()
            gc.collect()
            return True
        else:
            print("⚠ CUDA不可用，将使用CPU训练")
            return False
    
    def setup_model(self):
        """设置模型和分词器"""
        print("\n" + "=" * 50)
        print("加载模型")
        print("=" * 50)
        
        model_name = self.config.model.base_model
        cache_dir = self.config.paths.base_model_cache
        
        print(f"基础模型: {model_name}")
        print(f"缓存目录: {cache_dir}")
        
        try:
            # 加载分词器
            print("\n[1/4] 加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="right",
                cache_dir=cache_dir
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print("   ✓ 分词器加载完成")
            
            # 加载模型
            print("\n[2/4] 加载模型...")
            print(f"   - CUDA可用: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                device_map = "auto"
                dtype = torch.bfloat16
                print(f"   - 使用GPU + bfloat16")
            else:
                device_map = None
                dtype = torch.float32
                print(f"   - 使用CPU + float32")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            print("   ✓ 模型加载完成")
            
            # 启用梯度检查点
            print("\n[3/4] 配置训练参数...")
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
            print("   ✓ 梯度检查点已启用")
            
            # 配置LoRA
            print("\n[4/4] 配置LoRA...")
            peft_config = LoraConfig(
                r=self.config.lora.r,
                lora_alpha=self.config.lora.alpha,
                lora_dropout=self.config.lora.dropout,
                target_modules=self.config.lora.target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.model, peft_config)
            print("   ✓ LoRA配置完成")
            
            self._print_trainable_parameters()
            
            print("\n" + "=" * 50)
            print("✓ 模型加载完成!")
            print("=" * 50)
            
        except Exception as e:
            print(f"\n" + "=" * 50)
            print("❌ 模型加载失败!")
            print("=" * 50)
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print("\n详细错误信息:")
            traceback.print_exc()
            raise
    
    def _print_trainable_parameters(self):
        """打印可训练参数"""
        trainable_params = 0
        all_params = 0
        
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"\n可训练参数: {trainable_params:,} / {all_params:,} "
              f"({100 * trainable_params / all_params:.2f}%)")
    
    def load_dataset(self, data_path: str = None) -> Dataset:
        """加载训练数据"""
        data_path = data_path or os.path.join(
            self.config.paths.processed_data, "train_data.json"
        )
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"训练数据不存在: {data_path}\n"
                f"请先运行: python main.py process_data"
            )
        
        print(f"\n加载数据: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"原始样本数: {len(data)}")
        
        processed_data = []
        for item in data:
            messages = [
                {"role": "system", "content": "你是一个专业的国家自然科学基金申请书写作助手。"},
                {"role": "user", "content": f"{item['instruction']}\n\n{item.get('input', '')}".strip()},
                {"role": "assistant", "content": item['output']}
            ]
            
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except:
                text = f"### 指令:\n{item['instruction']}\n\n"
                if item.get('input'):
                    text += f"### 输入:\n{item['input']}\n\n"
                text += f"### 回答:\n{item['output']}"
            
            processed_data.append({"text": text})
        
        dataset = Dataset.from_list(processed_data)
        print(f"处理后样本数: {len(dataset)}")
        
        return dataset
    
    def train(self, data_path: str = None):
        """执行训练"""
        print("\n" + "=" * 50)
        print("🚀 开始训练流程")
        print("=" * 50)
        
        # 检查GPU
        has_gpu = self.check_gpu()
        
        # 加载模型
        self.setup_model()
        
        # 加载数据
        dataset = self.load_dataset(data_path)
        
        # 分词
        print("\n对数据进行分词...")
        
        def tokenize_function(examples):
            result = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.model.max_length,
                padding="max_length",
                return_tensors=None
            )
            result["labels"] = result["input_ids"].copy()
            return result
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="分词处理"
        )
        print(f"✓ 分词完成，共 {len(tokenized_dataset)} 条数据")
        
        # 训练参数
        output_dir = self.config.paths.finetuned_model
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            warmup_ratio=self.config.training.warmup_ratio,
            lr_scheduler_type="cosine",
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=3,
            fp16=False,
            bf16=has_gpu,
            gradient_checkpointing=True,
            max_grad_norm=self.config.training.max_grad_norm,
            optim="adamw_torch",
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        
        # 创建Trainer
        print("\n创建训练器...")
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        print("\n" + "=" * 50)
        print("开始训练")
        print("=" * 50)
        print(f"输出目录: {output_dir}")
        print(f"训练轮数: {self.config.training.num_epochs}")
        print(f"批次大小: {self.config.training.batch_size}")
        print(f"梯度累积: {self.config.training.gradient_accumulation_steps}")
        print(f"学习率: {self.config.training.learning_rate}")
        
        # 开始训练
        self.trainer.train()
        
        # 保存
        print("\n保存模型...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"\n✓ 模型已保存到: {output_dir}")
    
    def merge_and_save(self, output_dir: str = None):
        """合并LoRA权重"""
        output_dir = output_dir or self.config.paths.merged_model
        
        print("\n" + "=" * 50)
        print("合并LoRA权重")
        print("=" * 50)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        merged_model = self.model.merge_and_unload()
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✓ 合并后的模型已保存到: {output_dir}")
    
    def run(self, data_path: str = None, merge: bool = True):
        """完整训练流程"""
        print("\n" + "=" * 50)
        print("🚀 国自然写作助手 - 模型微调")
        print("=" * 50)
        
        try:
            self.train(data_path)
            
            if merge:
                self.merge_and_save()
            
            print("\n" + "=" * 50)
            print("✓ 训练完成!")
            print("=" * 50)
            print("\n下一步操作:")
            print("  1. 启动Web应用: python main.py run")
            print("  2. 或部署到Ollama: python main.py deploy")
            
        except KeyboardInterrupt:
            print("\n\n⚠ 训练被用户中断")
            
        except Exception as e:
            print(f"\n❌ 训练失败: {str(e)}")
            traceback.print_exc()
            raise