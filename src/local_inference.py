"""本地推理模块 - 直接加载微调模型"""

import os
import torch
import warnings
from typing import Generator, Optional
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread

from .config import get_config


class LocalModel:
    """本地模型推理"""
    
    def __init__(self, model_path: str = None):
        self.config = get_config()
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 模型路径优先级：参数 > 合并模型 > 微调模型 > 基础模型
        if model_path:
            self.model_path = model_path
        elif os.path.exists(self.config.paths.merged_model):
            self.model_path = self.config.paths.merged_model
        elif os.path.exists(self.config.paths.finetuned_model):
            self.model_path = self.config.paths.finetuned_model
        else:
            self.model_path = self.config.model.base_model
        
        self._loaded = False
    
    def load(self):
        """加载模型"""
        if self._loaded:
            return
        
        print(f"加载模型: {self.model_path}")
        print(f"设备: {self.device}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 检查是否是LoRA模型（adapter_config.json存在）
        adapter_config = os.path.join(self.model_path, "adapter_config.json")
        
        if os.path.exists(adapter_config):
            # LoRA模型：需要加载基础模型 + adapter
            print("检测到LoRA模型，加载基础模型和adapter...")
            
            import json
            with open(adapter_config, 'r') as f:
                config = json.load(f)
            base_model_name = config.get("base_model_name_or_path", self.config.model.base_model)
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model = self.model.merge_and_unload()  # 合并权重
        else:
            # 完整模型或合并后的模型
            print("加载完整模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
        
        self.model.eval()
        self._loaded = True
        print("✓ 模型加载完成")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ):
        """生成文本"""
        self.load()
        
        # 构建对话
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 应用chat模板
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            text = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        if stream:
            return self._generate_stream(inputs, max_new_tokens, temperature, top_p)
        else:
            return self._generate_batch(inputs, max_new_tokens, temperature, top_p)
    
    def _generate_batch(self, inputs, max_new_tokens, temperature, top_p) -> str:
        """批量生成"""
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 只取新生成的部分
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def _generate_stream(self, inputs, max_new_tokens, temperature, top_p) -> Generator:
        """流式生成"""
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            yield text
        
        thread.join()
    
    def unload(self):
        """卸载模型释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._loaded = False
        print("✓ 模型已卸载")


# 全局模型实例
_local_model: Optional[LocalModel] = None


def get_local_model() -> LocalModel:
    """获取本地模型实例"""
    global _local_model
    if _local_model is None:
        _local_model = LocalModel()
    return _local_model