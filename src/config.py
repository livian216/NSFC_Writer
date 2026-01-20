"""配置管理模块"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class PathConfig:
    raw_data: str = "./data/raw"
    processed_data: str = "./data/processed"
    literature_db: str = "./data/literature_db"
    base_model_cache: str = "./models/base"
    finetuned_model: str = "./models/finetuned/nsfc_writer"
    merged_model: str = "./models/finetuned/nsfc_writer_merged"


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_length: int = 4096
    dtype: str = "bfloat16"


@dataclass
class LoraConfig:
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 100
    max_grad_norm: float = 0.3


@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    model_name: str = "nsfc-writer"
    quantization: str = "q4_k_m"


@dataclass
class LiteratureConfig:
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 5


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    repeat_penalty: float = 1.1


@dataclass
class WebAppConfig:
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False


class Config:
    """统一配置管理类"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self._raw_config = self._load_yaml()
        
        # 初始化各配置模块
        self.paths = self._init_dataclass(PathConfig, "paths")
        self.model = self._init_dataclass(ModelConfig, "model")
        self.lora = self._init_dataclass(LoraConfig, "lora")
        self.quantization = self._init_dataclass(QuantizationConfig, "quantization")
        self.training = self._init_dataclass(TrainingConfig, "training")
        self.ollama = self._init_dataclass(OllamaConfig, "ollama")
        self.literature = self._init_dataclass(LiteratureConfig, "literature")
        self.generation = self._init_dataclass(GenerationConfig, "generation")
        self.webapp = self._init_dataclass(WebAppConfig, "webapp")
        
        # 确保目录存在
        self._ensure_directories()
    
    def _load_yaml(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _init_dataclass(self, cls, key: str):
        """初始化dataclass配置"""
        data = self._raw_config.get(key, {})
        if isinstance(data, dict):
            return cls(**{k: v for k, v in data.items() if hasattr(cls, '__dataclass_fields__') and k in cls.__dataclass_fields__})
        return cls()
    
    def _ensure_directories(self):
        """确保必要目录存在"""
        for path in [
            self.paths.raw_data,
            self.paths.processed_data,
            self.paths.literature_db,
            self.paths.base_model_cache,
            os.path.dirname(self.paths.finetuned_model)
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def save(self, path: str = None):
        """保存配置到文件"""
        path = path or self.config_path
        config_dict = {
            "paths": self.paths.__dict__,
            "model": self.model.__dict__,
            "lora": self.lora.__dict__,
            "quantization": self.quantization.__dict__,
            "training": self.training.__dict__,
            "ollama": self.ollama.__dict__,
            "literature": self.literature.__dict__,
            "generation": self.generation.__dict__,
            "webapp": self.webapp.__dict__
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)


# 全局配置实例
_config: Optional[Config] = None


def get_config(config_path: str = "configs/config.yaml") -> Config:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reload_config(config_path: str = "configs/config.yaml") -> Config:
    """重新加载配置"""
    global _config
    _config = Config(config_path)
    return _config