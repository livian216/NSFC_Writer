"""Ollama部署模块"""

import os
import subprocess
import requests
from pathlib import Path
from typing import Optional

from .config import get_config


class OllamaDeployer:
    """Ollama模型部署器"""
    
    def __init__(self):
        self.config = get_config()
        self.model_name = self.config.ollama.model_name
        self.ollama_host = self.config.ollama.host
    
    def check_ollama_service(self) -> bool:
        """检查Ollama服务是否运行"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> list:
        """列出已有模型"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                return [m['name'] for m in response.json().get('models', [])]
        except:
            pass
        return []
    
    def create_modelfile(self, gguf_path: str, output_path: str = None) -> str:
        """创建Modelfile"""
        modelfile_content = f'''# 国自然科学基金申请书写作助手
FROM {gguf_path}

SYSTEM """你是一个专业的国家自然科学基金申请书写作助手。

你的专长包括：
1. 立项依据：研究背景、国内外现状、研究意义
2. 研究内容：核心问题、研究要点、研究范围
3. 研究方案：技术路线、研究方法、实施步骤
4. 创新点：理论创新、方法创新、应用创新
5. 预期成果：论文、专利、应用价值
6. 研究基础：前期积累、实验条件、团队构成

请严格按照国自然申请书的写作规范提供专业内容。
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
"""
'''
        
        output_path = output_path or os.path.join(
            os.path.dirname(gguf_path), "Modelfile"
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        print(f"Modelfile已创建: {output_path}")
        return output_path
    
    def convert_to_gguf(self, model_path: str = None, quantization: str = None) -> str:
        """将模型转换为GGUF格式"""
        model_path = model_path or self.config.paths.merged_model
        quantization = quantization or self.config.ollama.quantization
        
        print("=" * 50)
        print("转换模型为GGUF格式")
        print("=" * 50)
        
        # 检查llama.cpp
        if not os.path.exists("llama.cpp"):
            print("正在下载llama.cpp...")
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp.git"],
                check=True
            )
            
            print("编译llama.cpp...")
            subprocess.run(["make", "-j", "-C", "llama.cpp"], check=True)
            
            print("安装Python依赖...")
            subprocess.run(
                ["pip", "install", "-r", "llama.cpp/requirements.txt"],
                check=True
            )
        
        # 转换为FP16 GGUF
        gguf_fp16 = os.path.join(model_path, "model-fp16.gguf")
        
        print("转换为FP16 GGUF...")
        subprocess.run([
            "python", "llama.cpp/convert_hf_to_gguf.py",
            model_path,
            "--outfile", gguf_fp16,
            "--outtype", "f16"
        ], check=True)
        
        # 量化
        gguf_quantized = os.path.join(model_path, f"{self.model_name}-{quantization}.gguf")
        
        print(f"量化为 {quantization}...")
        subprocess.run([
            "llama.cpp/build/bin/llama-quantize",
            gguf_fp16,
            gguf_quantized,
            quantization.upper()
        ], check=True)
        
        # 清理中间文件
        if os.path.exists(gguf_fp16):
            os.remove(gguf_fp16)
        
        print(f"GGUF模型已保存: {gguf_quantized}")
        return gguf_quantized
    
    def register_model(self, gguf_path: str = None):
        """注册模型到Ollama"""
        if not self.check_ollama_service():
            print("错误: Ollama服务未运行，请先启动 'ollama serve'")
            return False
        
        if gguf_path is None:
            # 查找GGUF文件
            model_dir = self.config.paths.merged_model
            gguf_files = list(Path(model_dir).glob("*.gguf"))
            if not gguf_files:
                print(f"错误: 未找到GGUF文件，请先运行转换")
                return False
            gguf_path = str(gguf_files[0])
        
        print("=" * 50)
        print(f"注册模型到Ollama: {self.model_name}")
        print("=" * 50)
        
        # 创建Modelfile
        modelfile_path = self.create_modelfile(gguf_path)
        
        # 注册模型
        subprocess.run([
            "ollama", "create",
            self.model_name,
            "-f", modelfile_path
        ], check=True)
        
        print(f"模型 {self.model_name} 已成功注册")
        
        # 显示已有模型
        print("\n当前Ollama模型:")
        for model in self.list_models():
            print(f"  - {model}")
        
        return True
    
    def deploy_base_model(self, base_model: str = "qwen2.5:7b"):
        """部署基础模型（不经过微调）"""
        print("=" * 50)
        print(f"拉取基础模型: {base_model}")
        print("=" * 50)
        
        if not self.check_ollama_service():
            print("错误: Ollama服务未运行")
            return False
        
        subprocess.run(["ollama", "pull", base_model], check=True)
        print(f"基础模型 {base_model} 已就绪")
        return True
    
    def run(self, model_path: str = None, skip_convert: bool = False):
        """完整部署流程"""
        print("=" * 50)
        print("开始Ollama部署")
        print("=" * 50)
        
        if not self.check_ollama_service():
            print("请先启动Ollama服务: ollama serve")
            return False
        
        if not skip_convert:
            gguf_path = self.convert_to_gguf(model_path)
        else:
            gguf_path = None
        
        success = self.register_model(gguf_path)
        
        if success:
            print("=" * 50)
            print("部署完成!")
            print(f"运行命令: ollama run {self.model_name}")
            print("=" * 50)
        
        return success