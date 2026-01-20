"""数据处理模块 - 处理Markdown文件生成训练数据"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import markdown
from bs4 import BeautifulSoup
from tqdm import tqdm

from .config import get_config


@dataclass
class NSFCSection:
    """国自然申请书模块数据"""
    section_type: str
    title: str
    content: str
    references: List[str]
    source_file: str
    quality_score: float = 0.0


@dataclass
class TrainingSample:
    """训练样本"""
    instruction: str
    input: str
    output: str
    section_type: str
    source: str


class DataProcessor:
    """Markdown数据处理器"""
    
    # 模块识别模式
    SECTION_PATTERNS = {
        "立项依据": [
            r"立项依据", r"研究背景", r"研究意义", r"国内外研究现状",
            r"研究现状", r"文献综述", r"背景与意义"
        ],
        "研究内容": [
            r"研究内容", r"研究问题", r"研究要点", r"研究范围",
            r"核心问题", r"主要内容"
        ],
        "研究方案": [
            r"研究方案", r"技术路线", r"研究方法", r"实验设计",
            r"研究步骤", r"实施方案", r"方法与技术"
        ],
        "创新点": [
            r"创新点", r"创新之处", r"特色与创新", r"创新性",
            r"理论创新", r"方法创新", r"应用创新"
        ],
        "预期成果": [
            r"预期成果", r"研究成果", r"预期产出", r"成果形式",
            r"考核指标", r"成果指标"
        ],
        "研究基础": [
            r"研究基础", r"工作基础", r"前期成果", r"研究积累",
            r"团队基础", r"实验条件"
        ]
    }
    
    # 指令模板
    INSTRUCTION_TEMPLATES = {
        "立项依据": [
            "请根据以下研究主题，撰写国自然申请书的立项依据部分，需包含研究背景、国内外研究现状、现有研究缺口及本研究的必要性。",
            "基于提供的研究方向，撰写一份完整的立项依据，要求逻辑清晰、引用规范。",
            "请为以下研究课题撰写立项依据，需要阐明研究的科学意义和应用价值。"
        ],
        "研究内容": [
            "请根据研究目标，撰写详细的研究内容，需明确核心问题、研究要点和范围界定。",
            "基于以下研究主题，规划具体的研究内容，要求层次分明、可操作性强。",
            "请设计本研究的核心研究内容，需紧扣研究目标，逻辑严密。"
        ],
        "研究方案": [
            "请设计详细的研究方案，包括研究方法、技术路线、实施步骤和进度安排。",
            "基于研究内容，制定可行的研究方案，需说明关键技术和解决策略。",
            "请撰写研究方案部分，要求方法科学、路线清晰、进度合理。"
        ],
        "创新点": [
            "请提炼本研究的创新点，需区分理论创新、方法创新和应用创新。",
            "基于研究内容，总结本研究的核心创新之处，需说明其独特性和先进性。",
            "请撰写创新点部分，要求精准凝练、突出特色、避免泛泛而谈。"
        ],
        "预期成果": [
            "请设计本研究的预期成果，包括论文、专利、技术方案等具体产出。",
            "基于研究目标，规划可量化的预期成果和考核指标。",
            "请撰写预期成果部分，需说明成果的应用价值和推广前景。"
        ],
        "研究基础": [
            "请撰写研究基础部分，阐述团队的前期积累和完成研究的能力。",
            "基于团队情况，说明开展本研究的研究基础和条件保障。",
            "请撰写研究基础，需包含前期成果、实验条件、团队构成等。"
        ]
    }
    
    def __init__(self, min_content_length: int = 100, quality_threshold: float = 0.5):
        self.config = get_config()
        self.min_content_length = min_content_length
        self.quality_threshold = quality_threshold
        self.processed_hashes = set()
    
    def process_markdown_file(self, file_path: str) -> List[NSFCSection]:
        """处理单个Markdown文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 转换为HTML解析
        html = markdown.markdown(content, extensions=['tables', 'fenced_code'])
        soup = BeautifulSoup(html, 'html.parser')
        
        sections = []
        current_section = None
        current_content = []
        
        for element in soup.children:
            if element.name in ['h1', 'h2', 'h3']:
                if current_section and current_content:
                    section = self._create_section(
                        current_section,
                        '\n'.join(current_content),
                        file_path
                    )
                    if section:
                        sections.append(section)
                
                current_section = element.get_text().strip()
                current_content = []
            elif element.name in ['p', 'ul', 'ol', 'blockquote']:
                if current_section:
                    current_content.append(element.get_text().strip())
        
        # 处理最后一个section
        if current_section and current_content:
            section = self._create_section(
                current_section,
                '\n'.join(current_content),
                file_path
            )
            if section:
                sections.append(section)
        
        return sections
    
    def _create_section(self, title: str, content: str, source: str) -> Optional[NSFCSection]:
        """创建模块对象"""
        section_type = self._identify_section_type(title)
        if not section_type:
            return None
        
        if len(content) < self.min_content_length:
            return None
        
        # 去重
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in self.processed_hashes:
            return None
        self.processed_hashes.add(content_hash)
        
        references = self._extract_references(content)
        quality_score = self._calculate_quality_score(content, section_type)
        
        return NSFCSection(
            section_type=section_type,
            title=title,
            content=content,
            references=references,
            source_file=source,
            quality_score=quality_score
        )
    
    def _identify_section_type(self, title: str) -> Optional[str]:
        """识别模块类型"""
        for section_type, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, title, re.IGNORECASE):
                    return section_type
        return None
    
    def _extract_references(self, content: str) -> List[str]:
        """提取引用"""
        references = []
        patterns = [
            r'\[(\d+)\]',
            r'\(([A-Z][a-z]+\s+et\s+al\.,?\s*\d{4})\)',
            r'（([^）]+\d{4}[^）]*)）',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            references.extend(matches)
        
        return list(set(references))
    
    def _calculate_quality_score(self, content: str, section_type: str) -> float:
        """计算质量分数"""
        score = 0.0
        
        # 长度分数
        length = len(content)
        if length > 500:
            score += 0.3
        elif length > 200:
            score += 0.2
        else:
            score += 0.1
        
        # 结构分数
        if re.search(r'[（(][一二三四五六七八九十\d][）)]', content):
            score += 0.2
        
        # 专业性分数
        academic_terms = ['研究', '分析', '方法', '理论', '机制', '模型', '实验', '数据']
        term_count = sum(1 for term in academic_terms if term in content)
        score += min(0.3, term_count * 0.05)
        
        # 引用分数
        ref_count = len(re.findall(r'\[\d+\]', content))
        if ref_count > 0:
            score += min(0.2, ref_count * 0.02)
        
        return min(1.0, score)
    
    def process_directory(self, dir_path: str = None) -> List[NSFCSection]:
        """处理目录中的所有Markdown文件"""
        dir_path = dir_path or self.config.paths.raw_data
        all_sections = []
        
        md_files = list(Path(dir_path).rglob("*.md"))
        
        print(f"发现 {len(md_files)} 个Markdown文件")
        
        for file_path in tqdm(md_files, desc="处理文件"):
            try:
                sections = self.process_markdown_file(str(file_path))
                all_sections.extend(sections)
            except Exception as e:
                print(f"处理失败 {file_path}: {e}")
        
        print(f"共提取 {len(all_sections)} 个有效模块")
        return all_sections
    
    def generate_training_samples(self, sections: List[NSFCSection]) -> List[TrainingSample]:
        """生成训练样本"""
        samples = []
        
        for section in sections:
            if section.quality_score < self.quality_threshold:
                continue
            
            templates = self.INSTRUCTION_TEMPLATES.get(section.section_type, [])
            
            for template in templates:
                input_context = self._build_input_context(section)
                
                sample = TrainingSample(
                    instruction=template,
                    input=input_context,
                    output=section.content,
                    section_type=section.section_type,
                    source=section.source_file
                )
                samples.append(sample)
        
        return samples
    
    def _build_input_context(self, section: NSFCSection) -> str:
        """构建输入上下文"""
        parts = []
        
        if section.title and section.title != section.section_type:
            parts.append(f"研究主题：{section.title}")
        
        if section.references:
            refs_str = "、".join(section.references[:5])
            parts.append(f"参考文献：{refs_str}")
        
        return "\n".join(parts) if parts else "请根据学术规范撰写。"
    
    def save_training_data(
        self,
        samples: List[TrainingSample],
        output_path: str = None,
        format: str = "alpaca"
    ):
        """保存训练数据"""
        output_path = output_path or os.path.join(
            self.config.paths.processed_data, "train_data.json"
        )
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == "alpaca":
            data = [{"instruction": s.instruction, "input": s.input, "output": s.output}
                    for s in samples]
        else:
            data = [asdict(s) for s in samples]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 {len(samples)} 条训练数据到 {output_path}")
        return output_path
    
    def run(self, input_dir: str = None, output_path: str = None) -> str:
        """完整运行流程"""
        print("=" * 50)
        print("开始处理训练数据")
        print("=" * 50)
        
        # 处理Markdown文件
        sections = self.process_directory(input_dir)
        
        # 统计模块分布
        section_counts = {}
        for section in sections:
            section_counts[section.section_type] = section_counts.get(section.section_type, 0) + 1
        
        print("\n模块分布:")
        for st, count in section_counts.items():
            print(f"  - {st}: {count}")
        
        # 生成训练样本
        samples = self.generate_training_samples(sections)
        print(f"\n生成 {len(samples)} 条训练样本")
        
        # 保存
        output_path = self.save_training_data(samples, output_path)
        
        print("=" * 50)
        print("数据处理完成")
        print("=" * 50)
        
        return output_path