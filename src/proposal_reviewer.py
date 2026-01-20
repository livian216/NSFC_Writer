"""标书审阅模块 - 分析和修改用户上传的标书"""

import os
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .config import get_config
from .literature_manager import PDFParser, DocxParser, LiteratureContent


@dataclass
class ReviewResult:
    """审阅结果"""
    section_name: str
    original_content: str
    issues: List[str]
    suggestions: List[str]
    revised_content: str
    score: int  # 1-10分


class ProposalReviewer:
    """标书审阅器"""
    
    # 各模块的审阅标准
    REVIEW_CRITERIA = {
        "立项依据": {
            "keywords": ["研究背景", "国内外", "研究现状", "意义", "必要性", "科学问题"],
            "requirements": [
                "是否清晰阐述研究背景",
                "是否全面综述国内外研究现状",
                "是否明确指出研究缺口",
                "是否论证研究的必要性和意义",
                "是否有规范的文献引用",
                "逻辑是否清晰连贯"
            ]
        },
        "研究内容": {
            "keywords": ["研究内容", "研究问题", "核心问题", "研究要点"],
            "requirements": [
                "是否明确核心研究问题",
                "研究内容是否具体可操作",
                "是否有清晰的层次结构",
                "是否紧扣研究目标",
                "研究范围是否适当"
            ]
        },
        "研究方案": {
            "keywords": ["技术路线", "研究方法", "实验", "步骤", "进度"],
            "requirements": [
                "技术路线是否清晰",
                "研究方法是否科学合理",
                "是否有详细的实施步骤",
                "是否说明关键技术难点",
                "进度安排是否合理",
                "是否论证可行性"
            ]
        },
        "创新点": {
            "keywords": ["创新", "特色", "首次", "新方法", "新理论"],
            "requirements": [
                "创新点是否明确具体",
                "是否区分创新类型",
                "是否与现有研究对比",
                "创新点是否有实质内容",
                "是否避免夸大其词"
            ]
        },
        "预期成果": {
            "keywords": ["成果", "论文", "专利", "指标", "应用"],
            "requirements": [
                "成果指标是否具体可量化",
                "是否包含学术成果",
                "是否说明应用价值",
                "指标是否合理可达",
                "是否有人才培养计划"
            ]
        },
        "研究基础": {
            "keywords": ["基础", "前期", "团队", "条件", "积累"],
            "requirements": [
                "是否展示相关研究积累",
                "是否介绍团队构成",
                "是否说明实验条件",
                "是否论证完成能力",
                "前期成果是否相关"
            ]
        }
    }
    
    # 审阅提示词模板
    REVIEW_PROMPT = """你是一位资深的国家自然科学基金评审专家。请对以下申请书的"{section_name}"部分进行专业评审。

【原文内容】
{content}

【评审要求】
请从以下几个方面进行评审：
{requirements}

【输出格式】
请按以下格式输出：

## 评分：X/10

## 主要问题
1. 问题一
2. 问题二
...

## 修改建议
1. 建议一
2. 建议二
...

## 修改后的完整内容
（请输出修改后的完整内容，保持学术规范性）
"""
    
    def __init__(self, generator=None):
        self.config = get_config()
        self.generator = generator
        self.pdf_parser = PDFParser()
        self.docx_parser = DocxParser()
    
    def parse_proposal(self, file_path: str) -> Dict[str, str]:
        """解析上传的标书文件"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            content = self.pdf_parser.parse(file_path)
        elif ext in ['.docx', '.doc']:
            content = self.docx_parser.parse(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        # 提取各模块内容
        sections = self._extract_sections(content.full_text)
        
        return sections
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """从全文中提取各模块"""
        sections = {}
        
        # 定义模块的可能标题
        section_patterns = {
            "立项依据": [
                r"[一1１][\s、\.．]+立项依据",
                r"立项依据[与和及]研究内容",
                r"项目的立项依据",
                r"研究背景[与和及]?意义"
            ],
            "研究内容": [
                r"[二2２][\s、\.．]+研究内容",
                r"主要研究内容",
                r"研究内容[与和及]目标"
            ],
            "研究方案": [
                r"[三3３][\s、\.．]+研究方案",
                r"研究方案[与和及]可行性",
                r"技术路线",
                r"研究方法"
            ],
            "创新点": [
                r"[四4４][\s、\.．]+创新点",
                r"特色[与和及]创新",
                r"创新之处",
                r"项目创新"
            ],
            "预期成果": [
                r"[五5５][\s、\.．]+预期成果",
                r"预期研究成果",
                r"预期目标",
                r"考核指标"
            ],
            "研究基础": [
                r"[六6６][\s、\.．]+研究基础",
                r"工作基础",
                r"研究基础[与和及]工作条件",
                r"前期工作"
            ]
        }
        
        # 查找各模块的位置
        section_positions = []
        
        for section_name, patterns in section_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    section_positions.append((matches[0].start(), section_name, matches[0].group()))
                    break
        
        # 按位置排序
        section_positions.sort(key=lambda x: x[0])
        
        # 提取各模块内容
        for i, (pos, name, title) in enumerate(section_positions):
            start = pos
            if i + 1 < len(section_positions):
                end = section_positions[i + 1][0]
            else:
                end = len(text)
            
            content = text[start:end].strip()
            # 移除标题
            content = re.sub(f"^{re.escape(title)}", "", content).strip()
            
            if len(content) > 50:  # 确保有实质内容
                sections[name] = content
        
        # 如果没有识别到模块，将整个文本作为"全文"
        if not sections:
            sections["全文"] = text
        
        return sections
    
    def review_section(
        self,
        section_name: str,
        content: str,
        use_model: bool = True
    ) -> ReviewResult:
        """审阅单个模块"""
        
        if use_model and self.generator:
            return self._review_with_model(section_name, content)
        else:
            return self._review_rule_based(section_name, content)
    
    def _review_with_model(self, section_name: str, content: str) -> ReviewResult:
        """使用模型审阅"""
        
        # 获取审阅标准
        criteria = self.REVIEW_CRITERIA.get(section_name, self.REVIEW_CRITERIA["立项依据"])
        requirements = "\n".join([f"- {r}" for r in criteria["requirements"]])
        
        # 构建提示词
        prompt = self.REVIEW_PROMPT.format(
            section_name=section_name,
            content=content[:3000],  # 限制长度
            requirements=requirements
        )
        
        system_prompt = "你是一位资深的国家自然科学基金评审专家，具有丰富的项目评审经验。请提供专业、具体、建设性的评审意见。"
        
        # 调用模型
        try:
            response = self.generator._generate(prompt, system_prompt)
            return self._parse_review_response(section_name, content, response)
        except Exception as e:
            print(f"模型审阅失败: {e}")
            return self._review_rule_based(section_name, content)
    
    def _parse_review_response(
        self,
        section_name: str,
        original: str,
        response: str
    ) -> ReviewResult:
        """解析模型的审阅响应"""
        
        # 提取评分
        score_match = re.search(r"评分[：:]\s*(\d+)", response)
        score = int(score_match.group(1)) if score_match else 6
        
        # 提取问题
        issues = []
        issues_section = re.search(r"主要问题[：:]?\s*([\s\S]*?)(?=##|修改建议|$)", response)
        if issues_section:
            issue_matches = re.findall(r"\d+[\.\、]\s*(.+?)(?=\n\d+[\.\、]|\n\n|$)", issues_section.group(1))
            issues = [m.strip() for m in issue_matches if m.strip()]
        
        # 提取建议
        suggestions = []
        suggestions_section = re.search(r"修改建议[：:]?\s*([\s\S]*?)(?=##|修改后|$)", response)
        if suggestions_section:
            suggestion_matches = re.findall(r"\d+[\.\、]\s*(.+?)(?=\n\d+[\.\、]|\n\n|$)", suggestions_section.group(1))
            suggestions = [m.strip() for m in suggestion_matches if m.strip()]
        
        # 提取修改后的内容
        revised_section = re.search(r"修改后的完整内容[：:]?\s*([\s\S]*?)$", response)
        revised_content = revised_section.group(1).strip() if revised_section else response
        
        return ReviewResult(
            section_name=section_name,
            original_content=original,
            issues=issues if issues else ["未能提取具体问题"],
            suggestions=suggestions if suggestions else ["未能提取具体建议"],
            revised_content=revised_content,
            score=min(10, max(1, score))
        )
    
    def _review_rule_based(self, section_name: str, content: str) -> ReviewResult:
        """基于规则的简单审阅"""
        
        criteria = self.REVIEW_CRITERIA.get(section_name, self.REVIEW_CRITERIA["立项依据"])
        
        issues = []
        suggestions = []
        score = 7
        
        # 检查长度
        if len(content) < 500:
            issues.append("内容过于简短，缺乏详细论述")
            suggestions.append("建议扩充内容，增加具体细节和论证")
            score -= 1
        
        # 检查关键词
        missing_keywords = []
        for kw in criteria["keywords"]:
            if kw not in content:
                missing_keywords.append(kw)
        
        if missing_keywords:
            issues.append(f"缺少关键内容：{', '.join(missing_keywords)}")
            suggestions.append(f"建议补充以下方面的内容：{', '.join(missing_keywords)}")
            score -= len(missing_keywords) * 0.5
        
        # 检查结构
        if not re.search(r"[（(][一二三四五1-5][）)]", content):
            issues.append("缺乏清晰的层次结构")
            suggestions.append("建议使用（1）（2）（3）等方式组织内容层次")
        
        # 检查引用（仅对立项依据）
        if section_name == "立项依据":
            if not re.search(r"\[\d+\]", content):
                issues.append("缺少文献引用标注")
                suggestions.append("建议添加规范的文献引用，如[1]、[2]等")
                score -= 1
        
        return ReviewResult(
            section_name=section_name,
            original_content=content,
            issues=issues if issues else ["内容基本符合要求"],
            suggestions=suggestions if suggestions else ["可进一步优化细节"],
            revised_content=content,  # 规则审阅不修改内容
            score=max(1, min(10, int(score)))
        )
    
    def review_full_proposal(
        self,
        file_path: str,
        use_model: bool = True
    ) -> Dict[str, ReviewResult]:
        """审阅完整标书"""
        
        # 解析标书
        sections = self.parse_proposal(file_path)
        
        # 审阅各模块
        results = {}
        for section_name, content in sections.items():
            print(f"正在审阅: {section_name}...")
            result = self.review_section(section_name, content, use_model)
            results[section_name] = result
            print(f"✓ {section_name} 评分: {result.score}/10")
        
        return results
    
    def generate_review_report(self, results: Dict[str, ReviewResult]) -> str:
        """生成审阅报告"""
        
        lines = ["# 📋 国自然申请书审阅报告\n"]
        lines.append("---\n")
        
        # 总体评分
        total_score = sum(r.score for r in results.values())
        avg_score = total_score / len(results) if results else 0
        
        lines.append(f"## 📊 总体评价\n")
        lines.append(f"- **平均得分**: {avg_score:.1f}/10\n")
        lines.append(f"- **审阅模块数**: {len(results)}\n\n")
        
        # 各模块详情
        for section_name, result in results.items():
            lines.append(f"## 📝 {section_name}\n")
            lines.append(f"**评分: {result.score}/10**\n\n")
            
            lines.append("### 发现的问题\n")
            for issue in result.issues:
                lines.append(f"- {issue}\n")
            
            lines.append("\n### 修改建议\n")
            for suggestion in result.suggestions:
                lines.append(f"- {suggestion}\n")
            
            lines.append("\n---\n")
        
        return "\n".join(lines)
    
    def generate_revised_proposal(self, results: Dict[str, ReviewResult]) -> str:
        """生成修改后的完整标书"""
        
        lines = ["# 国自然科学基金申请书（修改版）\n"]
        lines.append("---\n")
        
        section_order = ["立项依据", "研究内容", "研究方案", "创新点", "预期成果", "研究基础"]
        
        for section_name in section_order:
            if section_name in results:
                result = results[section_name]
                lines.append(f"\n## {section_name}\n")
                lines.append(result.revised_content)
                lines.append("\n")
        
        # 处理不在标准顺序中的模块
        for section_name, result in results.items():
            if section_name not in section_order:
                lines.append(f"\n## {section_name}\n")
                lines.append(result.revised_content)
                lines.append("\n")
        
        return "\n".join(lines)