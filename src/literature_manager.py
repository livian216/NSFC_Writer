"""文献管理模块 - 解析文献并构建RAG系统"""

# ===== 环境配置 =====
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# ===== 正常导入 =====
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from .config import get_config


@dataclass
class LiteratureContent:
    """文献内容"""
    title: str
    authors: List[str]
    abstract: str
    sections: Dict[str, str]
    references: List[str]
    full_text: str
    source_file: str
    file_type: str


@dataclass
class RetrievedContext:
    """检索结果"""
    content: str
    source: str
    section: str
    score: float


class BaseParser(ABC):
    """文献解析器基类"""
    
    @abstractmethod
    def parse(self, file_path: str) -> LiteratureContent:
        pass
    
    def extract_title(self, text: str) -> str:
        lines = text.strip().split('\n')
        for line in lines[:5]:
            line = line.strip()
            if 10 < len(line) < 200:
                return line
        return "未知标题"
    
    def extract_abstract(self, text: str) -> str:
        patterns = [
            r'Abstract[:\s]*(.+?)(?=\n\n|Introduction|Keywords)',
            r'摘\s*要[:\s]*(.+?)(?=\n\n|关键词|1\s|一、)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""


class PDFParser(BaseParser):
    """PDF解析器"""
    
    def parse(self, file_path: str) -> LiteratureContent:
        import fitz  # PyMuPDF
        
        doc = fitz.open(file_path)
        full_text = ""
        
        for page in doc:
            full_text += page.get_text()
        
        doc.close()
        
        return LiteratureContent(
            title=self.extract_title(full_text),
            authors=[],
            abstract=self.extract_abstract(full_text),
            sections={},
            references=[],
            full_text=full_text,
            source_file=file_path,
            file_type="pdf"
        )


class DocxParser(BaseParser):
    """Word解析器"""
    
    def parse(self, file_path: str) -> LiteratureContent:
        from docx import Document
        
        doc = Document(file_path)
        
        full_text = ""
        sections = {}
        current_section = "正文"
        current_content = []
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            
            if para.style.name.startswith('Heading'):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = text
                current_content = []
            else:
                current_content.append(text)
            
            full_text += text + '\n'
        
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return LiteratureContent(
            title=self.extract_title(full_text),
            authors=[],
            abstract=self.extract_abstract(full_text),
            sections=sections,
            references=[],
            full_text=full_text,
            source_file=file_path,
            file_type="docx"
        )


class MarkdownParser(BaseParser):
    """Markdown解析器"""
    
    def parse(self, file_path: str) -> LiteratureContent:
        import markdown
        from bs4 import BeautifulSoup
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        full_text = soup.get_text()
        
        sections = {}
        current_section = None
        current_content = []
        
        for element in soup.children:
            if element.name in ['h1', 'h2', 'h3']:
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = element.get_text().strip()
                current_content = []
            elif hasattr(element, 'get_text'):
                if current_section:
                    current_content.append(element.get_text().strip())
        
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return LiteratureContent(
            title=self.extract_title(full_text),
            authors=[],
            abstract=self.extract_abstract(full_text),
            sections=sections,
            references=[],
            full_text=full_text,
            source_file=file_path,
            file_type="markdown"
        )


class LiteratureManager:
    """文献管理器"""
    
    SUPPORTED_FORMATS = {
        '.pdf': PDFParser,
        '.docx': DocxParser,
        '.doc': DocxParser,
        '.md': MarkdownParser,
        '.markdown': MarkdownParser,
        '.txt': MarkdownParser,
    }
    
    def __init__(self, lazy_init: bool = True):
        self.config = get_config()
        self.parsers = {}
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self._initialized = False
        
        if not lazy_init:
            self._init_vector_db()
    
    def _ensure_initialized(self):
        """确保向量数据库已初始化"""
        if not self._initialized:
            self._init_vector_db()
    
    def _init_vector_db(self):
        """初始化向量数据库"""
        if self._initialized:
            return
            
        # 延迟导入
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer
        
        print(f"加载嵌入模型: {self.config.literature.embedding_model}")
        self.embedding_model = SentenceTransformer(
            self.config.literature.embedding_model
        )
        
        # 确保目录存在
        db_path = self.config.paths.literature_db
        Path(db_path).mkdir(parents=True, exist_ok=True)
        
        # 创建客户端
        self.chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        
        # 获取或创建集合
        self.collection = self.chroma_client.get_or_create_collection(
            name="nsfc_literature",
            metadata={"description": "国自然参考文献库"}
        )
        
        self._initialized = True
    
    def get_parser(self, file_path: str) -> BaseParser:
        """获取对应的解析器"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的格式: {ext}，支持: {list(self.SUPPORTED_FORMATS.keys())}")
        
        if ext not in self.parsers:
            self.parsers[ext] = self.SUPPORTED_FORMATS[ext]()
        
        return self.parsers[ext]
    
    def parse_file(self, file_path: str) -> LiteratureContent:
        """解析单个文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        parser = self.get_parser(file_path)
        return parser.parse(file_path)
    
    def _chunk_text(self, text: str) -> List[str]:
        """分割文本"""
        chunk_size = self.config.literature.chunk_size
        overlap = self.config.literature.chunk_overlap
        
        # 清理文本
        text = text.strip()
        if not text:
            return []
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end < len(text):
                # 尝试在句子边界分割
                for sep in ['。', '！', '？', '.', '!', '?', '\n\n', '\n']:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > chunk_size // 2:  # 确保块不会太小
                        end = start + last_sep + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 20:  # 过滤太短的块
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def add_literature(self, file_path: str) -> int:
        """添加文献到向量库"""
        self._ensure_initialized()
        
        # 转换为绝对路径
        file_path = os.path.abspath(file_path)
        
        print(f"解析文件: {os.path.basename(file_path)}")
        literature = self.parse_file(file_path)
        
        documents = []
        metadatas = []
        ids = []
        
        doc_id_base = abs(hash(file_path)) % 100000000
        
        # 处理摘要
        if literature.abstract and len(literature.abstract) > 20:
            chunks = self._chunk_text(literature.abstract)
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({
                    "source": file_path,
                    "title": literature.title[:100],  # 限制长度
                    "section": "摘要"
                })
                ids.append(f"doc{doc_id_base}_abs_{i}")
        
        # 处理全文
        if literature.full_text:
            chunks = self._chunk_text(literature.full_text)
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({
                    "source": file_path,
                    "title": literature.title[:100],
                    "section": "正文"
                })
                ids.append(f"doc{doc_id_base}_txt_{i}")
        
        # 添加到数据库
        if documents:
            print(f"  生成嵌入向量...")
            embeddings = self.embedding_model.encode(documents).tolist()
            
            print(f"  写入数据库...")
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        
        return len(documents)
    
    def add_files(self, file_paths: List[str]) -> Dict[str, int]:
        """批量添加文献"""
        results = {}
        
        for path in file_paths:
            try:
                count = self.add_literature(path)
                results[path] = count
                print(f"✓ {os.path.basename(path)}: {count} 个文本块")
            except Exception as e:
                results[path] = 0
                print(f"✗ {os.path.basename(path)}: {str(e)}")
        
        return results
    
    def add_directory(self, dir_path: str) -> Dict[str, int]:
        """添加目录中的所有文献"""
        dir_path = os.path.abspath(dir_path)
        
        if not os.path.exists(dir_path):
            print(f"目录不存在: {dir_path}")
            return {}
        
        file_paths = []
        for ext in self.SUPPORTED_FORMATS.keys():
            file_paths.extend(Path(dir_path).rglob(f"*{ext}"))
        
        if not file_paths:
            print(f"目录中没有支持的文献文件")
            return {}
        
        print(f"发现 {len(file_paths)} 个文献文件")
        return self.add_files([str(p) for p in file_paths])
    
    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedContext]:
        """检索相关内容"""
        self._ensure_initialized()
        
        top_k = top_k or self.config.literature.top_k
        
        # 检查是否有数据
        if self.collection.count() == 0:
            return []
        
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count())
        )
        
        contexts = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                contexts.append(RetrievedContext(
                    content=results['documents'][0][i],
                    source=results['metadatas'][0][i]['source'],
                    section=results['metadatas'][0][i]['section'],
                    score=1 - results['distances'][0][i] if results['distances'] else 0.5
                ))
        
        return contexts
    
    def build_context(self, query: str, top_k: int = None) -> str:
        """构建上下文文本"""
        contexts = self.retrieve(query, top_k)
        
        if not contexts:
            return ""
        
        parts = ["【参考文献内容】"]
        
        for i, ctx in enumerate(contexts, 1):
            source_name = os.path.basename(ctx.source)
            parts.append(f"\n[来源{i}] {source_name}")
            parts.append(f"{ctx.content[:500]}...")  # 限制长度
            parts.append("-" * 30)
        
        return "\n".join(parts)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        self._ensure_initialized()
        
        count = self.collection.count()
        
        sources = set()
        if count > 0:
            all_data = self.collection.get(limit=count)
            if all_data['metadatas']:
                for metadata in all_data['metadatas']:
                    sources.add(metadata.get('source', ''))
        
        return {
            "total_chunks": count,
            "total_documents": len(sources)
        }
    
    def clear_all(self):
        """清空文献库"""
        self._ensure_initialized()
        
        # 删除并重建集合
        self.chroma_client.delete_collection("nsfc_literature")
        self.collection = self.chroma_client.create_collection(
            name="nsfc_literature",
            metadata={"description": "国自然参考文献库"}
        )
        
        print("文献库已清空")
    
    @classmethod
    def supported_formats(cls) -> List[str]:
        """返回支持的格式"""
        return list(cls.SUPPORTED_FORMATS.keys())