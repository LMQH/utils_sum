"""
实现多种文本分块方式
本模块实现输入一段字符串文本以及选择的分块方式，输出一个完成分块的字符串列表

目前已知分块方式：
- 固定大小分块
- 递归字符分块
- 语义分块
- 基于文档结构的分块
- 段落分块（按双换行符）
- JSON行分块
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import List, Dict
import re
import numpy as np
import logging

logger = logging.getLogger("Web_API")

class TextSpliter:
    def __init__(self,
            text: str,
            api_key: str,
            separators: List[str],
            structure_markers: Dict[str, list],
            chunk_size: int = 500,
            overlap: int = 100,
            embedding_model: str = "bge-m3",
            base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name: str = "qwen-max"):

        self.text = text
        self.api_key = api_key
        self.base_url = base_url
        self.separators = separators
        self.structure_markers = structure_markers
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedding_model = embedding_model
        self.model_name = model_name

    def choose_chunk_method(self, chunk_method: str) -> List[str]:
        """
        选择分块方式
        :param chunk_method: 分块方式
        :return: 分块后的字符串列表
        """
        if chunk_method == 'fixed_size':
            return self.fixed_size_split()
        elif chunk_method == 'structure':
            return self.structure_split()
        elif chunk_method == 'recursive_character':
            return self.recursive_character_split()
        elif chunk_method == 'semantic':
            return self.semantic_split()
        elif chunk_method == 'json_line':
            return self.json_line_split()
        else:
            raise ValueError(f"未知的分块方法: {chunk_method}")

    # 固定大小分块
    def fixed_size_split(self) -> List[str]:
        """
        固定大小分块
        :return:
        """
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
        )
        return text_splitter.split_text(self.text)

    # 递归字符分块
    def recursive_character_split(self) -> List[str]:
        """
        递归字符分块
        :return:
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=self.separators
        )
        return text_splitter.split_text(self.text)

    # 语义分块 - API编码方法
    def _encode_with_api(self, sentences: List[str], embedding_dim: int) -> np.ndarray:
        """
        使用API模型对句子进行编码（内部方法）
        
        Args:
            sentences: 句子列表
            embedding_dim: 嵌入向量维度
        
        Returns:
            嵌入向量数组
        
        Raises:
            Exception: API调用失败时抛出异常
        """
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        embeddings = []
        
        # API限制：输入长度范围 [1, 8192]（字符数）
        MAX_INPUT_LENGTH = 8000  # 留一些余量，避免边界问题
        
        def split_long_text(text: str, max_length: int) -> List[str]:
            """将超长文本分割成多个部分"""
            if len(text) <= max_length:
                return [text]
            
            parts = []
            # 尝试在句号、换行符等位置分割
            separators = ['。', '.\n', '.\r\n', '.\r', '\n\n', '\n']
            current_text = text
            
            while len(current_text) > max_length:
                # 找到最后一个分隔符的位置（在max_length范围内）
                best_pos = -1
                best_sep = None
                
                for sep in separators:
                    # 从max_length位置向前查找分隔符
                    search_start = max(0, max_length - 1000)  # 在max_length前1000字符内查找
                    pos = current_text.rfind(sep, search_start, max_length)
                    if pos > best_pos:
                        best_pos = pos
                        best_sep = sep
                
                if best_pos > 0:
                    # 在分隔符后分割
                    part = current_text[:best_pos + len(best_sep)]
                    parts.append(part.strip())
                    current_text = current_text[best_pos + len(best_sep):].strip()
                else:
                    # 如果找不到合适的分隔符，直接按长度截断
                    part = current_text[:max_length]
                    parts.append(part.strip())
                    current_text = current_text[max_length:].strip()
            
            if current_text:
                parts.append(current_text)
            
            return parts
        
        for idx, sentence in enumerate(sentences):
            # 检查句子长度
            if len(sentence) > MAX_INPUT_LENGTH:
                logger.warning(f"句子 {idx+1} 长度 {len(sentence)} 超过限制 {MAX_INPUT_LENGTH}，将进行分割")
                sentence_parts = split_long_text(sentence, MAX_INPUT_LENGTH)
                
                # 对每个部分分别调用API，然后平均嵌入向量
                part_embeddings = []
                for part in sentence_parts:
                    completion = client.embeddings.create(
                        model=self.embedding_model,
                        input=part,
                        dimensions=embedding_dim,
                        encoding_format="float"
                    )
                    part_embedding = completion.to_dict()['data'][0]['embedding']
                    part_embeddings.append(part_embedding)
                
                # 将多个部分的嵌入向量平均（加权平均，按长度加权）
                if len(part_embeddings) == 1:
                    embedding = np.array(part_embeddings[0])
                else:
                    # 按长度加权平均
                    weights = [len(part) for part in sentence_parts]
                    total_weight = sum(weights)
                    weighted_embeddings = [np.array(emb) * w / total_weight 
                                          for emb, w in zip(part_embeddings, weights)]
                    embedding = np.sum(weighted_embeddings, axis=0)
            else:
                # 正常长度的句子，直接调用API
                completion = client.embeddings.create(
                    model=self.embedding_model,
                    input=sentence,
                    dimensions=embedding_dim,
                    encoding_format="float"
                )
                embedding = np.array(completion.to_dict()['data'][0]['embedding'])
            
            embeddings.append(embedding)
        
        return np.array(embeddings)

    # 语义分块
    def semantic_split(self, similarity_threshold: float = 0.7, embedding_dim: int = 1024) -> List[str]:
        """
        语义分块：基于语义相似度将句子分组
        
        原理：
        1. 将文本按句号分割成句子
        2. 使用嵌入模型对每个句子进行编码
        3. 计算相邻句子之间的余弦相似度
        4. 当相似度低于阈值或累积字符数达到chunk_size时，创建新的块
        
        Args:
            similarity_threshold: 语义相似度阈值，低于此值则创建新块（默认0.7）
            embedding_dim: 嵌入向量维度，仅用于API模型（默认1024）
        
        Returns:
            分块后的文本列表
        """
        # 检查模型名是否为API模型名
        api_model_names = ["text-embedding-v4", "text-embedding-v3", "text-embedding-v2", "text-embedding"]
        is_api_model = self.embedding_model.lower() in [name.lower() for name in api_model_names]
        
        # 按句号分割句子，保留句号
        sentences = []
        parts = self.text.split('.')
        for i, part in enumerate(parts):
            if part.strip():
                # 如果不是最后一个部分，添加句号
                sentence = part.strip() + ('.' if i < len(parts) - 1 else '')
                sentences.append(sentence)
        
        if not sentences:
            return [self.text]
        
        # 对每个句子进行编码
        # 优先使用API模型，失败时回退到本地模型
        embeddings = None
        use_api = False
        
        if is_api_model and self.api_key:
            # 尝试使用API模型
            try:
                logger.info(f"尝试使用API模型 '{self.embedding_model}' 进行语义分块")
                embeddings = self._encode_with_api(sentences, embedding_dim)
                use_api = True
                logger.info(f"API模型调用成功，使用 '{self.embedding_model}' 完成语义分块")
            except Exception as api_error:
                logger.warning(f"API模型调用失败: {api_error}，将回退到本地模型")
                # API调用失败，回退到本地模型
                embeddings = None
        
        # 如果API未使用或失败，使用本地模型
        if embeddings is None:
            try:
                # 如果配置的是API模型名，尝试使用默认的本地模型
                if is_api_model:
                    fallback_model = "BAAI/bge-m3"
                    logger.info(f"使用本地模型 '{fallback_model}' 进行语义分块（API模型 '{self.embedding_model}' 不可用）")
                else:
                    fallback_model = self.embedding_model
                    logger.info(f"使用本地模型 '{fallback_model}' 进行语义分块")
                
                model = SentenceTransformer(fallback_model)
                embeddings = model.encode(sentences, normalize_embeddings=True)
                embeddings = np.array(embeddings)
            except Exception as e:
                logger.error(f"加载本地模型失败: {e}")
                # 如果加载失败，尝试使用备选模型
                try:
                    semantic_model = "BAAI/bge-m3"
                    logger.info(f"尝试使用备选模型: {semantic_model}")
                    model = SentenceTransformer(semantic_model)
                    embeddings = model.encode(sentences, normalize_embeddings=True)
                    embeddings = np.array(embeddings)
                except Exception as e2:
                    logger.error(f"加载备选模型也失败: {e2}")
                    raise ValueError(f"无法加载任何嵌入模型。API调用失败，本地模型也加载失败。错误: {e2}")
        
        # 确保embeddings是numpy数组
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        # 归一化嵌入向量
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除零
        embeddings = embeddings / norms
        
        # 继续后续的分块逻辑
        chunks = []
        current_chunk_text = []
        current_size = 0
        
        # 计算余弦相似度的辅助函数
        def cosine_similarity(vec1, vec2):
            """计算两个向量的余弦相似度"""
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)
            
            # 如果当前块为空，直接添加
            if not current_chunk_text:
                current_chunk_text.append(sentence)
                current_size += sentence_size
                continue
            
            # 计算当前句子与前一个句子的相似度
            # 如果相似度低，说明语义发生了变化
            prev_embedding = embeddings[i - 1]
            current_embedding = embeddings[i]
            similarity = cosine_similarity(prev_embedding, current_embedding)
            
            # 判断是否需要创建新块
            should_split = False
            
            # 条件1：相似度低于阈值（语义发生变化）
            if similarity < similarity_threshold:
                should_split = True
            
            # 条件2：累积字符数超过chunk_size（达到大小限制）
            if current_size + sentence_size > self.chunk_size:
                should_split = True
            
            if should_split:
                # 保存当前块
                chunk_text = ' '.join(current_chunk_text)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                
                # 开始新块
                current_chunk_text = [sentence]
                current_size = sentence_size
            else:
                # 继续添加到当前块
                current_chunk_text.append(sentence)
                current_size += sentence_size
        
        # 添加最后一个块
        if current_chunk_text:
            chunk_text = ' '.join(current_chunk_text)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        # 如果没有生成任何块，返回整个文本
        if not chunks:
            return [self.text]
        
        return chunks

    # 基于文档结构的分块
    def structure_split(self) -> List[str]:
        """
        基于文档结构的分块
        支持多种分隔符模式：
        1. line_separator: 按行切分，每行作为一个块
        2. paragraph_separator: 按段落切分（如双换行符）
        3. custom_separator: 自定义字符串分隔符
        4. title: 按标题标记切分（默认，向后兼容）
        
        :return: 分块后的字符串列表
        """
        if not self.structure_markers:
            # 如果没有配置，使用默认的Markdown标题标记
            return self._split_by_title()
        
        # 1. 优先检查是否配置了行分隔符
        if 'line_separator' in self.structure_markers:
            return self._split_by_line()
        
        # 2. 检查是否配置了段落分隔符
        if 'paragraph_separator' in self.structure_markers:
            separator = self.structure_markers['paragraph_separator']
            if isinstance(separator, str):
                return self._split_by_separator(separator)
            elif isinstance(separator, list) and separator:
                # 支持多个段落分隔符，按优先级使用第一个
                return self._split_by_separator(separator[0])
        
        # 3. 检查是否配置了自定义分隔符
        if 'custom_separator' in self.structure_markers:
            separator = self.structure_markers['custom_separator']
            if isinstance(separator, str):
                return self._split_by_separator(separator, keep_separator=True)
            elif isinstance(separator, list) and separator:
                # 支持多个自定义分隔符，按优先级使用第一个
                return self._split_by_separator(separator[0], keep_separator=True)
        
        # 4. 默认使用标题切分（向后兼容）
        return self._split_by_title()
    
    def _split_by_line(self) -> List[str]:
        """按行切分，每行作为一个块"""
        lines = self.text.splitlines()
        chunks = [line.strip() for line in lines if line.strip()]
        return chunks
    
    def _split_by_separator(self, separator: str, keep_separator: bool = False) -> List[str]:
        """
        按指定分隔符切分
        :param separator: 分隔符字符串，支持转义字符（如 \\n 表示换行符）
        :param keep_separator: 是否在每个块开头保留分隔符，默认为 False
        """
        if not separator:
            raise ValueError("分隔符不能为空")
        
        # 处理转义字符（如 \\n, \\t 等）
        # 如果包含反斜杠，尝试解码转义字符
        try:
            # 使用 codecs 模块的 unescape 或者直接使用字符串的 encode/decode
            if '\\' in separator:
                # 将字符串中的转义序列转换为实际字符
                # 例如: "\\n" -> "\n", "\\t" -> "\t"
                separator = separator.encode().decode('unicode_escape')
        except (UnicodeDecodeError, ValueError):
            # 如果解码失败，使用原始分隔符
            pass
        
        chunks = self.text.split(separator)
        
        if keep_separator:
            # 保留分隔符在每个块的开头
            result = []
            text_starts_with_separator = self.text.startswith(separator)
            
            for i, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if not chunk:  # 跳过空块
                    continue
                
                # 判断是否需要添加分隔符
                # 1. 第一个块：只有当文本以分隔符开头时才添加
                # 2. 后续块：都添加分隔符（因为split会在分隔符前后都产生块）
                if i == 0:
                    if text_starts_with_separator:
                        result.append(separator + chunk)
                    else:
                        result.append(chunk)
                else:
                    # 后续块都添加分隔符
                    result.append(separator + chunk)
            
            return result
        else:
            # 清理并过滤空块（原有逻辑）
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
            return chunks
    
    def _split_by_title(self) -> List[str]:
        """按标题标记切分（原有逻辑）"""
        # 检查配置，如果没有配置则使用默认的Markdown标题标记
        if not self.structure_markers or 'title' not in self.structure_markers:
            # 使用默认的Markdown标题标记
            title_markers = ["# ", "## ", "### ", "#### ", "##### ", "###### "]
        else:
            title_markers = self.structure_markers['title']
        
        if not title_markers:
            raise ValueError("title 标记列表不能为空，请配置 structure_markers['title']")
        
        # 创建正则表达式，匹配标题行
        # 允许标题前有空白字符，标题标记后可以有空白字符
        escaped_markers = [re.escape(marker) for marker in title_markers]
        title_pattern = r'^\s*(?:' + '|'.join(escaped_markers) + r')\s*.+$'

        # 将文本按行分割
        lines = self.text.splitlines()
        chunks = []
        current_chunk = []

        for line in lines:
            # 检查是否是标题行（直接匹配原始行，不strip，以保留格式）
            if re.match(title_pattern, line):
                # 如果当前块不为空，先保存之前的块
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                # 开始新的块，从标题开始
                current_chunk = [line]
            elif line.strip():  # 非空行，添加到当前块
                current_chunk.append(line)

        # 添加最后一个块
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        # 返回清理后的块
        return [chunk.strip() for chunk in chunks if chunk.strip()]


    # JSON行分块
    def json_line_split(self) -> List[str]:
        """
        JSON行分块：每行一个 JSON（典型 JSONL）的分块方式
        """
        return [line.strip() for line in self.text.strip().split('\n') if line.strip()]

if __name__ == '__main__':
    # 测试不同方式的文本分块
    with open(r"/data/output_内-巴士集团售后处理.md", 'r', encoding='utf-8') as f:
        text = f.read()

    splitter = TextSpliter(text, "api_key", ["}"], {"title": ["# ", "## ", "### ", "#### "]}, chunk_size=500, overlap=100)

    # chunks = splitter.fixed_size_split()
    # chunks = splitter.recursive_character_split()
    # chunks = splitter.semantic_split()
    # chunks = splitter.json_line_split()
    chunks = splitter.choose_chunk_method('structure')


    for i in range(len(chunks)):
        print(chunks[i])
        print("=" * 160)