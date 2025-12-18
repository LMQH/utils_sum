import logging

logger = logging.getLogger("Web_API")

class ParentChildSpliter:
    def __init__(self, file_name=None, content=None, parent_size=500, child_size=100, child_overlap_size=15):
        if not file_name or not file_name.endswith(".md"):
            error_msg = f"文件路径错误，必须为Markdown文件。当前文件名: {file_name}。提示：Word文档(.docx/.doc)需要先转换为Markdown格式。"
            logger.error(error_msg)
            raise ValueError(error_msg)
        # 确保child_overlap_size小于child_size，防止死循环
        if child_overlap_size >= child_size:
            logger.warning(f"子块重叠大小({child_overlap_size})大于等于子块大小({child_size})，自动调整为子块大小的四分之一")
            child_overlap_size = child_size / 4
        if content is None:
            error_msg = "文件内容不能为空"
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.file_name = file_name
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap_size = int(child_overlap_size)
        self.md_content = [line for line in content.splitlines() if line not in ["", "\n"]]
        self.title_stack = []
        self.parent_chunks = []
        self.structured_parent_chunks = []
        self.structured_child_chunks = []

    def process(self):
        """处理文件分块，返回结构化的父子块列表"""
        self.split_parent_chunk()    # 父分段
        self.save_parent_chunk()     # 保存父分段
        self.split_save_child_chunk()    # 划分并保存子分段
        return self.get_structured_chunks()

    def get_parent_chunks(self):
        """获取父块"""
        return self.structured_parent_chunks

    def get_structured_chunks(self):
        """合并父子块"""
        structured_chunks = self.structured_parent_chunks + self.structured_child_chunks
        return structured_chunks

    def get_chunks_list(self):
        """获取父子分块列表"""
        chunks_list = [v['content'] for v in self.get_structured_chunks()]
        return chunks_list

    def get_parents_title(self):
        """获取当前标题的所有父标题"""
        if self.title_stack:
            return "\n".join(self.title_stack)
        else:
            return None

    def upgrade_title_stack(self, new_title, level):
        """更新标题栈，将层级低或等于新标题的标题弹出"""
        for title in self.title_stack[::-1]:   #从后往前遍历
            if title.count("#") >= level:
                self.title_stack.remove(title)
        self.title_stack.append(new_title)
        return self.title_stack

    def save_chunk(self, current_chunk):
        """将当前块保存到分块列表中"""
        self.parent_chunks.append("\n".join(current_chunk)) # 保存当前块
        # 如果 get_parents_title() 返回 None，返回空列表而不是包含 None 的列表
        parent_title = self.get_parents_title()
        new_chunk = [parent_title] if parent_title is not None else []
        return new_chunk, 0

    def add_chunk(self, line, chunk, size):
        """将当前内容保存到块中"""
        chunk.append(line)  # 加入内容
        size += len(line)
        return size

    def split_parent_chunk(self):
        """ 
        划分父块
        
        该方法将Markdown内容按照特定规则划分为父块，处理以下几种内容类型：
        1. 标题行 - 以#开头的行
        2. 图片 - 以![图片]开头的行，可能包含表格和描述文本
        3. 表格 - 以|开头的行
        4. 普通文本 - 其他内容
        
        划分规则：
        - 图片内容会被分组处理，包括图片标记、内部表格和描述文本
        - 表格会与后续的总结语句一起作为一个块
        - 普通文本会根据parent_size阈值进行分割
        - 每个块会自动添加父标题作为上下文信息
        """
        current_chunk = []  # 当前正在构建的父块
        current_figure_chunk = []  # 当前包含图片的临时块
        current_table = []  # 当前表格的临时块
        is_table = False    # 标记是否正在处理表格
        add_summary = False  # 标记是否需要额外添加总结语句（针对视觉模型处理错误的情况，e.g.:[内容]\n该表格......）
        current_size = 0    # 当前父块的大小（字符数）
        current_figure_chunk_size = 0  # 当前图片块的大小（字符数）
        level = 0   # 当前标题的层级（#的数量）

        # 逐行处理Markdown内容
        for line in self.md_content:
            # ========== 1. 处理标题行 ==========
            if line.startswith("#"):    
                level = line.count("#")
                self.upgrade_title_stack(line, level) # 更新标题栈
                current_size = self.add_chunk(line, current_chunk, current_size)
                if current_size >= self.parent_size:
                    current_chunk, current_size = self.save_chunk(current_chunk)
                continue
            else: 
                # ========== 2. 处理非标题行 ==========
                # 如果当前块为空，添加父标题作为上下文
                if current_chunk == [] and current_figure_chunk == [] and self.get_parents_title() is not None:   
                    current_size = self.add_chunk(self.get_parents_title(), current_chunk, current_size)
                
                # ========== 2.1 处理图片内容 ==========
                if line.startswith("![图片]"):
                    # 开始新的图片块
                    current_figure_chunk = []
                    current_figure_chunk_size = 0
                    current_figure_chunk_size = self.add_chunk(line, current_figure_chunk, current_figure_chunk_size)
                    continue
                
                if add_summary:
                    add_summary = False
                    current_figure_chunk_size = self.add_chunk("\n"+line, current_figure_chunk, current_figure_chunk_size)
                    # 将图片块合并到主块中
                    current_chunk.extend(current_figure_chunk)
                    current_size += current_figure_chunk_size
                    current_figure_chunk_size = 0
                    current_figure_chunk = []
                    # 检查是否需要保存当前块
                    if current_size >= self.parent_size:    
                        current_chunk, current_size = self.save_chunk(current_chunk)
                        current_figure_chunk_size = 0
                    continue    

                # 如果当前正在处理图片块
                if current_figure_chunk != []:   
                    # 图片块内的特殊处理逻辑
                    if line == "[":  # 图片描述开始标记，跳过
                        continue
                    if line.startswith("|"):  # 图片内包含表格
                        is_table = True
                    if is_table and not line.startswith("|"):  # 表格结束
                        if line == "]":
                            is_table = False
                            add_summary = True
                            continue
                        is_table = False
                        current_figure_chunk_size = self.add_chunk("\n"+line, current_figure_chunk, current_figure_chunk_size)
                        continue
                    if line == "]":    # 图片块结束标记
                        # 将图片块合并到主块中
                        current_chunk.extend(current_figure_chunk)
                        current_size += current_figure_chunk_size
                        current_figure_chunk_size = 0
                        current_figure_chunk = []
                        # 检查是否需要保存当前块
                        if current_size >= self.parent_size:    
                            current_chunk, current_size = self.save_chunk(current_chunk)
                            current_figure_chunk_size = 0
                        continue    
                    # 继续添加图片块内容
                    current_figure_chunk_size = self.add_chunk(line, current_figure_chunk, current_figure_chunk_size)
                    continue

                # ========== 2.2 处理表格内容 ==========
                if line.startswith("|"):    
                    # 开始新的表格块
                    current_size = self.add_chunk(line, current_table, current_size)
                    continue    
                
                # 如果当前正在处理表格块
                if current_table != []:   
                    # 保存当前块，然后处理表格
                    current_chunk, current_size = self.save_chunk(current_chunk)
                    # 将表格添加到新块
                    current_chunk.extend(current_table)
                    current_table = []
                    # 添加表格后的总结语句
                    current_size = self.add_chunk("\n"+line, current_chunk, current_size)
                    # 保存包含表格的块
                    current_chunk, current_size = self.save_chunk(current_chunk)
                    continue
                
                # ========== 2.3 处理普通文本 ==========
                # 检查添加新内容是否会超过大小阈值
                if current_size + len(line) >= self.parent_size:  
                    # 先保存当前块，再开始新块
                    current_chunk, current_size = self.save_chunk(current_chunk)
                    current_size = self.add_chunk(line, current_chunk, current_size)
                    continue
                else:   
                    # 直接添加到当前块
                    current_size = self.add_chunk(line, current_chunk, current_size)
                    continue
        
        # ========== 3. 处理最后的剩余块 ==========
        if current_chunk:
            self.parent_chunks.append("\n".join(current_chunk))
        return self.parent_chunks

    def save_parent_chunk(self):
        """将当前父块进行结构化保存"""
        file_name = self.file_name
        for i, chunk in enumerate(self.parent_chunks):
            self.structured_parent_chunks.append({
                "file_name": file_name,
                "pc_type": "parent",
                "index": i,
                "content": chunk
            })
        logger.info("父块结构化保存")
        logger.debug(f"父块结构化保存完成:{self.structured_parent_chunks}")
        self.parent_chunks = []
        return self.structured_parent_chunks

    def split_save_child_chunk(self):
        """ 划分子块并结构化保存 """
        logger.info("子块划分中...")
        for chunk in self.structured_parent_chunks:
            # 父块大小 < chunk_size，直接保存
            if len(chunk['content']) < self.child_size:  
                self.structured_child_chunks.append({
                "file_name": chunk['file_name'],
                "pc_type": "child",
                "index": chunk['index'],
                "content": chunk['content']
            }) 
            else:
                start = 0
                end = start + self.child_size
                # 添加安全计数器，防止意外死循环
                safety_counter = 0
                max_iterations = len(chunk['content']) // max(1, self.child_size - self.child_overlap_size) + 10
                
                while end < len(chunk['content']):
                    # 安全检查
                    safety_counter += 1
                    if safety_counter > max_iterations:
                        logger.warning(f"子块划分可能陷入死循环，已强制退出。文件: {chunk['file_name']}, 索引: {chunk['index']}")
                        break
                        
                    self.structured_child_chunks.append({
                        "file_name": chunk['file_name'],
                        "pc_type": "child",
                        "index": chunk['index'],
                        "content": chunk['content'][start:end]
                    })
                    start = end - self.child_overlap_size
                    # 确保start至少前进1个位置，防止死循环
                    if start <= end - self.child_overlap_size:
                        start = end - self.child_overlap_size + 1
                    end = start + self.child_size
        logger.info("子块结构化保存")
        return self.structured_child_chunks