# =============================================================================
# document.py — 文档解析 & 分块服务
# =============================================================================
#
# 【这个文件在干什么】
# 处理用户上传文件的两步流水线：
# 1. 解析（parse_file）：把不同格式的文件（PDF/Word/TXT）提取为纯文本
# 2. 分块（chunk_text）：把长文本切成长度可控的小块
#
# 这两个步骤是 RAG 系统质量的基础——解析出错 = 后续全是错的，
# 分块太糙 = LLM 看到残缺的上下文。
# =============================================================================

import os

# 允许上传的文件扩展名集合。
# 用 set 而不是 list 是因为"检查是否存在"操作在 set 上是 O(1)，list 上是 O(n)。
# 虽然只有 3 个元素差别不大，但这是习惯问题——语义上就是"一组不重复的值"。
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def parse_file(file_path: str) -> str:
    """
    把文件解析为纯文本字符串。
    
    【设计思路】
    这是一个"路由器"而不是"处理器"——它根据文件扩展名分发到不同的解析器。
    当前版本直接在函数里写 if-elif，简单但扩展性差。
    如果要支持更多格式（如 .md, .epub, .html），更好的设计是
    策略模式（每个格式一个 Parser 类，注册到一个字典里）。
    
    【每种解析器的底层原理】
    - .txt: 直接读文件，最可靠的方式
    - .pdf: PyPDF2 读取 PDF 内部的 text objects（文字对象）。
      注意很多 PDF 是扫描件，没有文字层——这种文件 extract_text() 返回空字符串。
      当前没有 OCR 降级方案，扫描件会静默失败。
      PyPDF2 不会做 layout 分析，多栏 PDF 的文字顺序可能错乱。
    - .docx: python-docx 读取 OOXML 内部的 paragraphs。
      当前只提取段落文本，表格内容会丢失（因为表格的单元格也是 paragraph 但需要特殊处理）。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    # os.path.splitext("report.pdf")   → ("report", ".pdf")
    # os.path.splitext("a.b.c.txt")    → ("a.b.c", ".txt")  # 只取最后一个点
    # .lower() 保证 ".PDF" 和 ".pdf" 被同等对待

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"不支持的文件格式: {ext}，仅支持 {ALLOWED_EXTENSIONS}"
        )

    if ext == ".txt":
        # 【编码处理】明确指定 utf-8，避免 Windows 系统默认用 GBK 编码出乱码
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        # reader.pages 是一个 Page 对象的列表
        # page.extract_text() 返回该页上所有文字（按 PDF 内部存储顺序，不保证阅读顺序）
        # or "" 处理返回 None 的情况（有些 PDF 页面 extract_text 返回 None）
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif ext == ".docx":
        from docx import Document
        doc = Document(file_path)
        # doc.paragraphs 是 Paragraph 对象的列表
        # p.text 提取段落的纯文本（丢失格式、样式信息，但保留了段落边界）
        # 【注意】表格中的文字不在这里！doc.tables 才是表格数据。
        return "\n".join(p.text for p in doc.paragraphs)

    return ""  # 理论上不会走到这里（已被上面的 ext check 拦截）


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    把一篇文章按固定长度切成多个小块。
    
    【当前实现的局限性（这是 RAG 质量的最大瓶颈）】
    这是最简单的"滑动窗口"切分：
    - chunk_size=500, overlap=50 意味着：
      块1: text[0:500]
      块2: text[450:950]   ← 前 50 个字符和块1 重复
      块3: text[900:1400]  ← 前 50 个字符和块2 重复
    
    问题：
    1. 按字符数切，不看句子边界 → 可能在句子中间截断
       "Python是一种解释" 和 "型编程语言" 被切成两块
    2. overlap 不能真正解决截断问题——它只保证重叠区的字符在两个块里都有，
       但不保证重叠区正好在句子边界
    3. 段落的语义边界完全被忽略 —— 一个段落讨论"列表操作"，
       下一个讨论"字典操作"，但它们在同一个 chunk 里
    
    【更好的方案（第二阶段要替换的内容）】
    - 递归字符分割：先按段落切(\n\n)，超长的再按句号切，再超长的按逗号切
    - 语义分块：用 embedding 的余弦相似度找话题转折点
    - 分层块：小块(256)用于检索精准匹配，大块(1024)用于喂给 LLM
    
    【边界情况】
    - 空字符串：返回 []（blocks 为空列表）
    - 短文本（< chunk_size）：返回 [text]（整个文本作为一个块）
    - 单字：start 每次跳 450 (chunk_size - overlap)，可能产生大量小块
    - 最后一个块肯定小于 chunk_size，会被 if end > len(text) 的检查截断
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # 下一个块的起点 = 当前起点 + (chunk_size - overlap)
        # 例：500 - 50 = 450 → 每次前进 450 字符
        start += (chunk_size - overlap)
        # 如果起点已经到文本末尾了，停止（防止最后一个块和倒数第二个块重复太多）
        if start >= len(text):
            break

    return chunks