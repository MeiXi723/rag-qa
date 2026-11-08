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
    句界感知分块（v2.0，替代 v1 的纯字符滑窗）。

    【算法】
    1. 先按句末标点（。！？!?；;和换行）把文本切成句子单元
    2. 贪心打包：单元依次装入当前块，装不下就开新块
    3. 新块开头带上一块末尾 overlap 个字符（与 v1 overlap 语义对齐）
    4. 单句超过 chunk_size（无标点长段/表格流）→ 该单元退回 v1 固定窗口滑切

    好处：绝大多数块不再把句子拦腰截断，"Python是一种解释" / "型编程语言"
    这种碎片不会再喂给 embedding 和 LLM。

    【边界情况】
    - 短文本（≤ chunk_size）：返回 [text]，与 v1 一致
    - overlap ≥ chunk_size：非法参数，防御性收窄为 chunk_size//2（否则窗口不前进）
    """
    if len(text) <= chunk_size:
        return [text]
    if overlap >= chunk_size or overlap < 0:
        overlap = chunk_size // 2

    chunks: list[str] = []
    for unit in _split_units(text):
        if len(unit) > chunk_size:
            chunks.extend(_hard_split(unit, chunk_size, overlap))
            continue
        if chunks and len(chunks[-1]) + len(unit) <= chunk_size:
            chunks[-1] += unit
            continue
        tail = chunks[-1][-overlap:] if chunks else ""
        candidate = tail + unit
        chunks.append(candidate if len(candidate) <= chunk_size else unit)

    return chunks


def _split_units(text: str) -> list[str]:
    """按句末标点切单元，标点保留在单元尾部（切了才知道哪里是句子边界）。"""
    boundaries = "。！？!?；;\n"
    units: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in boundaries:
            units.append("".join(buf))
            buf = []
    if buf:
        units.append("".join(buf))
    return [u for u in units if u]


def _hard_split(unit: str, chunk_size: int, overlap: int) -> list[str]:
    """v1 的固定窗口滑切，仅用于超长无标点单元的兜底。"""
    out = []
    start = 0
    while start < len(unit):
        out.append(unit[start:start + chunk_size])
        start += chunk_size - overlap
        if start >= len(unit):
            break
    return out
