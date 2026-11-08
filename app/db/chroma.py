# =============================================================================
# chroma.py — 向量存储（基于 FAISS）
# =============================================================================
#
# 【注意】文件名叫 chroma.py 但用的是 FAISS，不是 ChromaDB。
# 这是历史遗留命名——项目初版可能打算用 Chroma，后来换了 FAISS 但没改文件名。
#
# 【这个文件在干什么】
# 实现一个自建的向量存储系统，包含：
# 1. FAISS 向量索引（用于快速搜索）
# 2. JSON 元数据存储（记录每个向量属于哪个文档、哪个 chunk）
# 3. 磁盘持久化（应用重启后数据不丢）
#
# 【核心数据结构】
# self.index  — FAISS 的 IndexFlatL2 对象，存所有向量
# self.records — Python list[dict]，存每个向量的元数据（文本、文档名、页码等）
#
# 这两个结构的元素是一一对应的：
#   self.index  的第 i 个向量
#   self.records 的第 i 个 dict  ← 描述这个向量的元信息
# =============================================================================

import os
import json
import numpy as np
import faiss


class VectorStore:
    """
    自建的向量存储系统。

    【为什么不用 Chroma/Milvus/Qdrant 而是自己写？】
    这是教学/原型阶段的选择：
    - FAISS 是 Meta 开源的高性能向量搜索库，核心用 C++ 实现，快
    - 但 FAISS 是库不是数据库：它只管向量搜索，不管 metadata 过滤、持久化、并发
    - 所以需要自己写一层"metadata 管理"（records 列表 + JSON 文件）
    - 生产环境用 Qdrant/Milvus 能省掉这些胶水代码

    【persist_dir 和 collection_name】
    persist_dir: 存放索引文件的目录
    collection_name: 索引文件的标识（当前只支持一个 collection）

    持久化产生两个文件：
    {persist_dir}/{collection_name}.index       ← FAISS 二进制索引文件
    {persist_dir}/{collection_name}.data.json  ← 元数据 JSON 文件

    重启应用时，如果这两个文件存在，会自动加载恢复。
    """

    def __init__(self, persist_dir: str, collection_name: str = "documents"):
        # 确保目录存在——如果 data/chroma 不存在就创建
        os.makedirs(persist_dir, exist_ok=True)
        self.index_path = os.path.join(persist_dir, f"{collection_name}.index")
        self.data_path = os.path.join(persist_dir, f"{collection_name}.data.json")

        # 初始化状态
        self.index: faiss.IndexFlatL2 | None = None  # FAISS 索引对象（初始为 None 表示空库）
        self.records: list[dict] = []  # 每条记录的元数据

        # 如果磁盘上已有索引文件，就加载恢复
        # 这意味着即使重启应用，之前上传的文档仍然在
        if os.path.exists(self.index_path) and os.path.exists(self.data_path):
            # faiss.read_index() 读取二进制索引文件
            self.index = faiss.read_index(self.index_path)
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.records = json.load(f)

    def add(self, ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]):
        """
        向向量库中添加一批文档块。

        【参数说明】
        虽然当前业务代码每次只添加 1 个块（循环调用 add），
        但这个方法的签名是批量的——支持一次添加多个块。

        【FAISS 的向量格式要求】
        FAISS 要求向量是 numpy float32 数组，形状为 (N, d)，
        其中 N = 向量数量，d = 每个向量的维度。

        numpy.array(embeddings) 把 Python list[list[float]] 转为 numpy ndarray，
        .astype("float32") 确保精度匹配（FAISS 内部用 float32 计算）。
        """
        vectors = np.array(embeddings).astype("float32")

        # 第一次添加时，FAISS 索引还不存在 → 创建它
        # IndexFlatL2 是"暴力搜索"索引（Flat = 不压缩，L2 = 欧几里得距离）
        # vectors.shape[1] = 向量的维度（bge-small 是 384）
        #
        # 【为什么用 IndexFlatL2 而不是 HNSW？】
        # - Flat L2 的搜索复杂度是 O(N)（每次都遍历全部向量）
        # - HNSW 可以达到 O(log N)（图结构的近似搜索）
        # - 当前选 FlatL2 是因为代码量小、适合原型。数据量大后要换 HNSW 或 IVF
        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])

        # 把向量加入 FAISS 索引
        self.index.add(vectors)

        # 同时把元数据加入 records 列表
        # records 和 index 的元素一一对应，通过位置（下标）关联
        # 例如 self.records[5] 就是 self.index 中第 5 个向量的元信息
        for i, chunk_id in enumerate(ids):
            self.records.append({
                "id": chunk_id,                              # 如 "abc123-chunk0"
                "text": texts[i],                            # chunk 的完整文本
                "embedding": embeddings[i],                  # 向量本身（用于重建索引）
                "document_id": metadatas[i].get("document_id", ""),
                "document_name": metadatas[i].get("document_name", ""),
                "chunk_index": metadatas[i].get("chunk_index", 0),
                "page": metadatas[i].get("page"),
            })

        # 每次添加都立即保存到磁盘（保证应用崩溃时数据不丢）
        self._save()

    def search(self, query_embedding: list[float], n_results: int = 3) -> dict:
        """
        搜索与 query 最相似的 n_results 个文档块。

        【L2 距离的含义】
        L2 距离 = sqrt(sum((a_i - b_i)^2))，越小表示越相似。
        FAISS 的 IndexFlatL2 返回的就是 L2 距离。

        【返回值格式模仿 Chroma API】
        为了和 Chroma 的调用方式兼容（历史原因），返回格式为：
        {
            "ids": [["id1", "id2", ...]],
            "documents": [["text1", "text2", ...]],
            "metadatas": [[{...}, {...}, ...]],
            "distances": [[0.1, 0.3, ...]]
        }
        外层 [] 代表按 batch，内层 [] 是每个 query 的结果。
        因为每次只传一个 query，所以取 [0]。
        """
        # 空库防御
        if self.index is None or self.index.ntotal == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        # 准备查询向量：形状为 (1, d) 的 float32 numpy 数组
        query = np.array([query_embedding]).astype("float32")
        # k 不能超过索引中的总向量数
        k = min(n_results, self.index.ntotal)

        # FAISS 的 search 方法：
        # 输入 query (1, d)，返回 (distances, indices) 两个 (1, k) 数组
        # distances: L2 距离，越近越好
        # indices: 在 self.index 中的位置（也是 self.records 的下标）
        distances, indices = self.index.search(query, k)

        ids, docs, metas = [], [], []
        for idx in indices[0]:
            # 防御性检查：idx 不应该越界，但加了更安全
            if 0 <= idx < len(self.records):
                r = self.records[idx]  # 通过下标取元数据
                ids.append(r["id"])
                docs.append(r["text"])
                metas.append({
                    "document_id": r["document_id"],
                    "document_name": r["document_name"],
                    "chunk_index": r["chunk_index"],
                    "page": r["page"],
                })

        return {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [distances[0].tolist()]
        }

    def delete_by_document_id(self, document_id: str):
        """
        删除属于某个文档的所有向量块。

        【这里的性能问题（项目最脆弱的点之一）】
        当前实现：
        1. 遍历所有 records，过滤掉目标 document_id
        2. 如果 records 确实减少了 → 重建整个 FAISS 索引
        3. 保存到磁盘

        "_rebuild_index()" 的开销是 O(N*d)，其中 N = 所有向量数，d = 向量维度。
        如果库里有 50000 个向量且每个 384 维：
        - 删除 1 条记录 → 重建 50000*384 的索引
        这显然浪费。更好的做法应该是用"逻辑删除"（标记 deleted=True），
        隔一段时间或积累到一定量再批量物理删除和重建。

        或者用支持原生删除的向量存储方案（如 Qdrant）。
        """
        old_count = len(self.records)
        self.records = [r for r in self.records if r["document_id"] != document_id]

        if len(self.records) == old_count:
            return  # 没删掉任何记录（目标 document_id 不存在）

        # 有记录被删了 → 必须重建索引
        # FAISS 的 IndexFlatL2 不支持删除单条向量的操作
        self._rebuild_index()
        self._save()

    def _rebuild_index(self):
        """
        从 records 重新构建 FAISS 索引。

        【工作原理】
        1. 清空索引
        2. 把 records 中所有 embedding 拼成一个 numpy 矩阵
        3. 创建新的 IndexFlatL2 并 add 所有向量

        【时间复杂度】
        O(N*d)，N = 向量数量，d = 维度。
        对于生产环境（百万级向量），这个操作可能需要数秒到数十秒。
        """
        if not self.records:
            self.index = None
            return

        dim = len(self.records[0]["embedding"])
        self.index = faiss.IndexFlatL2(dim)
        # 从 records 中提取所有 embedding，拼成 (N, d) 的矩阵
        vectors = np.array([r["embedding"] for r in self.records]).astype("float32")
        self.index.add(vectors)

    def _save(self):
        """
        把当前状态持久化到磁盘。

        两个文件：
        - .index: FAISS 的 write_index 生成二进制文件
        - .data.json: Python 的 json.dump 生成 JSON 文件

        【潜在的数据不一致风险】
        如果"写完 .index 成功"但"写 .data.json 失败"（比如磁盘满了），
        两个文件就不一致了。
        更健壮的做法是原子写入（先写临时文件，再 rename），
        但目前这样在原型阶段可以接受。
        """
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    def close(self):
        """显式保存（应用关闭时调用）"""
        self._save()
