<div align="center">

# rag-qa

**金融文档 RAG 知识库问答 —— 带来源引用和置信度**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-72%20passing-brightgreen?style=flat-square)](https://github.com/MeiXi723/rag-qa/actions)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

上传 PDF / Word / TXT → 提问 → 带引用的回答，并告诉你这个答案可不可信

</div>

---

## 这个项目在关心什么

RAG demo 满地都是，但真拿去问金融文档，会遇到三个问题：

| 问题 | 这里的解法 |
| :--- | :--- |
| 只做向量检索，专有名词查不到 | **BM25 + 向量双路召回**，RRF 融合只看排名不看分值，双路命中天然置顶 |
| 检索结果里全是碎块和重复 | **质量过滤**：NFKC 归一化哈希去重 + 最小长度丢碎块 |
| 模型答得像模像样但引用了不存在的来源 | **答案置信度评估**：`high` / `medium` / `low`，检测幻觉引用和"证据不足"话术 |

<sub>检索层不是抄教程，是从 AFAC2026 参赛管道移植过来的实测方案。</sub>

---

## 检索架构

```
            ┌────────────┐   ┌────────────┐
  query ──► │ 向量腿 FAISS │   │ BM25 腿     │   jieba 分词
            │ (语义相似)   │   │ (字面命中)  │
            └─────┬──────┘   └──────┬─────┘
                  └────► RRF 融合 ◄──┘        只看排名不看分值，
                          │                   双路命中天然置顶
                  质量过滤（归一化去重 + 碎块丢弃）
                          │
                  CrossEncoder 精排 top-25   BAAI/bge-reranker-base，
                          │                  加载失败自动降级 RRF
                       TOP_K ──► LLM 生成 ──► 答案置信度评估
                                              （引用有效性 / 证据不足话术检测）
```

| 阶段 | 文件 | 说明 |
| :--- | :--- | :--- |
| 双路召回 | `app/services/retrieval/bm25.py` + `app/db/chroma.py` | BM25 语料与向量库记录同步，增删文档自动标脏惰性重建 |
| RRF 融合 | `app/services/retrieval/fusion.py` | Reciprocal Rank Fusion，移植自 afac2026 竞赛实测方案 |
| 质量过滤 | `app/services/retrieval/quality.py` | NFKC 归一化哈希去重 + 最小长度过滤 |
| 精排 | `app/services/retrieval/rerank.py` | 可选增强：未安装依赖时静默降级，**检索永不因精排挂掉** |
| 置信度 | `app/services/retrieval/validate.py` | 单模型自检：引用编号有效性、证据不足话术、幻觉引用检测 |
| 分块 | `app/services/document.py` | 句界感知分块：按句末标点打包，句子不再被拦腰截断 |

所有开关集中在 `app/config.py`（`ENABLE_HYBRID` / `ENABLE_RERANK` / `RRF_K` /
`RETRIEVAL_CANDIDATES` 等），`ENABLE_HYBRID=false` 一键回退 v1 纯向量路径。

---

## 技术栈

| 层 | 选型 |
| :--- | :--- |
| API | FastAPI 0.115 + Uvicorn，JWT (python-jose) 认证 |
| 向量检索 | FAISS IndexFlatL2 + fastembed（BGE bge-small-zh-v1.5，384 维本地 CPU） |
| 稀疏检索 | rank-bm25 + jieba |
| 精排（可选） | sentence-transformers CrossEncoder（bge-reranker-base） |
| 生成 | DeepSeek（OpenAI 兼容），Redis 响应缓存 + 多轮历史裁剪 |
| 存储 | SQLite (SQLAlchemy) + FAISS JSON 元数据 + Redis 7 |
| 部署 | Docker Compose 双服务健康检查，GitHub Actions CI |

---

## 快速开始

```bash
# 方式一：本地 venv
make install          # 建 .venv + 装依赖（阿里云镜像，含精排增强）
cp .env.example .env
make run              # http://localhost:8000/docs

# 方式二：Docker（精排增强默认开启）
docker compose up -d --build

# 跳过精排省 ~1GB 镜像体积：
# docker compose build --build-arg INSTALL_RERANK=false
```

最小 `.env`：

```bash
DEEPSEEK_API_KEY=sk-xxx
HF_ENDPOINT=https://hf-mirror.com
```

---

## API

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| POST | `/api/auth/register` / `/api/auth/login` | 注册 / 登录，返回 JWT |
| POST | `/api/documents/upload` | 上传文档（multipart，解析 + 分块 + 双索引入库） |
| GET/DELETE | `/api/documents` / `/{doc_id}` | 文档列表 / 删除 |
| POST | `/api/qa/ask` | 提问，返回 `answer` + `sources` + `answer_confidence` |
| GET | `/api/conversations[/{id}]` | 会话列表 / 详情（含消息历史） |
| GET | `/health` | 存活探针（无鉴权） |

`answer_confidence` 三档：

| 值 | 含义 |
| :--- | :--- |
| `high` | 引用了有效来源 |
| `medium` | 有答案但零引用 |
| `low` | 证据不足或幻觉引用 |

---

## 测试

```bash
make test    # 72 个用例：单测全 mock 化（不下载模型、不连真实服务）
make lint    # ruff
make cov     # 覆盖率报告
```

- 检索层单测表驱动：RRF 分数公式、BM25 分词口径、精排降级路径全部锁定
- 集成测试带 JWT 全链路（注册 → 上传 → 提问 → 溯源 → 清理），数据库 / 索引全部临时目录隔离
- CI：lint + pytest + 覆盖率，精排依赖不进 CI

---

## 已知限制

- **FAISS 用的是 Flat 索引，O(N) 检索。** 数据量上来必须换 HNSW / IVF，当前没做
- **索引非原子写盘**：`add` 即写，断电可能截断 JSON
- 分块按句界，不是语义边界（语义分块在 Roadmap）
- 精排是可选依赖，不装就降级为 RRF 排序，效果有差距
- 只支持单轮检索增强，没有做 query 改写 / multi-hop

---

## Roadmap

- [ ] FAISS 换 HNSW / IVF（当前 Flat O(N)，数据量上来后必须换）
- [ ] 索引原子写盘
- [ ] 语义分块（embedding 相似度找话题边界）+ 分层块（小块检索 + 大块生成）
- [ ] 生成侧流式输出（SSE）
- [ ] 双模型交叉验证（afac Validator 完整版，需第二供应商 key）

---

## 许可

[MIT](LICENSE)
