# =============================================================================
# llm.py — LLM 服务（聊天 + Embedding）
# =============================================================================
#
# 【这个文件在干什么】
# 封装了两个核心能力：
# 1. 文本→向量（embed）：把一段文字变成 384 维的浮点数数组
# 2. 向量→回复（chat）：把对话历史发给 LLM，拿到文字回复
#
# 【为什么把 chat 和 embed 放在同一个类里？】
# 当前项目为了简单，把它们耦合在了一起。
# 但这两个能力应该分开——因为：
# - chat 用 DeepSeek 的云端 API（需要网络、按 token 收费）
# - embed 用本地的 BGE 模型（不需要网络、本地 CPU 算）
# 分开后你可以单独换掉其中一个而不影响另一个。
# 这是第二阶段重构要改的。
# =============================================================================

from openai import OpenAI
from fastembed import TextEmbedding


class LLMService:
    """
    LLM 服务的统一入口。

    【初始化时做了什么】
    - chat_client: 一个 OpenAI 兼容的 HTTP 客户端，指向 DeepSeek 的服务器
    - embed_model: 一个本地模型对象，加载了 bge-small-zh-v1.5

    【OpenAI SDK 为什么能连 DeepSeek？】
    因为 DeepSeek 的 API 设计与 OpenAI 完全兼容——同样的 /v1/chat/completions 路径，
    同样的 messages 格式。这是目前行业的默认做法：大家都兼容 OpenAI 的接口标准。
    所以你换到其他兼容供应商（硅基流动、OpenRouter、本地 Ollama），
    只需要改 base_url，业务代码一行不动。
    """

    def __init__(self, api_key: str, base_url: str):
        # OpenAI() 构造函数会记住 api_key 和 base_url，
        # 但不会立刻连接服务器——连接是懒加载的，第一次调用 .chat.completions.create() 时才建连接。
        # base_url 后面 SDK 会自动拼接 /v1 和 /chat/completions 等路径。
        self.chat_client = OpenAI(api_key=api_key, base_url=base_url)

        # TextEmbedding() 会：
        # 1. 从 HuggingFace 下载 BAAI/bge-small-zh-v1.5 模型（如果本地没有）
        #    ^ 这个模型约 100MB，首次运行需要下载，取决于网速
        # 2. 把模型加载到内存中（CPU）
        # 3. 模型是 CPU 跑的，不依赖 GPU
        #
        # 【环境变量 HF_ENDPOINT】
        # 国内网络访问 HuggingFace 可能很慢，设置 HF_ENDPOINT=https://hf-mirror.com
        # 可以用国内镜像加速下载。这个变量在 docker-compose.yml 里设了。
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")

    def embed(self, text: str) -> list[float]:
        from app.cache import cache_get_embedding, cache_set_embedding

        cached = cache_get_embedding(text)
        if cached is not None:
            return cached

        result = list(self.embed_model.embed([text]))[0].tolist()
        cache_set_embedding(text, result)
        return result

    def chat(self, messages: list[dict], model: str = "deepseek-chat", temperature: float = 0.7) -> str:
        """
        调用 LLM 聊天接口。

        【参数说明】
        - messages: [{"role": "user", "content": "..."}, ...]
          role 可以是 "system"（系统指令）、"user"（用户）、"assistant"（AI）
        - model: 模型名。deepseek-chat = DeepSeek-V3
        - temperature: 0~1 之间，越高越"有创意"但也越随机。
          RAG 场景建议 0.3~0.7——既要基于事实（不能太随机），
          又不能太死板（完全 temperature=0 会让回答读起来生硬）。

        【返回什么】
        只返回 LLM 回复的纯文本 content。
        没有返回 token 使用量、finish_reason 等信息——在生产环境中这些很重要，
        但在原型阶段够用了。

        【当前实现的问题】
        1. 同步调用：self.chat_client.chat.completions.create() 会阻塞线程
           直到 LLM 返回。如果有用户在等待，其他用户也被堵住了。
           → 异步化是第二阶段的改造点
        2. 没有重试：如果 DeepSeek 服务器返回 429（限流）或 500（超时），
           这个调用会直接抛异常，调用方需要自己处理
           → 应该加指数退避重试逻辑
        3. 没有流式：LLM 在服务器端全部生成完才返回，用户要等 3-5 秒
           → 流式输出可以显著改善体验
        """
        resp = self.chat_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        # resp.choices 是一个列表（通常只有一个元素，除非设置 n>1）
        # choices[0].message.content 是 LLM 生成的文字
        return resp.choices[0].message.content

    # 注意：没有 close() 方法。
    # fastembed 模型常驻内存，OpenAI HTTP 客户端会自动管理连接池。
    # 应用关闭时会由 Python 垃圾回收处理，不需要手动清理。
