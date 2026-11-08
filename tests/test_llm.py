import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.services import llm as llm_module
from app.services.llm import LLMService


@pytest.fixture
def svc():
    # 单测不真加载 fastembed 模型：模型 100MB+，CI/新环境无缓存即挂。
    # 只 mock 构造，chat 链路照常走 OpenAI SDK 的可 patch 结构。
    with patch.object(llm_module, "TextEmbedding", MagicMock()):
        yield LLMService(api_key="sk-test", base_url="https://test.api.com")


def test_embed_returns_vector(svc):
    emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    with patch.object(svc, "embed", return_value=emb.tolist()):
        result = svc.embed("测试文本")
        assert len(result) == 3
        assert abs(result[0] - 0.1) < 0.001
        assert abs(result[1] - 0.2) < 0.001
        assert abs(result[2] - 0.3) < 0.001


def test_chat_returns_content(svc):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="你好！"))]

    with patch.object(svc.chat_client.chat.completions, "create", return_value=mock_resp):
        result = svc.chat([{"role": "user", "content": "你好"}])
        assert result == "你好！"


def test_chat_with_system_message(svc):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="根据参考资料，答案是42。"))]

    with patch.object(svc.chat_client.chat.completions, "create", return_value=mock_resp) as create_mock:
        messages = [
            {"role": "system", "content": "你是知识库问答助手。"},
            {"role": "user", "content": "问题：生命的意义是什么？"},
        ]
        result = svc.chat(messages)
        assert result == "根据参考资料，答案是42。"

        called_messages = create_mock.call_args[1]["messages"]
        assert len(called_messages) == 2
        assert called_messages[0]["role"] == "system"
        assert called_messages[1]["role"] == "user"
