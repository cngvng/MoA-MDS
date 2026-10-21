from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.llms.factory import build_llm
from src.llms.llm import OpenAICompatibleLLM


def test_factory_requires_hosted_credentials():
    with pytest.raises(ValueError, match="Missing API key"):
        build_llm({"llm": {"model_id": "synthetic"}})


def test_factory_accepts_local_endpoint_without_key():
    llm = build_llm({"llm": {"model_id": "synthetic", "base_url": "http://localhost:8000/v1"}})
    assert str(llm.client.base_url) == "http://localhost:8000/v1/"
    assert llm.model_id == "synthetic"
    llm.client.close()


def test_unsupported_provider_is_rejected():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        build_llm({"llm": {"provider": "unknown", "model_id": "synthetic"}})


@pytest.mark.parametrize("has_usage", [True, False])
def test_backend_payload_response_and_usage(monkeypatch, has_usage):
    llm = OpenAICompatibleLLM("synthetic", api_key="offline-test-only",
                             default_temperature=None, max_tokens_param="max_completion_tokens")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Tóm tắt thử nghiệm"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15) if has_usage else None,
    )
    create = Mock(return_value=response)
    monkeypatch.setattr(llm.client.chat.completions, "create", create)
    result = llm.generate_answer("System", "Synthetic input", max_tokens=50)
    payload = create.call_args.kwargs
    assert payload["messages"] == [{"role": "system", "content": "System"},
                                   {"role": "user", "content": "Synthetic input"}]
    assert payload["max_completion_tokens"] == 50
    assert "temperature" not in payload
    assert "max_tokens" not in payload
    assert result.text == "Tóm tắt thử nghiệm"
    assert result.usage.total_tokens == (15 if has_usage else None)
    llm.client.close()
