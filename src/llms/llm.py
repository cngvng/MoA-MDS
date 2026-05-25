from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from openai import OpenAI


@dataclass
class LLMUsage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Optional[int]]:
        return asdict(self)


@dataclass
class LLMResult:
    text: str
    model_id: str
    usage: LLMUsage
    saved_path: Optional[str] = None
    raw_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OpenAICompatibleLLM:
    """
    A professional OpenAI-compatible LLM client wrapper.

    Works with:
    - OpenAI official API
    - vLLM OpenAI-compatible server
    - Ollama OpenAI-compatible API
    - LM Studio / llama.cpp / other OpenAI-compatible providers
    - Any self-hosted LLM server exposing /v1/chat/completions

    Important:
    This class does NOT run the model itself.
    It calls an existing OpenAI-compatible inference endpoint.
    """

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_temperature: Optional[float] = 0.2,
        default_top_p: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
        max_tokens_param: str = "max_tokens",
        timeout: Optional[float] = 120.0,
        max_retries: Optional[int] = None,
        output_dir: str | Path = "llm_outputs",
        system_role: Literal["system", "developer"] = "system",
        default_extra_body: Optional[Dict[str, Any]] = None,
        default_request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            model_id:
                Model name used by the target provider/server.
                Examples:
                - "gpt-4.1"
                - "gpt-4o-mini"
                - "Qwen/Qwen2.5-7B-Instruct"
                - "llama3.1:8b"
                - "meta-llama/Llama-3.1-8B-Instruct"

            api_key:
                Provider API key.
                For local servers that ignore auth, use "EMPTY", "ollama", or any dummy string.

            base_url:
                OpenAI-compatible endpoint.
                Examples:
                - OpenAI official: None
                - vLLM: "http://localhost:8000/v1"
                - Ollama: "http://localhost:11434/v1"

            system_role:
                Use "system" for widest OpenAI-compatible support.
                Use "developer" only when you specifically target newer OpenAI-style prompting.
        """

        self.model_id = model_id
        self.base_url = base_url
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.default_max_tokens = default_max_tokens
        self.max_tokens_param = max_tokens_param
        self.timeout = timeout
        self.output_dir = Path(output_dir)
        self.system_role = system_role
        self.default_extra_body = default_extra_body or {}
        self.default_request_kwargs = default_request_kwargs or {}

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")

        # Many local OpenAI-compatible servers require an api_key field syntactically,
        # even if they ignore it.
        if not resolved_api_key and base_url:
            resolved_api_key = "EMPTY"

        if not resolved_api_key:
            raise ValueError(
                "Missing api_key. Provide api_key directly or set OPENAI_API_KEY."
            )

        client_kwargs: Dict[str, Any] = {
            "api_key": resolved_api_key,
            "timeout": timeout,
        }

        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries

        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.last_result: Optional[LLMResult] = None

    def build_messages(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({
                "role": self.system_role,
                "content": system_prompt,
            })

        if history:
            messages.extend(history)

        messages.append({
            "role": "user",
            "content": user_prompt,
        })

        return messages

    def generate_answer(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        save_text: bool = False,
        output_filename: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **request_kwargs: Any,
    ) -> LLMResult:
        """
        Generate answer as plain text.

        Returns:
            LLMResult containing:
            - text
            - model_id
            - token usage if returned by provider
            - saved_path if save_text=True
        """

        messages = self.build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history,
        )

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
        }

        final_temperature = (
            self.default_temperature if temperature is None else temperature
        )
        if final_temperature is not None:
            payload["temperature"] = final_temperature

        final_top_p = self.default_top_p if top_p is None else top_p
        final_max_tokens = (
            self.default_max_tokens if max_tokens is None else max_tokens
        )

        if final_top_p is not None:
            payload["top_p"] = final_top_p

        if final_max_tokens is not None:
            payload[self.max_tokens_param] = final_max_tokens

        if stop is not None:
            payload["stop"] = stop

        merged_extra_body = {
            **self.default_extra_body,
            **(extra_body or {}),
        }

        if merged_extra_body:
            payload["extra_body"] = merged_extra_body

        payload.update(self.default_request_kwargs)
        payload.update(request_kwargs)

        response = self.client.chat.completions.create(**payload)

        text = self._extract_text(response)
        usage = self._extract_usage(response)

        saved_path = None
        if save_text:
            saved_path = self.save_text(
                text=text,
                filename=output_filename,
            )

        result = LLMResult(
            text=text,
            model_id=self.model_id,
            usage=usage,
            saved_path=saved_path,
            raw_metadata={
                "id": getattr(response, "id", None),
                "created": getattr(response, "created", None),
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "finish_reason": self._extract_finish_reason(response),
                "base_url": self.base_url,
                "request_params": {
                    "temperature": final_temperature,
                    "top_p": final_top_p,
                    self.max_tokens_param: final_max_tokens,
                },
            },
        )

        self.last_result = result
        return result

    def generate_text(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        **kwargs: Any,
    ) -> str:
        """
        Convenience function for agent cores that only need plain text.
        """
        result = self.generate_answer(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs,
        )
        return result.text

    def save_text(
        self,
        text: str,
        filename: Optional[str] = None,
    ) -> str:
        """
        Save generated answer to a .txt file synchronously.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model_name = self.model_id.replace("/", "_").replace(":", "_")
            filename = f"{safe_model_name}_{timestamp}.txt"

        if not filename.endswith(".txt"):
            filename += ".txt"

        path = self.output_dir / filename
        path.write_text(text, encoding="utf-8")

        return str(path)

    def get_token_usage(self) -> LLMUsage:
        """
        Return token usage from the last generation.

        Note:
        Some OpenAI-compatible local servers do not return token usage.
        In that case, values will be None.
        """
        if self.last_result is None:
            return LLMUsage()

        return self.last_result.usage

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except Exception as exc:
            raise RuntimeError("Cannot extract text from LLM response.") from exc

        if content is None:
            return ""

        if isinstance(content, str):
            return content

        # Some multimodal/chat APIs may return structured content.
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif "text" in item:
                        parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)

        return str(content)

    @staticmethod
    def _extract_usage(response: Any) -> LLMUsage:
        usage = getattr(response, "usage", None)

        if usage is None:
            return LLMUsage()

        return LLMUsage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )

    @staticmethod
    def _extract_finish_reason(response: Any) -> Optional[str]:
        try:
            return response.choices[0].finish_reason
        except Exception:
            return None
