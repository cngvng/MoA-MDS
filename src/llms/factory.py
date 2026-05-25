from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from dotenv import find_dotenv, load_dotenv

from src.llms.llm import OpenAICompatibleLLM
from src.utils.config import Config, load_config


@dataclass(frozen=True)
class LLMFactoryConfig:
    provider: str
    model_id: str
    base_url: Optional[str]
    api_key_env: str
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]
    max_tokens_param: str
    reasoning_effort: Optional[str]
    timeout: Optional[float]
    max_retries: Optional[int]
    backoff_seconds: Optional[float]
    allow_missing_usage: bool
    output_dir: str
    dotenv_path: Optional[str]

    def safe_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["api_key_env"] = self.api_key_env
        payload["api_key_loaded"] = bool(os.getenv(self.api_key_env))
        return payload


def build_llm(
    config: Optional[Config | Mapping[str, Any]] = None,
    *,
    dotenv_path: str | Path = ".env",
    model_id: Optional[str] = None,
) -> OpenAICompatibleLLM:
    factory_config = load_llm_factory_config(
        config=config,
        dotenv_path=dotenv_path,
        model_id=model_id,
    )

    if factory_config.provider != "openai_compatible":
        raise ValueError(f"Unsupported LLM provider: {factory_config.provider}")

    api_key = os.getenv(factory_config.api_key_env)
    if not api_key and not factory_config.base_url:
        raise ValueError(
            f"Missing API key. Set environment variable {factory_config.api_key_env}."
        )

    return OpenAICompatibleLLM(
        model_id=factory_config.model_id,
        api_key=api_key,
        base_url=factory_config.base_url,
        default_temperature=factory_config.temperature,
        default_top_p=factory_config.top_p,
        default_max_tokens=factory_config.max_tokens,
        max_tokens_param=factory_config.max_tokens_param,
        timeout=factory_config.timeout,
        max_retries=factory_config.max_retries,
        output_dir=factory_config.output_dir,
        default_request_kwargs=_drop_none({
            "reasoning_effort": factory_config.reasoning_effort,
        }),
    )


def load_llm_factory_config(
    config: Optional[Config | Mapping[str, Any]] = None,
    *,
    dotenv_path: str | Path = ".env",
    model_id: Optional[str] = None,
) -> LLMFactoryConfig:
    loaded_dotenv = _load_dotenv(dotenv_path)
    cfg = _ensure_config(config)
    llm_config = _get_mapping(cfg, "llm")
    generation_config = _mapping_value(llm_config, "generation")
    retry_config = _mapping_value(llm_config, "retry")
    usage_config = _mapping_value(llm_config, "usage_tracking")

    resolved_model_id = model_id or _value(llm_config, "model_id")
    if not resolved_model_id:
        raise ValueError("Missing llm.model_id in config.")

    return LLMFactoryConfig(
        provider=str(_value(llm_config, "provider", "openai_compatible")),
        model_id=str(resolved_model_id),
        base_url=_optional_str(_value(llm_config, "base_url")),
        api_key_env=str(_value(llm_config, "api_key_env", "OPENAI_API_KEY")),
        temperature=_optional_float(_value(generation_config, "temperature")),
        top_p=_optional_float(_value(generation_config, "top_p")),
        max_tokens=_optional_int(_value(generation_config, "max_tokens")),
        max_tokens_param=str(_value(generation_config, "max_tokens_param", "max_tokens")),
        reasoning_effort=_optional_str(_value(generation_config, "reasoning_effort")),
        timeout=_optional_float(_value(generation_config, "timeout", 120)),
        max_retries=_optional_int(_value(retry_config, "max_retries")),
        backoff_seconds=_optional_float(_value(retry_config, "backoff_seconds")),
        allow_missing_usage=bool(_value(usage_config, "allow_missing_usage", True)),
        output_dir=str(_value(llm_config, "output_dir", "llm_outputs")),
        dotenv_path=loaded_dotenv,
    )


def _load_dotenv(dotenv_path: str | Path) -> Optional[str]:
    path = Path(dotenv_path)
    if path.exists():
        load_dotenv(path, override=False)
        return str(path)

    discovered = find_dotenv(usecwd=True)
    if discovered:
        load_dotenv(discovered, override=False)
        return discovered

    return None


def _ensure_config(config: Optional[Config | Mapping[str, Any]]) -> Config | Mapping[str, Any]:
    if config is not None:
        return config
    return load_config()


def _get_mapping(config: Config | Mapping[str, Any], path: str) -> Mapping[str, Any]:
    value = config.get(path, {}) if isinstance(config, Config) else config.get(path, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Config path {path!r} must be a mapping.")
    return value


def _mapping_value(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _value(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return config.get(key, default)


def _optional_str(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _drop_none(values: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}
