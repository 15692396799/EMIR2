from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import requests

from memory.config import ModelConfig


def disable_request_thinking(payload: dict[str, Any], provider: str) -> None:
    """Apply the client-wide policy after merging all caller-supplied fields."""
    for key in ("reasoning", "reasoning_effort", "thinking", "think", "enable_thinking", "include_reasoning"):
        payload.pop(key, None)
    template = payload.get("chat_template_kwargs")
    if isinstance(template, dict):
        payload["chat_template_kwargs"] = {
            **template, "enable_thinking": False, "thinking": False,
        }
    if provider == "openrouter":
        payload["reasoning"] = {"enabled": False, "effort": "none"}
    elif provider in {"dashscope_bailian", "modelscope_openai_compatible"}:
        payload["enable_thinking"] = False
    elif provider == "vllm_openai_compatible":
        payload["chat_template_kwargs"] = {
            **(payload.get("chat_template_kwargs") or {}),
            "enable_thinking": False, "thinking": False,
        }
    else:
        model = str(payload.get("model", "")).lower().split("/")[-1]
        if model.startswith(("gpt-5", "o1", "o3", "o4")):
            payload["reasoning_effort"] = "none"
        elif model.startswith("claude"):
            payload["thinking"] = {"type": "disabled"}
        elif model.startswith("gemini"):
            payload["reasoning_effort"] = "none"
        elif not model.startswith(("gpt-", "chatgpt-")):
            payload["enable_thinking"] = False


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        json_schema: Mapping[str, Any] | None = None,
    ) -> str:
        ...


def with_output_token_limit(config: ModelConfig, limit: int) -> ModelConfig:
    """Clone a model config with a provider-native completion-token limit."""
    bounded = max(1, int(limit))
    extra = dict(config.extra)
    provider = config.provider.lower()
    if provider == "ollama":
        extra["num_predict"] = bounded
    else:
        body = dict(extra.get("extra_body") or {})
        body["max_tokens"] = bounded
        extra["extra_body"] = body
    return replace(config, extra=extra)


@dataclass
class ChatCallResult:
    content: str
    token_usage: dict[str, Any] | None = None
    raw_usage: Any | None = None


@dataclass
class ChatBatchRequest:
    custom_id: str
    messages: list[dict[str, str]]
    json_mode: bool = False


@dataclass
class BatchJob:
    id: str
    status: str
    raw: dict[str, Any]
    input_file_id: str | None = None
    output_file_id: str | None = None
    error_file_id: str | None = None
    results: list[dict[str, Any]] | None = None


@dataclass
class ChatBatchResult:
    custom_id: str
    content: str | None
    raw: dict[str, Any]
    error: dict[str, Any] | None = None
    token_usage: dict[str, Any] | None = None
    raw_usage: Any | None = None


class BatchChatClient(Protocol):
    def write_chat_batch_input(self, path: Path, requests_: list[ChatBatchRequest]) -> None:
        ...

    def submit_chat_batch(
        self,
        input_path: Path,
        completion_window: str,
        metadata: dict[str, str] | None = None,
    ) -> BatchJob:
        ...

    def get_batch(self, batch_id: str) -> BatchJob:
        ...

    def download_batch_file(self, file_id: str) -> list[dict[str, Any]]:
        ...

    def parse_chat_batch_results(self, lines: list[dict[str, Any]]) -> list[ChatBatchResult]:
        ...

    def cancel_batch(self, batch_id: str) -> BatchJob:
        ...


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class NoopChatClient:
    def chat(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        json_schema: Mapping[str, Any] | None = None,
    ) -> str:
        return self.chat_with_metadata(
            messages, json_mode=json_mode, json_schema=json_schema
        ).content

    def chat_with_metadata(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        json_schema: Mapping[str, Any] | None = None,
    ) -> ChatCallResult:
        if json_mode:
            return ChatCallResult('{"directions":[],"facts":[],"sufficient":true,"selected_navigation_ids":[]}')
        return ChatCallResult("")


@dataclass
class OllamaChatClient:
    endpoint_url: str
    model: str
    temperature: float = 0.0
    timeout: int = 120
    options: dict[str, Any] | None = None

    def chat(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        json_schema: Mapping[str, Any] | None = None,
    ) -> str:
        return self.chat_with_metadata(
            messages, json_mode=json_mode, json_schema=json_schema
        ).content

    def chat_with_metadata(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        json_schema: Mapping[str, Any] | None = None,
    ) -> ChatCallResult:
        options = {"temperature": self.temperature}
        options.update(self.options or {})
        options.pop("think", None)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": options,
        }
        if json_mode:
            payload["format"] = dict(json_schema) if json_schema is not None else "json"
        response = requests.post(self.endpoint_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        raw_usage = _ollama_raw_usage(data)
        return ChatCallResult(
            content=data.get("message", {}).get("content", ""),
            token_usage=_standard_token_usage(raw_usage),
            raw_usage=raw_usage,
        )


@dataclass
class OllamaEmbeddingClient:
    endpoint_url: str
    legacy_endpoint_url: str
    model: str
    timeout: int = 120

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts}
        response = requests.post(self.endpoint_url, json=payload, timeout=self.timeout)
        if response.status_code == 404:
            return [self._embed_one_legacy(text) for text in texts]
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if embeddings is None and "embedding" in data:
            embeddings = [data["embedding"]]
        if embeddings is None:
            raise RuntimeError(f"Ollama embedding response has no embeddings: {data}")
        return embeddings

    def _embed_one_legacy(self, text: str) -> list[float]:
        payload = {"model": self.model, "prompt": text}
        response = requests.post(self.legacy_endpoint_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["embedding"]


@dataclass
class OpenAICompatibleChatClient:
    endpoint_url: str
    model: str
    api_key: str | None = None
    temperature: float = 0.0
    timeout: int = 60
    extra_body: dict[str, Any] | None = None
    provider: str = "openai"

    def chat(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        json_schema: Mapping[str, Any] | None = None,
    ) -> str:
        return self.chat_with_metadata(
            messages, json_mode=json_mode, json_schema=json_schema
        ).content

    def chat_with_metadata(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        json_schema: Mapping[str, Any] | None = None,
    ) -> ChatCallResult:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        payload.update(self.extra_body or {})
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        disable_request_thinking(payload, self.provider)
        response = requests.post(
            self.endpoint_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        raw_usage = _raw_usage_from_response(data)
        return ChatCallResult(
            content=data["choices"][0]["message"]["content"],
            token_usage=_standard_token_usage(raw_usage),
            raw_usage=raw_usage,
        )


@dataclass
class OpenAICompatibleBatchChatClient:
    chat_completions_endpoint_url: str
    files_endpoint_url: str
    file_content_endpoint_template: str
    batches_endpoint_url: str
    batch_status_endpoint_template: str
    batch_cancel_endpoint_template: str
    batch_request_url: str
    model: str
    api_key: str | None = None
    temperature: float = 0.0
    timeout: int = 60
    extra_body: dict[str, Any] | None = None
    provider: str = "openai"

    def write_chat_batch_input(self, path: Path, requests_: list[ChatBatchRequest]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for request_ in requests_:
                handle.write(json.dumps(self._batch_line(request_), ensure_ascii=False) + "\n")

    def submit_chat_batch(
        self,
        input_path: Path,
        completion_window: str,
        metadata: dict[str, str] | None = None,
    ) -> BatchJob:
        input_file = self._upload_batch_file(input_path)
        payload: dict[str, Any] = {
            "input_file_id": input_file["id"],
            "endpoint": self.batch_request_url,
            "completion_window": completion_window,
        }
        if metadata:
            payload["metadata"] = metadata
        response = requests.post(
            self.batches_endpoint_url,
            headers=self._json_headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._batch_job(response.json())

    def get_batch(self, batch_id: str) -> BatchJob:
        response = requests.get(
            self.batch_status_endpoint_template.format(batch_id=batch_id),
            headers=self._json_headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._batch_job(response.json())

    def download_batch_file(self, file_id: str) -> list[dict[str, Any]]:
        response = requests.get(
            self.file_content_endpoint_template.format(file_id=file_id),
            headers=self._json_headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        lines: list[dict[str, Any]] = []
        for raw_line in response.text.splitlines():
            if raw_line.strip():
                lines.append(json.loads(raw_line))
        return lines

    def parse_chat_batch_results(self, lines: list[dict[str, Any]]) -> list[ChatBatchResult]:
        results: list[ChatBatchResult] = []
        for line in lines:
            custom_id = str(line.get("custom_id") or "")
            error = line.get("error")
            content = None
            raw_usage = None
            response = line.get("response")
            if isinstance(response, dict):
                body = response.get("body")
                if isinstance(body, dict):
                    raw_usage = _raw_usage_from_response(body)
                    try:
                        content = body["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, TypeError):
                        content = None
            results.append(
                ChatBatchResult(
                    custom_id=custom_id,
                    content=content,
                    raw=line,
                    error=error,
                    token_usage=_standard_token_usage(raw_usage),
                    raw_usage=raw_usage,
                )
            )
        return results

    def cancel_batch(self, batch_id: str) -> BatchJob:
        response = requests.post(
            self.batch_cancel_endpoint_template.format(batch_id=batch_id),
            headers=self._json_headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._batch_job(response.json())

    def _batch_line(self, request_: ChatBatchRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": request_.messages,
            "temperature": self.temperature,
        }
        body.update(self.extra_body or {})
        if request_.json_mode:
            body["response_format"] = {"type": "json_object"}
        disable_request_thinking(body, self.provider)
        return {
            "custom_id": request_.custom_id,
            "method": "POST",
            "url": self.batch_request_url,
            "body": body,
        }

    def _upload_batch_file(self, path: Path) -> dict[str, Any]:
        headers = self._auth_headers()
        with path.open("rb") as handle:
            response = requests.post(
                self.files_endpoint_url,
                headers=headers,
                data={"purpose": "batch"},
                files={"file": (path.name, handle, "application/jsonl")},
                timeout=self.timeout,
            )
        response.raise_for_status()
        return response.json()

    def _json_headers(self) -> dict[str, str]:
        headers = self._auth_headers()
        headers["Content-Type"] = "application/json"
        return headers

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _batch_job(self, data: dict[str, Any]) -> BatchJob:
        return BatchJob(
            id=str(data.get("id") or ""),
            status=str(data.get("status") or ""),
            raw=data,
            input_file_id=data.get("input_file_id"),
            output_file_id=data.get("output_file_id"),
            error_file_id=data.get("error_file_id"),
        )


@dataclass
class OpenAICompatibleEmbeddingClient:
    endpoint_url: str
    model: str
    api_key: str | None = None
    timeout: int = 60

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "input": texts}
        response = requests.post(
            self.endpoint_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in data]


@dataclass
class HashEmbeddingClient:
    dimensions: int = 384

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text).tolist() for text in texts]

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in re.findall(r"[\w']+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector


@dataclass
class ResilientEmbeddingClient:
    primary: EmbeddingClient
    fallback: EmbeddingClient

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.primary.embed_texts(texts)
        except Exception:
            return self.fallback.embed_texts(texts)


def _hash_fallback_dimensions(config: ModelConfig) -> int:
    for key in ("fallback_dimensions", "embedding_dimensions", "dimensions"):
        value = config.extra.get(key)
        if value is None:
            continue
        try:
            dimensions = int(value)
        except (TypeError, ValueError):
            continue
        if dimensions > 0:
            return dimensions
    model = config.model.lower().split(":", 1)[0]
    known_dimensions = {
        "all-minilm": 384,
        "all-minilm-l6-v2": 384,
        "bge-m3": 1024,
        "mxbai-embed-large": 1024,
        "nomic-embed-text": 768,
        "qwen3-embedding": 4096,
    }
    return known_dimensions.get(model, 384)


def _standard_token_usage(raw_usage: Any) -> dict[str, Any] | None:
    if not isinstance(raw_usage, dict):
        return None
    prompt_tokens = _number_or_none(
        raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", raw_usage.get("prompt_eval_count")))
    )
    completion_tokens = _number_or_none(
        raw_usage.get("completion_tokens", raw_usage.get("output_tokens", raw_usage.get("eval_count")))
    )
    total_tokens = _number_or_none(raw_usage.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _ollama_raw_usage(data: dict[str, Any]) -> dict[str, Any] | None:
    usage = _usage_related_fields(data)
    return usage or None


def _raw_usage_from_response(data: dict[str, Any]) -> Any | None:
    raw_usage = data.get("usage")
    if isinstance(raw_usage, dict):
        usage = dict(raw_usage)
        for key, value in _usage_related_fields(data).items():
            if key != "usage":
                usage[key] = value
        return usage
    if raw_usage is not None:
        return raw_usage
    usage = _usage_related_fields(data)
    return usage or None


def _usage_related_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if any(marker in key.lower() for marker in ("usage", "token", "count", "eval", "duration"))
    }


def _number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def make_chat_client(config: ModelConfig) -> ChatClient:
    from memory.ollama_routing import ollama_endpoint
    provider = config.provider.lower()
    if provider == "openrouter" and config.model.endswith(":batch"):
        raise ValueError("OpenRouter :batch models require batch mode, not chat/completions")
    if provider == "noop":
        return NoopChatClient()
    if provider == "ollama":
        endpoint_url = ollama_endpoint("chat") or os.environ.get(str(config.extra.get("chat_endpoint_env") or "OLLAMA_CHAT_ENDPOINT"))
        if not endpoint_url:
            raise ValueError("Ollama chat client requires OLLAMA_CHAT_ENDPOINT")
        request_fields = {"chat_endpoint_env", "think", "timeout"}
        options = {key: value for key, value in config.extra.items() if key not in request_fields}
        return OllamaChatClient(
            endpoint_url=endpoint_url,
            model=config.model,
            temperature=config.temperature,
            timeout=int(config.extra.get("timeout", 120)),
            options=options,
        )
    if provider in {
        "openai",
        "vllm_openai_compatible",
        "modelscope_openai_compatible",
        "dashscope_bailian",
        "greatrouter",
        "openrouter",
    }:
        endpoint_url = config.resolved_chat_completions_endpoint
        if not endpoint_url:
            raise ValueError(f"{provider} chat client requires a resolved chat completions endpoint")
        api_key = config.resolved_api_key
        if provider in {"openai", "dashscope_bailian", "greatrouter", "openrouter"} and not api_key:
            api_key_env = config.api_key_env or config.default_api_key_env or "API key env"
            raise ValueError(f"{provider} chat client requires {api_key_env}")
        return OpenAICompatibleChatClient(
            provider=provider,
            endpoint_url=endpoint_url,
            model=config.model,
            api_key=api_key,
            temperature=config.temperature,
            extra_body=_chat_extra_body(config),
            timeout=int(config.extra.get("timeout", 60)),
        )
    raise ValueError(f"Unsupported chat provider: {config.provider}")


def make_batch_chat_client(config: ModelConfig) -> BatchChatClient:
    provider = config.provider.lower()
    if provider == "openrouter":
        from memory.openrouter import OpenRouterBatchChatClient

        if not config.resolved_api_key:
            raise ValueError("openrouter batch client requires OPENROUTER_API_KEY")
        endpoint = config.resolved_batches_endpoint
        if not endpoint:
            raise ValueError("openrouter batch client requires OPENROUTER_BATCHES_ENDPOINT")
        return OpenRouterBatchChatClient(
            batches_endpoint_url=endpoint,
            batch_status_endpoint_template=(config.resolved_batch_status_endpoint_template
                                            or endpoint.rstrip("/") + "/{batch_id}"),
            model=config.model.removesuffix(":batch"),
            api_key=config.resolved_api_key,
            temperature=config.temperature,
            timeout=int(config.extra.get("timeout", 60)),
            extra_body=_chat_extra_body(config),
        )
    if provider not in {"openai", "dashscope_bailian"}:
        raise ValueError(f"{config.provider} does not support LoCoMo batch judge evaluation")
    required = {
        "chat completions endpoint": config.resolved_chat_completions_endpoint,
        "files endpoint": config.resolved_files_endpoint,
        "file content endpoint template": config.resolved_file_content_endpoint_template,
        "batches endpoint": config.resolved_batches_endpoint,
        "batch status endpoint template": config.resolved_batch_status_endpoint_template,
        "batch cancel endpoint template": config.resolved_batch_cancel_endpoint_template,
        "batch request URL": config.resolved_batch_request_url,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"{provider} batch client requires resolved {', '.join(missing)}")
    api_key = config.resolved_api_key
    if not api_key:
        api_key_env = config.api_key_env or config.default_api_key_env or "API key env"
        raise ValueError(f"{provider} batch client requires {api_key_env}")
    return OpenAICompatibleBatchChatClient(
        provider=provider,
        chat_completions_endpoint_url=required["chat completions endpoint"] or "",
        files_endpoint_url=required["files endpoint"] or "",
        file_content_endpoint_template=required["file content endpoint template"] or "",
        batches_endpoint_url=required["batches endpoint"] or "",
        batch_status_endpoint_template=required["batch status endpoint template"] or "",
        batch_cancel_endpoint_template=required["batch cancel endpoint template"] or "",
        batch_request_url=required["batch request URL"] or "",
        model=config.model,
        api_key=api_key,
        temperature=config.temperature,
        extra_body=_chat_extra_body(config),
    )


def _chat_extra_body(config: ModelConfig) -> dict[str, Any] | None:
    body = dict(config.extra.get("extra_body") or {})
    if config.extra.get("adapter"):
        body.setdefault("adapter", config.extra["adapter"])
    return body or None


def make_embedding_client(config: ModelConfig, resilient: bool = True) -> EmbeddingClient:
    from memory.ollama_routing import ollama_endpoint
    provider = config.provider.lower()
    if provider == "hash":
        return HashEmbeddingClient()
    if provider == "ollama":
        endpoint_url = ollama_endpoint("embed") or os.environ.get(str(config.extra.get("embed_endpoint_env") or "OLLAMA_EMBED_ENDPOINT"))
        legacy_endpoint_url = ollama_endpoint("legacy_embed") or os.environ.get(
            str(config.extra.get("legacy_embeddings_endpoint_env") or "OLLAMA_LEGACY_EMBEDDINGS_ENDPOINT")
        )
        if not endpoint_url or not legacy_endpoint_url:
            raise ValueError("Ollama embedding client requires OLLAMA_EMBED_ENDPOINT and OLLAMA_LEGACY_EMBEDDINGS_ENDPOINT")
        client: EmbeddingClient = OllamaEmbeddingClient(
            endpoint_url=endpoint_url,
            legacy_endpoint_url=legacy_endpoint_url,
            model=config.model,
        )
        fallback = HashEmbeddingClient(dimensions=_hash_fallback_dimensions(config))
        return ResilientEmbeddingClient(client, fallback) if resilient else client
    if provider in {"openai", "vllm_openai_compatible", "modelscope_openai_compatible", "dashscope_bailian"}:
        endpoint_url = config.resolved_embeddings_endpoint
        if not endpoint_url:
            raise ValueError(f"{provider} embedding client requires a resolved embeddings endpoint")
        api_key = config.resolved_api_key
        if provider in {"openai", "dashscope_bailian"} and not api_key:
            api_key_env = config.api_key_env or config.default_api_key_env or "API key env"
            raise ValueError(f"{provider} embedding client requires {api_key_env}")
        client = OpenAICompatibleEmbeddingClient(
            endpoint_url=endpoint_url,
            model=config.model,
            api_key=api_key,
        )
        fallback = HashEmbeddingClient(dimensions=_hash_fallback_dimensions(config))
        return ResilientEmbeddingClient(client, fallback) if resilient else client
    raise ValueError(f"Unsupported embedding provider: {config.provider}")


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def parse_json_object_strict(
    text: str,
    *,
    source: str = "Structured model",
    bare_list_field: str | None = None,
) -> dict[str, Any]:
    """Parse a JSON object, optionally wrapping a task-approved bare list."""
    if not str(text).strip():
        raise ValueError(f"{source} returned an empty JSON response")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as direct_error:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(
                f"{source} returned invalid or truncated JSON"
            ) from direct_error
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as embedded_error:
            raise ValueError(
                f"{source} returned invalid or truncated JSON"
            ) from embedded_error
    if isinstance(value, list) and bare_list_field is not None:
        return {bare_list_field: value}
    if not isinstance(value, dict):
        raise ValueError(f"{source} returned JSON that is not an object")
    if not value:
        raise ValueError(f"{source} returned an empty JSON object")
    return value
