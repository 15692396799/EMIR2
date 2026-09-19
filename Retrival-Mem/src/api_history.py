from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


class ApiHistoryLogger:
    def __init__(self, output_dir: str | Path | None):
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._lock = threading.Lock()
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.output_dir is not None

    def log(self, module: str, record: dict[str, Any]) -> None:
        if self.output_dir is None:
            return
        path = self.output_dir / f"{self._safe_module(module)}.jsonl"
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def call_chat(
        self,
        *,
        module: str,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        json_mode: bool,
        json_schema: Mapping[str, Any] | None,
        call: Callable[[], Any],
        context: dict[str, Any] | None = None,
    ) -> str:
        start = time.perf_counter()
        timestamp = _utc_timestamp()
        try:
            result = call()
            content, token_usage, raw_usage = _chat_result_parts(result)
        except Exception as exc:
            self.log(
                module,
                {
                    "timestamp": timestamp,
                    "module": module,
                    "provider": provider,
                    "model": model,
                    "json_mode": json_mode,
                    "context": _jsonable(context or {}),
                    "request": {
                        "messages": _jsonable(messages),
                        "json_schema": _jsonable(json_schema),
                    },
                    "response": None,
                    "token_usage": None,
                    "raw_usage": None,
                    "duration_ms": _elapsed_ms(start),
                    "success": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
            raise
        self.log(
            module,
            {
                "timestamp": timestamp,
                "module": module,
                "provider": provider,
                "model": model,
                "json_mode": json_mode,
                "context": _jsonable(context or {}),
                "request": {
                    "messages": _jsonable(messages),
                    "json_schema": _jsonable(json_schema),
                },
                "response": {"content": content},
                "token_usage": _jsonable(token_usage),
                "raw_usage": _jsonable(raw_usage),
                "duration_ms": _elapsed_ms(start),
                "success": True,
                "error": None,
            },
        )
        return content

    def log_operation(
        self,
        *,
        module: str,
        provider: str,
        model: str,
        operation: str,
        request: dict[str, Any],
        call: Callable[[], Any],
        context: dict[str, Any] | None = None,
    ) -> Any:
        start = time.perf_counter()
        timestamp = _utc_timestamp()
        try:
            response = call()
        except Exception as exc:
            self.log(
                module,
                {
                    "timestamp": timestamp,
                    "module": module,
                    "provider": provider,
                    "model": model,
                    "operation": operation,
                    "context": _jsonable(context or {}),
                    "request": _jsonable(request),
                    "response": None,
                    "token_usage": None,
                    "raw_usage": None,
                    "duration_ms": _elapsed_ms(start),
                    "success": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
            raise
        token_usage, raw_usage = _operation_token_metadata(response)
        self.log(
            module,
            {
                "timestamp": timestamp,
                "module": module,
                "provider": provider,
                "model": model,
                "operation": operation,
                "context": _jsonable(context or {}),
                "request": _jsonable(request),
                "response": _jsonable(response),
                "token_usage": _jsonable(token_usage),
                "raw_usage": _jsonable(raw_usage),
                "duration_ms": _elapsed_ms(start),
                "success": True,
                "error": None,
            },
        )
        return response

    def _safe_module(self, module: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", module.strip())
        return safe or "unknown"


class LoggedChatClient:
    def __init__(
        self,
        base_client,
        logger: ApiHistoryLogger | None,
        module: str,
        provider: str,
        model: str,
        context: dict[str, Any] | None = None,
    ):
        self.base_client = base_client
        self.logger = logger
        self.module = module
        self.provider = provider
        self.model = model
        self.context = context or {}

    def chat(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        json_schema: Mapping[str, Any] | None = None,
    ) -> str:
        if self.logger is None:
            return self.base_client.chat(
                messages, json_mode=json_mode, json_schema=json_schema
            )
        return self.logger.call_chat(
            module=self.module,
            provider=self.provider,
            model=self.model,
            messages=messages,
            json_mode=json_mode,
            json_schema=json_schema,
            call=lambda: call_chat_with_metadata(
                self.base_client,
                messages,
                json_mode=json_mode,
                json_schema=json_schema,
            ),
            context=self.context,
        )


class LoggedBatchChatClient:
    def __init__(
        self,
        base_client,
        logger: ApiHistoryLogger | None,
        module: str,
        provider: str,
        model: str,
        context: dict[str, Any] | None = None,
    ):
        self.base_client = base_client
        self.logger = logger
        self.module = module
        self.provider = provider
        self.model = model
        self.context = context or {}

    def write_chat_batch_input(self, path: Path, requests_: list[Any]) -> None:
        request = {
            "path": str(path),
            "requests": [_batch_request_to_dict(item) for item in requests_],
        }
        if self.logger is None:
            return self.base_client.write_chat_batch_input(path, requests_)
        return self.logger.log_operation(
            module=self.module,
            provider=self.provider,
            model=self.model,
            operation="write_chat_batch_input",
            request=request,
            call=lambda: self.base_client.write_chat_batch_input(path, requests_),
            context=self.context,
        )

    def submit_chat_batch(self, input_path: Path, completion_window: str, metadata: dict[str, str] | None = None):
        request = {
            "input_path": str(input_path),
            "completion_window": completion_window,
            "metadata": metadata or {},
        }
        if self.logger is None:
            return self.base_client.submit_chat_batch(input_path, completion_window, metadata=metadata)
        return self.logger.log_operation(
            module=self.module,
            provider=self.provider,
            model=self.model,
            operation="submit_chat_batch",
            request=request,
            call=lambda: self.base_client.submit_chat_batch(input_path, completion_window, metadata=metadata),
            context=self.context,
        )

    def get_batch(self, batch_id: str):
        if self.logger is None:
            return self.base_client.get_batch(batch_id)
        return self.logger.log_operation(
            module=self.module,
            provider=self.provider,
            model=self.model,
            operation="get_batch",
            request={"batch_id": batch_id},
            call=lambda: self.base_client.get_batch(batch_id),
            context=self.context,
        )

    def download_batch_file(self, file_id: str) -> list[dict[str, Any]]:
        if self.logger is None:
            return self.base_client.download_batch_file(file_id)
        return self.logger.log_operation(
            module=self.module,
            provider=self.provider,
            model=self.model,
            operation="download_batch_file",
            request={"file_id": file_id},
            call=lambda: self.base_client.download_batch_file(file_id),
            context=self.context,
        )

    def parse_chat_batch_results(self, lines: list[dict[str, Any]]):
        return self.base_client.parse_chat_batch_results(lines)

    def cancel_batch(self, batch_id: str):
        if self.logger is None:
            return self.base_client.cancel_batch(batch_id)
        return self.logger.log_operation(
            module=self.module,
            provider=self.provider,
            model=self.model,
            operation="cancel_batch",
            request={"batch_id": batch_id},
            call=lambda: self.base_client.cancel_batch(batch_id),
            context=self.context,
        )


def _batch_request_to_dict(request_: Any) -> dict[str, Any]:
    return {
        "custom_id": getattr(request_, "custom_id", ""),
        "messages": _jsonable(getattr(request_, "messages", [])),
        "json_mode": bool(getattr(request_, "json_mode", False)),
    }


def call_chat_with_metadata(
    client: Any,
    messages: list[dict[str, str]],
    json_mode: bool = False,
    json_schema: Mapping[str, Any] | None = None,
) -> Any:
    chat_with_metadata = getattr(client, "chat_with_metadata", None)
    if callable(chat_with_metadata):
        return chat_with_metadata(
            messages, json_mode=json_mode, json_schema=json_schema
        )
    return client.chat(
        messages, json_mode=json_mode, json_schema=json_schema
    )


def _chat_result_parts(result: Any) -> tuple[str, dict[str, Any] | None, Any | None]:
    if hasattr(result, "content"):
        content = getattr(result, "content")
        token_usage = getattr(result, "token_usage", None)
        raw_usage = getattr(result, "raw_usage", None)
        return _content_to_str(content), _jsonable(token_usage) if token_usage is not None else None, (
            _jsonable(raw_usage) if raw_usage is not None else None
        )
    if isinstance(result, dict) and "content" in result:
        token_usage = result.get("token_usage")
        raw_usage = result.get("raw_usage")
        return _content_to_str(result.get("content")), _jsonable(token_usage) if token_usage is not None else None, (
            _jsonable(raw_usage) if raw_usage is not None else None
        )
    return _content_to_str(result), None, None


def _operation_token_metadata(response: Any) -> tuple[dict[str, Any] | None, Any | None]:
    # OpenRouter batch jobs carry completed results inline, without a download call.
    if isinstance(getattr(response, "results", None), list):
        return _batch_token_metadata(response.results)
    if hasattr(response, "token_usage") or hasattr(response, "raw_usage"):
        token_usage = getattr(response, "token_usage", None)
        raw_usage = getattr(response, "raw_usage", None)
        return _jsonable(token_usage) if token_usage is not None else None, (
            _jsonable(raw_usage) if raw_usage is not None else None
        )
    if isinstance(response, list):
        return _batch_token_metadata(response)
    raw_usage = _raw_usage_from_response(response)
    if raw_usage is None:
        return None, None
    return _standard_token_usage(raw_usage), _jsonable(raw_usage)


def _batch_token_metadata(lines: list[Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    items: list[dict[str, Any]] = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    seen = {key: False for key in totals}

    for line in lines:
        raw_usage = _raw_usage_from_response(line)
        if raw_usage is None:
            continue
        item = {"usage": _jsonable(raw_usage)}
        if isinstance(line, dict) and line.get("custom_id") is not None:
            item["custom_id"] = str(line["custom_id"])
        items.append(item)
        standard = _standard_token_usage(raw_usage)
        if not standard:
            continue
        for key in totals:
            value = standard.get(key)
            if isinstance(value, (int, float)):
                totals[key] += value
                seen[key] = True

    if not items:
        return None, None

    token_usage: dict[str, Any] = {
        key: totals[key] if seen[key] else None
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    token_usage["request_count"] = len(lines)
    token_usage["request_count_with_usage"] = len(items)
    return token_usage, {"items": items}


def _raw_usage_from_response(response: Any) -> Any | None:
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if isinstance(usage, dict):
        merged = dict(usage)
        for key, value in _usage_related_fields(response).items():
            if key != "usage":
                merged[key] = value
        return merged
    if usage is not None:
        return usage
    nested_response = response.get("response")
    if isinstance(nested_response, dict):
        body = nested_response.get("body")
        if isinstance(body, dict):
            nested_usage = _raw_usage_from_response(body)
            if nested_usage is not None:
                return nested_usage
    body = response.get("body")
    if isinstance(body, dict):
        nested_usage = _raw_usage_from_response(body)
        if nested_usage is not None:
            return nested_usage
    related = _usage_related_fields(response)
    return related or None


def _usage_related_fields(response: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in response.items()
        if any(marker in key.lower() for marker in ("usage", "token", "count", "eval", "duration"))
    }


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


def _number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _content_to_str(content: Any) -> str:
    if content is None:
        return ""
    return content if isinstance(content, str) else str(content)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _jsonable(vars(value))
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_jsonable(item) for item in value]
        return str(value)
