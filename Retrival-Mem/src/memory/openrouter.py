"""OpenRouter's inline Batch API (not OpenAI's file-upload Batch API)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from memory.clients import (
    BatchJob, ChatBatchRequest, ChatBatchResult, OpenAICompatibleBatchChatClient,
    disable_request_thinking,
)


@dataclass
class OpenRouterBatchChatClient:
    batches_endpoint_url: str
    batch_status_endpoint_template: str
    model: str
    api_key: str
    temperature: float = 0.0
    timeout: int = 60
    extra_body: dict[str, Any] | None = None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def write_chat_batch_input(self, path: Path, requests_: list[ChatBatchRequest]) -> None:
        if not requests_ or len({r.custom_id for r in requests_}) != len(requests_):
            raise ValueError("Batch requests must be nonempty with unique custom_id values")
        lines = []
        for item in requests_:
            if not item.custom_id:
                raise ValueError("Batch custom_id must not be empty")
            body = dict(self.extra_body or {})
            body.update(messages=item.messages, temperature=self.temperature)
            body["model"] = self.model
            body["stream"] = False
            if item.json_mode:
                body["response_format"] = {"type": "json_object"}
            disable_request_thinking(body, "openrouter")
            lines.append({"custom_id": item.custom_id, "body": body})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
                        encoding="utf-8")

    def submit_chat_batch(self, input_path: Path, completion_window: str,
                          metadata: dict[str, str] | None = None) -> BatchJob:
        if completion_window != "24h":
            raise ValueError("OpenRouter supports only the 24h completion window")
        items = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if not items or len({item["custom_id"] for item in items}) != len(items):
            raise ValueError("Batch requests must be nonempty with unique custom_id values")
        for item in items:
            disable_request_thinking(item["body"], "openrouter")
        # Field order matters: OpenRouter stream-parses endpoint/model before requests.
        payload = {"endpoint": "/v1/chat/completions", "model": self.model,
                   "requests": items}
        response = requests.post(self.batches_endpoint_url, headers=self._headers(),
                                 json=payload, timeout=self.timeout)
        self._raise_for_status(response)
        return self._job(response.json())

    def get_batch(self, batch_id: str) -> BatchJob:
        response = requests.get(
            self.batch_status_endpoint_template.format(batch_id=quote(batch_id, safe="")),
            headers=self._headers(), timeout=self.timeout,
        )
        self._raise_for_status(response)
        return self._job(response.json())

    def _raise_for_status(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            # requests omits the API's validation error from its default message.
            detail = response.text.replace(self.api_key, "[REDACTED]") if self.api_key else response.text
            raise requests.HTTPError(
                f"{exc}; OpenRouter response: {detail[:16000]}",
                response=response, request=exc.request,
            ) from exc

    @staticmethod
    def _job(data: dict[str, Any]) -> BatchJob:
        if not data.get("id") or not data.get("status"):
            raise ValueError("OpenRouter returned an invalid batch object")
        return BatchJob(id=str(data["id"]), status=str(data["status"]), raw=data,
                        results=data.get("results"))

    def parse_chat_batch_results(self, lines: list[dict[str, Any]]) -> list[ChatBatchResult]:
        results = OpenAICompatibleBatchChatClient.parse_chat_batch_results(self, lines)
        for result in results:
            response = result.raw.get("response") or {}
            status = response.get("status_code", 200)
            body = response.get("body") or {}
            if not 200 <= status < 300 or body.get("error"):
                result.error = body.get("error") or {"status_code": status}
                result.content = None
            if result.content is not None and not isinstance(result.content, str):
                result.error = {"message": "Expected text chat completion"}
                result.content = None
        return results

    def download_batch_file(self, file_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError("OpenRouter returns results inline in get_batch")

    def cancel_batch(self, batch_id: str) -> BatchJob:
        raise NotImplementedError("OpenRouter's documented batch API has no cancellation endpoint")
