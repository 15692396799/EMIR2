"""Azure OpenAI chat client for the roles this experiment owns.

Azure is not a drop-in for the OpenAI-compatible client used everywhere else:

* the path carries the *deployment* (``/openai/deployments/<name>/chat/completions``)
  and an ``api-version`` query parameter,
* authentication is normally the ``api-key`` header, not ``Authorization: Bearer``,
* the payload is otherwise OpenAI's.

The upstream clients (which the experiment keeps read-only) always send a
Bearer token, so this module supplies a small client of the same shape for the
roles we construct ourselves: the answer model and the judge. It exists so the
judge can move off OpenRouter's geo-gated ``openai/gpt-4o-mini`` without
touching ``Retrival-Mem``.

Configuration comes from the environment (see ``Experiment/.env.example``):

* ``AZURE_OPENAI_ENDPOINT``   e.g. ``https://my-resource.openai.azure.com``
* ``AZURE_OPENAI_DEPLOYMENT`` deployment name, e.g. ``gpt-4o-mini``
* ``AZURE_OPENAI_API_VERSION`` e.g. ``2024-10-21``
* ``AZURE_API_KEY``
* ``AZURE_OPENAI_AUTH_HEADER`` optional, ``api-key`` (default) or
  ``authorization`` for gateways that want a Bearer token instead
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import requests


class AzureConfigurationError(RuntimeError):
    """Raised when the Azure environment is incomplete."""


def _resolve(model_config: Any, key: str, env: str, default: str | None = None) -> str:
    """Config ``extra`` first, then the environment."""
    extra = getattr(model_config, "extra", None) or {}
    value = extra.get(key)
    if value in (None, ""):
        value = os.getenv(env)
    value = "" if value is None else str(value)
    if not value:
        if default is not None:
            return default
        raise AzureConfigurationError(f"Azure client requires {env} (or {key} in the config)")
    return value


def build_azure_endpoint(endpoint: str, deployment: str, api_version: str) -> str:
    base = str(endpoint).strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise AzureConfigurationError(f"AZURE_OPENAI_ENDPOINT must be a URL, got {base!r}")
    if deployment:
        if "/deployments/" in base:
            path = base
        else:
            path = f"{base}/openai/deployments/{deployment}/chat/completions"
    else:
        path = base
    joiner = "&" if "?" in path else "?"
    return f"{path}{joiner}api-version={api_version}"


@dataclass
class AzureChatClient:
    """Minimal chat client with the same ``chat`` surface as upstream's."""

    endpoint: str
    deployment: str
    api_version: str
    api_key: str
    temperature: float = 0.0
    timeout: int = 120
    max_tokens: int | None = None
    auth_header: str = "api-key"
    url: str = field(init=False)

    def __post_init__(self) -> None:
        self.url = build_azure_endpoint(self.endpoint, self.deployment, self.api_version)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_header.lower() == "api-key":
            headers["api-key"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        json_schema: Mapping[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.deployment:
            payload["model"] = self.deployment
        if self.max_tokens:
            payload["max_tokens"] = int(self.max_tokens)
        if json_mode:
            # Azure accepts json_schema on recent api-versions and falls back to
            # json_object everywhere else, so send the richer form when we have it.
            if json_schema is not None:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "memconflict_judge",
                        "schema": dict(json_schema),
                        "strict": False,
                    },
                }
            else:
                payload["response_format"] = {"type": "json_object"}
        response = requests.post(
            self.url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def build_azure_chat_client(model_config: Any):
    """Build an :class:`AzureChatClient` from a Retrival-Mem model config."""
    extra = getattr(model_config, "extra", None) or {}
    body = extra.get("extra_body") or {}
    max_tokens = extra.get("max_tokens") or body.get("max_tokens")
    return AzureChatClient(
        endpoint=_resolve(model_config, "endpoint", "AZURE_OPENAI_ENDPOINT"),
        deployment=str(extra.get("deployment") or os.getenv("AZURE_OPENAI_DEPLOYMENT") or ""),
        api_version=_resolve(
            model_config, "api_version", "AZURE_OPENAI_API_VERSION", "2024-10-21"
        ),
        api_key=_resolve(model_config, "api_key", "AZURE_API_KEY"),
        temperature=float(getattr(model_config, "temperature", 0.0) or 0.0),
        timeout=int(extra.get("timeout") or getattr(model_config, "timeout", 120) or 120),
        max_tokens=max_tokens,
        auth_header=str(
            extra.get("auth_header") or os.getenv("AZURE_OPENAI_AUTH_HEADER") or "api-key"
        ),
    )


__all__ = [
    "AzureChatClient",
    "AzureConfigurationError",
    "build_azure_chat_client",
    "build_azure_endpoint",
]
