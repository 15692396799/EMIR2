from pathlib import Path
from typing import Any

from memory.base import MemoryBackend, RetrievalResult
from memory.config import AppConfig, load_config
from memory.registry import create_backend as _create_registered_backend, register_backend
from memory.retrieve import RetrievalRequest, create_strategy
from memory.results import (
    EpisodicMemoryItem,
    NavigationItem,
    RetrievalBundle,
    RetrievalRoundTrace,
    SemanticMemoryItem,
)


def _create_v4(config: AppConfig, **kwargs: Any) -> MemoryBackend:
    from memory.v4.memory_system import MemorySystem as V4MemorySystem
    return V4MemorySystem(config, **kwargs)


def _parse_v4_config(config: AppConfig):
    from memory.v4.config import V4MemoryConfig
    return V4MemoryConfig.from_app_config(config)


register_backend(
    "v4", _create_v4, config_parser=_parse_v4_config, replace=True,
)


def create_memory_backend(config: AppConfig, **kwargs: Any) -> MemoryBackend:
    return _create_registered_backend(config, **kwargs)


class MemorySystem:
    """Public façade selecting an isolated memory backend from configuration."""

    def __init__(self, config: AppConfig | None = None, **kwargs: Any):
        self.config = config or load_config()
        self.backend = create_memory_backend(self.config, **kwargs)

    @classmethod
    def from_config_file(cls, config_path: str | Path = "configs/default.yaml") -> "MemorySystem":
        return cls(load_config(config_path))

    def ingest_conversation(
        self,
        namespace: str,
        conversation: Any,
        metadata: dict[str, Any] | None = None,
        *,
        participants: list[dict[str, str]] | None = None,
    ) -> None:
        ingest = getattr(self.backend, "ingest", None)
        if callable(ingest):
            ingest(namespace, conversation, metadata, participants=participants)
            return
        legacy_ingest = getattr(self.backend, "ingest_conversation", None)
        if not callable(legacy_ingest):
            raise TypeError("Memory backend does not implement ingest")
        if participants is None:
            legacy_ingest(namespace, conversation, metadata)
        else:
            legacy_ingest(namespace, conversation, metadata, participants=participants)

    def retrieve(
        self,
        question: str,
        namespace: str,
        context: Any = None,
        *,
        participants: list[dict[str, str]] | None = None,
        strategy: str | None = None,
        controller_profile: str | None = None,
        capture_trajectory: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        strategy_id = strategy or self.config.retrieval.get("strategy") or "multi_round"
        controller_profile = controller_profile or self.config.retrieval.get("controller_profile")
        if strategy_id:
            strategies = self.config.retrieval.get("strategies") or {}
            strategy_config = dict(strategies.get(str(strategy_id), {}) or {})
            kwargs = {}
            if controller_profile and strategy_id == "multi_round":
                getter = getattr(self.backend, "get_controller", None)
                if callable(getter):
                    kwargs["controller"] = getter(controller_profile)
            strategy = create_strategy(str(strategy_id), **kwargs)
            result = strategy.retrieve(
                self.backend,
                RetrievalRequest(
                    question=question,
                    namespace=namespace,
                    context=context,
                    participants=tuple(participants or ()),
                    strategy=str(strategy_id), controller_profile=controller_profile,
                    capture_trajectory=capture_trajectory, metadata=dict(metadata or {}),
                ),
                strategy_config,
            )
            return RetrievalResult.from_bundle(result)

    def is_namespace_ready(
        self, namespace: str, *, participants: list[dict[str, str]] | None = None
    ) -> bool:
        ready = getattr(self.backend, "is_ready", None)
        if callable(ready):
            return bool(ready(namespace, participants=participants))
        legacy_ready = getattr(self.backend, "is_namespace_ready", None)
        if not callable(legacy_ready):
            raise TypeError("Memory backend does not implement readiness")
        if participants is None:
            return bool(legacy_ready(namespace))
        return bool(legacy_ready(namespace, participants=participants))


    def list_memory_nodes(
        self, namespace: str, *, participants: list[dict[str, str]] | None = None,
    ) -> list[Any]:
        method = getattr(self.backend, "list_memory_nodes", None)
        if not callable(method):
            raise TypeError(f"Backend {self.backend.backend_id!r} cannot enumerate memory nodes")
        return list(method(namespace, participants=participants))

    def close(self) -> None:
        self.backend.close()

    def __getattr__(self, name: str) -> Any:
        # Preserve access to backend-specific maintenance APIs used by existing
        # evaluation and tooling while keeping the agent-facing API uniform.
        return getattr(self.backend, name)

__all__ = [
    "AppConfig",
    "EpisodicMemoryItem",
    "MemoryBackend",
    "MemorySystem",
    "NavigationItem",
    "RetrievalBundle",
    "RetrievalResult",
    "RetrievalRoundTrace",
    "SemanticMemoryItem",
    "load_config",
    "create_memory_backend",
]
