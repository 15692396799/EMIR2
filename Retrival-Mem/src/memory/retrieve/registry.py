from __future__ import annotations

from collections.abc import Callable
from typing import Any

from memory.retrieve.base import RetrievalStrategy


StrategyFactory = Callable[..., RetrievalStrategy]
_STRATEGIES: dict[str, StrategyFactory] = {}


def register_strategy(strategy_id: str, factory: StrategyFactory, *, replace: bool = False) -> None:
    key = _normalize(strategy_id)
    if key in _STRATEGIES and not replace:
        raise ValueError(f"Retrieval strategy {key!r} is already registered")
    _STRATEGIES[key] = factory


def registered_strategies() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGIES))


def create_strategy(strategy_id: str, **kwargs: Any) -> RetrievalStrategy:
    key = _normalize(strategy_id)
    factory = _STRATEGIES.get(key)
    if factory is None:
        available = ", ".join(registered_strategies()) or "none"
        raise ValueError(f"Unregistered retrieval strategy: {key}; registered: {available}")
    strategy = factory(**kwargs)
    declared = _normalize(getattr(strategy, "strategy_id", key))
    if declared != key:
        raise ValueError(f"Retrieval strategy identity mismatch: {key!r} != {declared!r}")
    return strategy


def _normalize(value: Any) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    if not key or not all(char.isalnum() or char in {"_", "."} for char in key):
        raise ValueError(f"Invalid retrieval strategy id: {value!r}")
    return key
