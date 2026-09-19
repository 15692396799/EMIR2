from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from memory.config import AppConfig


BackendFactory = Callable[..., Any]
ConfigParser = Callable[[AppConfig], Any]


@dataclass(frozen=True)
class BackendRegistration:
    backend_id: str
    factory: BackendFactory
    config_parser: ConfigParser | None = None
    capabilities: Any = None


_BACKENDS: dict[str, BackendRegistration] = {}
_REQUIRED_ADAPTER_METHODS = {
    "global_search",
    "expand",
    "finalize",
    "close",
}


def register_backend(
    backend_id: str,
    factory: BackendFactory,
    *,
    config_parser: ConfigParser | None = None,
    capabilities: Any = None,
    replace: bool = False,
) -> None:
    """Register a memory backend factory without importing the backend eagerly."""
    key = _normalise_backend_id(backend_id)
    if key in _BACKENDS and not replace:
        raise ValueError(f"Memory backend {key!r} is already registered")
    _BACKENDS[key] = BackendRegistration(
        key, factory, config_parser, capabilities
    )


def registered_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def load_factory(spec: str) -> BackendFactory:
    """Load a `module:factory` entry point from an architecture directory."""
    module_name, separator, attribute = str(spec).partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("memory.adapter must use the form 'module.path:create_backend'")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise TypeError(f"Configured memory adapter {spec!r} is not callable")
    return factory


def create_backend(config: AppConfig, **kwargs: Any) -> Any:
    backend_id = _normalise_backend_id(config.memory.backend)
    adapter_spec = config.memory.adapter
    registration = _BACKENDS.get(backend_id)
    factory = load_factory(adapter_spec) if adapter_spec else (registration.factory if registration else None)
    if factory is None:
        available = ", ".join(registered_backends()) or "none"
        raise ValueError(
            f"Unsupported memory backend: {backend_id}. Registered backends: {available}; "
            "external backends must set memory.adapter"
        )
    parsed_config = registration.config_parser(config) if registration and registration.config_parser else None
    if parsed_config is not None:
        kwargs.setdefault("backend_config", parsed_config)
    backend = factory(config, **kwargs)
    declared_id = getattr(backend, "backend_id", getattr(getattr(backend, "capabilities", None), "backend_id", backend_id))
    if _normalise_backend_id(declared_id) != backend_id:
        raise ValueError(
            f"Backend identity mismatch: config requests {backend_id!r}, adapter declares {declared_id!r}"
        )
    if adapter_spec:
        missing = sorted(name for name in _REQUIRED_ADAPTER_METHODS if not callable(getattr(backend, name, None)))
        if not callable(getattr(backend, "ingest", None)) and not callable(getattr(backend, "ingest_conversation", None)):
            missing.append("ingest")
        if not callable(getattr(backend, "is_ready", None)) and not callable(getattr(backend, "is_namespace_ready", None)):
            missing.append("is_ready")
        if missing:
            raise TypeError(f"External backend {backend_id!r} is missing adapter methods: {', '.join(missing)}")
    return backend


def _normalise_backend_id(value: Any) -> str:
    key = str(value or "").strip().lower()
    if not key or not all(character.isalnum() or character in {"_", "-", "."} for character in key):
        raise ValueError(f"Invalid memory backend id: {value!r}")
    return key
