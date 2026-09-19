"""Worker-local endpoint overrides for evaluation; legacy clients remain the default."""
from contextlib import contextmanager
from contextvars import ContextVar

_routes: ContextVar[dict[str, str] | None] = ContextVar("ollama_routes", default=None)


def ollama_endpoint(kind: str) -> str | None:
    routes = _routes.get()
    return routes.get(kind) if routes else None


@contextmanager
def ollama_routes(routes: dict[str, str]):
    token = _routes.set(routes)
    try:
        yield
    finally:
        _routes.reset(token)
