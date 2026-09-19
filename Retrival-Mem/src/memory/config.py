from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from run_artifacts import reject_plaintext_secrets


def _int_at_least(value: Any, minimum: int, path: str) -> int:
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{path} must be at least {minimum}")
    return parsed


def _float_at_least(value: Any, minimum: float, path: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{path} must be at least {minimum}")
    return parsed


def _float_range(value: Any, low: float, high: float, path: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not low <= parsed <= high:
        raise ValueError(f"{path} must be between {low} and {high}")
    return parsed


def load_dotenv(path: str | Path = ".env", override: bool = True) -> dict[str, str]:
    env_path = Path(path)
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return values


@dataclass
class ModelConfig:
    provider: str
    model: str
    base_url_env: str | None = None
    api_key_env: str | None = None
    chat_completions_endpoint_env: str | None = None
    embeddings_endpoint_env: str | None = None
    files_endpoint_env: str | None = None
    file_content_endpoint_template_env: str | None = None
    batches_endpoint_env: str | None = None
    batch_status_endpoint_template_env: str | None = None
    batch_cancel_endpoint_template_env: str | None = None
    batch_request_url_env: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        known = {
            "provider",
            "model",
            "base_url_env",
            "api_key_env",
            "chat_completions_endpoint_env",
            "embeddings_endpoint_env",
            "files_endpoint_env",
            "file_content_endpoint_template_env",
            "batches_endpoint_env",
            "batch_status_endpoint_template_env",
            "batch_cancel_endpoint_template_env",
            "batch_request_url_env",
            "base_url",
            "api_key",
            "temperature",
        }
        kwargs = {key: data.get(key) for key in known if key in data}
        extra = {key: value for key, value in data.items() if key not in known}
        return cls(extra=extra, **kwargs)

    @property
    def resolved_base_url(self) -> str | None:
        if self.base_url:
            return self.base_url
        if self.base_url_env:
            return os.environ.get(self.base_url_env)
        return self._default_provider_env_value("BASE_URL")

    @property
    def resolved_chat_completions_endpoint(self) -> str | None:
        return self._resolved_provider_endpoint(self.chat_completions_endpoint_env, "CHAT_COMPLETIONS_ENDPOINT")

    @property
    def resolved_embeddings_endpoint(self) -> str | None:
        return self._resolved_provider_endpoint(self.embeddings_endpoint_env, "EMBEDDINGS_ENDPOINT")

    @property
    def resolved_files_endpoint(self) -> str | None:
        return self._resolved_provider_endpoint(self.files_endpoint_env, "FILES_ENDPOINT")

    @property
    def resolved_file_content_endpoint_template(self) -> str | None:
        return self._resolved_provider_endpoint(
            self.file_content_endpoint_template_env,
            "FILE_CONTENT_ENDPOINT_TEMPLATE",
        )

    @property
    def resolved_batches_endpoint(self) -> str | None:
        return self._resolved_provider_endpoint(self.batches_endpoint_env, "BATCHES_ENDPOINT")

    @property
    def resolved_batch_status_endpoint_template(self) -> str | None:
        return self._resolved_provider_endpoint(
            self.batch_status_endpoint_template_env,
            "BATCH_STATUS_ENDPOINT_TEMPLATE",
        )

    @property
    def resolved_batch_cancel_endpoint_template(self) -> str | None:
        return self._resolved_provider_endpoint(
            self.batch_cancel_endpoint_template_env,
            "BATCH_CANCEL_ENDPOINT_TEMPLATE",
        )

    @property
    def resolved_batch_request_url(self) -> str | None:
        return self._resolved_provider_endpoint(self.batch_request_url_env, "BATCH_REQUEST_URL")

    def _resolved_provider_endpoint(self, explicit_env: str | None, suffix: str) -> str | None:
        if explicit_env:
            return os.environ.get(explicit_env)
        return self._default_provider_env_value(suffix)

    def _default_provider_env_value(self, suffix: str) -> str | None:
        prefix = _provider_env_prefix(self.provider)
        if not prefix:
            return None
        return os.environ.get(f"{prefix}_{suffix}")

    @property
    def default_api_key_env(self) -> str | None:
        prefix = _provider_env_prefix(self.provider)
        if not prefix:
            return None
        return f"{prefix}_API_KEY"

    @property
    def default_base_url_env(self) -> str | None:
        prefix = _provider_env_prefix(self.provider)
        if not prefix:
            return None
        return f"{prefix}_BASE_URL"

    @property
    def default_chat_completions_endpoint_env(self) -> str | None:
        prefix = _provider_env_prefix(self.provider)
        if not prefix:
            return None
        return f"{prefix}_CHAT_COMPLETIONS_ENDPOINT"

    @property
    def default_embeddings_endpoint_env(self) -> str | None:
        prefix = _provider_env_prefix(self.provider)
        if not prefix:
            return None
        return f"{prefix}_EMBEDDINGS_ENDPOINT"

    @property
    def default_files_endpoint_env(self) -> str | None:
        prefix = _provider_env_prefix(self.provider)
        if not prefix:
            return None
        return f"{prefix}_FILES_ENDPOINT"

    @property
    def default_file_content_endpoint_template_env(self) -> str | None:
        prefix = _provider_env_prefix(self.provider)
        if not prefix:
            return None
        return f"{prefix}_FILE_CONTENT_ENDPOINT_TEMPLATE"

    @property
    def default_batches_endpoint_env(self) -> str | None:
        prefix = _provider_env_prefix(self.provider)
        if not prefix:
            return None
        return f"{prefix}_BATCHES_ENDPOINT"

    @property
    def default_batch_status_endpoint_template_env(self) -> str | None:
        prefix = _provider_env_prefix(self.provider)
        if not prefix:
            return None
        return f"{prefix}_BATCH_STATUS_ENDPOINT_TEMPLATE"

    @property
    def default_batch_cancel_endpoint_template_env(self) -> str | None:
        prefix = _provider_env_prefix(self.provider)
        if not prefix:
            return None
        return f"{prefix}_BATCH_CANCEL_ENDPOINT_TEMPLATE"

    @property
    def default_batch_request_url_env(self) -> str | None:
        prefix = _provider_env_prefix(self.provider)
        if not prefix:
            return None
        return f"{prefix}_BATCH_REQUEST_URL"

    @property
    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        default_api_key_env = self.default_api_key_env
        if default_api_key_env:
            return os.environ.get(default_api_key_env)
        return None


def _provider_env_prefix(provider: str) -> str | None:
    mapping = {
        "openai": "OPENAI",
        "dashscope_bailian": "DASHSCOPE",
        "greatrouter": "GREATROUTER",
        "openrouter": "OPENROUTER",
    }
    return mapping.get(provider.lower())


@dataclass
class RelaxedFallbackConfig:
    enabled: bool = False
    min_similarity: float = 0.35
    top_k: int = 32

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RelaxedFallbackConfig":
        data = data or {}
        top_k = _int_at_least(data.get("top_k", 32), 1, "relaxed_fallback.top_k")
        if top_k > 32:
            raise ValueError("relaxed_fallback.top_k must be at most 32")
        return cls(
            enabled=bool(data.get("enabled", False)),
            min_similarity=float(data.get("min_similarity", 0.35)),
            top_k=top_k,
        )


@dataclass
class WindowPlannerConfig:
    mode: str = "hybrid"
    model_name: str = "window_planner"
    token_encoding: str = "cl100k_base"
    min_builder_input_tokens: int = 2048
    target_builder_input_tokens: int = 8192
    max_builder_input_tokens: int = 30000
    max_window_atoms: int = 64
    max_window_chars: int = 24000
    target_window_chars: int = 12000
    max_window_turns: int = 0
    min_window_turns: int = 2
    overlap_turns: int = 2
    boundary_batch_size: int = 8
    boundary_workers: int = 4
    confidence_threshold: float = 0.65
    max_retries: int = 1
    retry_backoff_seconds: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "WindowPlannerConfig":
        data = data or {}
        mode = str(data.get("mode", "hybrid")).lower()
        if mode not in {"rule", "model", "hybrid"}:
            raise ValueError("window_planner.mode must be rule, model, or hybrid")
        token_encoding = str(data.get("token_encoding", "cl100k_base")).strip()
        if not token_encoding:
            raise ValueError("window_planner.token_encoding must not be empty")
        min_tokens = _int_at_least(data.get("min_builder_input_tokens", 2048), 1, "window_planner.min_builder_input_tokens")
        target_tokens = _int_at_least(data.get("target_builder_input_tokens", 8192), 1, "window_planner.target_builder_input_tokens")
        max_tokens = _int_at_least(data.get("max_builder_input_tokens", 30000), 1, "window_planner.max_builder_input_tokens")
        if not min_tokens <= target_tokens < max_tokens <= 30000:
            raise ValueError(
                "window_planner token limits must satisfy "
                "min_builder_input_tokens <= target_builder_input_tokens < "
                "max_builder_input_tokens <= 30000"
            )
        return cls(
            mode=mode,
            model_name=str(data.get("model_name", "window_planner")),
            token_encoding=token_encoding,
            min_builder_input_tokens=min_tokens,
            target_builder_input_tokens=target_tokens,
            max_builder_input_tokens=max_tokens,
            max_window_atoms=_int_at_least(data.get("max_window_atoms", 64), 1, "window_planner.max_window_atoms"),
            max_window_chars=_int_at_least(data.get("max_window_chars", 24000), 1, "window_planner.max_window_chars"),
            target_window_chars=_int_at_least(data.get("target_window_chars", 12000), 1, "window_planner.target_window_chars"),
            max_window_turns=_int_at_least(data.get("max_window_turns", 0), 0, "window_planner.max_window_turns"),
            min_window_turns=_int_at_least(data.get("min_window_turns", 2), 1, "window_planner.min_window_turns"),
            overlap_turns=_int_at_least(data.get("overlap_turns", 2), 0, "window_planner.overlap_turns"),
            boundary_batch_size=_int_at_least(data.get("boundary_batch_size", 8), 1, "window_planner.boundary_batch_size"),
            boundary_workers=_int_at_least(data.get("boundary_workers", 4), 1, "window_planner.boundary_workers"),
            confidence_threshold=_float_range(data.get("confidence_threshold", 0.65), 0.0, 1.0, "window_planner.confidence_threshold"),
            max_retries=_int_at_least(data.get("max_retries", 1), 0, "window_planner.max_retries"),
            retry_backoff_seconds=_float_at_least(data.get("retry_backoff_seconds", 0.5), 0.0, "window_planner.retry_backoff_seconds"),
        )


@dataclass
class ApiRateLimitConfig:
    enabled: bool = True
    max_in_flight: int = 16
    requests_per_minute: int | None = 120
    tokens_per_minute: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ApiRateLimitConfig":
        data = data or {}
        rpm = data.get("requests_per_minute", 120)
        tpm = data.get("tokens_per_minute")
        return cls(
            enabled=bool(data.get("enabled", True)),
            max_in_flight=_int_at_least(data.get("max_in_flight", 16), 1, "api_rate_limit.max_in_flight"),
            requests_per_minute=None if rpm is None else _int_at_least(rpm, 1, "api_rate_limit.requests_per_minute"),
            tokens_per_minute=None if tpm is None else _int_at_least(tpm, 1, "api_rate_limit.tokens_per_minute"),
        )


@dataclass
class BuildParallelismConfig:
    source_workers: int = 3
    namespace_workers_per_source: int = 2
    window_api_workers_per_namespace: int = 8
    publish_workers_per_source: int = 1
    api_rate_limit: ApiRateLimitConfig = field(default_factory=ApiRateLimitConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BuildParallelismConfig":
        data = data or {}
        return cls(
            source_workers=_int_at_least(data.get("source_workers", 3), 1, "build_parallelism.source_workers"),
            namespace_workers_per_source=_int_at_least(data.get("namespace_workers_per_source", 2), 1, "build_parallelism.namespace_workers_per_source"),
            window_api_workers_per_namespace=_int_at_least(data.get("window_api_workers_per_namespace", 8), 1, "build_parallelism.window_api_workers_per_namespace"),
            publish_workers_per_source=_int_at_least(data.get("publish_workers_per_source", 1), 1, "build_parallelism.publish_workers_per_source"),
            api_rate_limit=ApiRateLimitConfig.from_dict(data.get("api_rate_limit")),
        )


@dataclass
class MemoryConfig:
    backend: str = "v4"
    adapter: str | None = None
    memory_extraction_batch_turns: int = 8
    memory_extraction_workers: int = 1
    llm_max_retries: int = 2
    llm_retry_backoff_seconds: float = 1.0
    include_retrieval_trace_in_answer_prompt: bool = False
    answer_context_mode: str = "summary only"
    backends: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryConfig":
        answer_context_mode = str(data.get("answer_context_mode", "summary only")).strip().lower()
        if answer_context_mode not in {"summary only", "content aware"}:
            raise ValueError(
                'memory.answer_context_mode must be "summary only" or "content aware"'
            )
        return cls(
            backend=str(data.get("backend", "v4")).lower(),
            adapter=(str(data["adapter"]).strip() if data.get("adapter") else None),
            memory_extraction_batch_turns=_int_at_least(data.get("memory_extraction_batch_turns", 8), 1, "memory.memory_extraction_batch_turns"),
            memory_extraction_workers=_int_at_least(data.get("memory_extraction_workers", 1), 1, "memory.memory_extraction_workers"),
            llm_max_retries=_int_at_least(data.get("llm_max_retries", 2), 0, "memory.llm_max_retries"),
            llm_retry_backoff_seconds=_float_at_least(data.get("llm_retry_backoff_seconds", 1.0), 0.0, "memory.llm_retry_backoff_seconds"),
            include_retrieval_trace_in_answer_prompt=bool(
                data.get("include_retrieval_trace_in_answer_prompt", False)
            ),
            answer_context_mode=answer_context_mode,
            backends={str(key): dict(value or {}) for key, value in (data.get("backends") or {}).items()},
        )


@dataclass
class EvaluationConfig:
    benchmarks: list[str] = field(default_factory=lambda: ["locomo"])
    benchmark_dir: str = "benchmark"
    output_dir: str = "runs"
    proactive_membench: dict[str, Any] = field(default_factory=dict)
    locomo: dict[str, Any] = field(default_factory=dict)
    protocol: str = "research"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationConfig":
        protocol = str(data.get("protocol") or "research").strip().lower()
        if protocol not in {"research", "canonical_mem0"}:
            raise ValueError("evaluation.protocol must be research or canonical_mem0")
        return cls(
            benchmarks=list(data.get("benchmarks", ["locomo"])),
            benchmark_dir=data.get("benchmark_dir", "benchmark"),
            output_dir=data.get("output_dir", "runs"),
            proactive_membench=dict(data.get("proactive_membench") or {}),
            locomo=dict(data.get("locomo") or {}),
            protocol=protocol,
        )


@dataclass
class AppConfig:
    embedding: ModelConfig
    slm: ModelConfig
    answer_model: ModelConfig
    judge_model: ModelConfig
    memory_builder: ModelConfig
    memory: MemoryConfig
    evaluation: EvaluationConfig
    controller: ModelConfig | None = None
    decomposition_gate: ModelConfig | None = None
    window_planner: ModelConfig | None = None
    adjudication_model: ModelConfig | None = None
    entity_judge: ModelConfig | None = None
    retrieval: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(
    config_path: str | Path = "configs/default.yaml",
    env_path: str | Path = ".env",
) -> AppConfig:
    load_dotenv(env_path, override=True)
    config_data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    reject_plaintext_secrets(config_data)
    _validate_v4_pipeline(config_data)
    models = config_data.get("models", {})
    default_judge_model = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    }
    default_adjudication_model = {
        "provider": "dashscope_bailian",
        "model": "qwen3.7-plus",
        "temperature": 0.0,
    }
    default_entity_judge = {
        "provider": "ollama",
        "model": "llama3.2:1b",
        "temperature": 0.0,
    }
    return AppConfig(
        embedding=ModelConfig.from_dict(models.get("embedding", {})),
        slm=ModelConfig.from_dict(models.get("slm", {})),
        answer_model=ModelConfig.from_dict(models.get("answer_model", {})),
        judge_model=ModelConfig.from_dict(models.get("judge_model") or default_judge_model),
        memory_builder=ModelConfig.from_dict(models.get("memory_builder", {})),
        memory=MemoryConfig.from_dict(config_data.get("memory", {})),
        evaluation=EvaluationConfig.from_dict(config_data.get("evaluation", {})),
        controller=(ModelConfig.from_dict(models["controller"]) if models.get("controller") else None),
        decomposition_gate=(
            ModelConfig.from_dict(models["decomposition_gate"])
            if models.get("decomposition_gate")
            else None
        ),
        window_planner=ModelConfig.from_dict(models.get("window_planner") or models.get("slm", {})),
        adjudication_model=ModelConfig.from_dict(models.get("adjudication_model") or default_adjudication_model),
        entity_judge=ModelConfig.from_dict(models.get("entity_judge") or default_entity_judge),
        retrieval=dict(config_data.get("retrieval") or {}),
        raw=config_data,
    )






def _validate_v4_pipeline(config_data: dict[str, Any]) -> None:
    memory = config_data.get("memory") or {}
    retrieval = config_data.get("retrieval") or {}
    if str(memory.get("backend") or "v4") != "v4":
        raise ValueError("Only memory.backend=v4 is supported")
    if str(retrieval.get("strategy") or "multi_round") != "multi_round":
        raise ValueError("Only retrieval.strategy=multi_round is supported")

def configure_backend_output_paths(
    config: AppConfig, database_path: str, faiss_path: str,
) -> None:
    """Set generated storage paths without backend branching in evaluation code."""
    options = config.memory.backends.get(config.memory.backend)
    if options is not None:
        options["database_path"] = database_path
        options["faiss_path"] = faiss_path
        raw_memory = config.raw.setdefault("memory", {})
        raw_backends = raw_memory.setdefault("backends", {})
        raw_options = raw_backends.setdefault(config.memory.backend, {})
        raw_options["database_path"] = database_path
        raw_options["faiss_path"] = faiss_path
        return
    raise ValueError(
        f"Backend {config.memory.backend!r} must define memory.backends.{config.memory.backend}"
    )


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    return {
        "models": {
            "embedding": _model_to_dict(config.embedding),
            "slm": _model_to_dict(config.slm),
            "answer_model": _model_to_dict(config.answer_model),
            "judge_model": _model_to_dict(config.judge_model),
            "memory_builder": _model_to_dict(config.memory_builder),
            **({"controller": _model_to_dict(config.controller)} if config.controller else {}),
            **(
                {"decomposition_gate": _model_to_dict(config.decomposition_gate)}
                if config.decomposition_gate
                else {}
            ),
            **({"window_planner": _model_to_dict(config.window_planner)} if config.window_planner else {}),
            **({"adjudication_model": _model_to_dict(config.adjudication_model)} if config.adjudication_model else {}),
            **({"entity_judge": _model_to_dict(config.entity_judge)} if config.entity_judge else {}),
        },
        "memory": {
            **{key: value for key, value in vars(config.memory).items() if key != "backends"},
            "backends": dict(config.memory.backends),
        },
        "evaluation": vars(config.evaluation),
        "retrieval": dict(config.retrieval),
    }


def _model_to_dict(config: ModelConfig) -> dict[str, Any]:
    data = {
        "provider": config.provider,
        "model": config.model,
        "base_url_env": config.base_url_env or config.default_base_url_env,
        "api_key_env": config.api_key_env or config.default_api_key_env,
        "chat_completions_endpoint_env": config.chat_completions_endpoint_env
        or config.default_chat_completions_endpoint_env,
        "embeddings_endpoint_env": config.embeddings_endpoint_env or config.default_embeddings_endpoint_env,
        "files_endpoint_env": config.files_endpoint_env or config.default_files_endpoint_env,
        "file_content_endpoint_template_env": config.file_content_endpoint_template_env
        or config.default_file_content_endpoint_template_env,
        "batches_endpoint_env": config.batches_endpoint_env or config.default_batches_endpoint_env,
        "batch_status_endpoint_template_env": config.batch_status_endpoint_template_env
        or config.default_batch_status_endpoint_template_env,
        "batch_cancel_endpoint_template_env": config.batch_cancel_endpoint_template_env
        or config.default_batch_cancel_endpoint_template_env,
        "batch_request_url_env": config.batch_request_url_env or config.default_batch_request_url_env,
        "base_url": config.resolved_base_url,
        "chat_completions_endpoint": config.resolved_chat_completions_endpoint,
        "embeddings_endpoint": config.resolved_embeddings_endpoint,
        "files_endpoint": config.resolved_files_endpoint,
        "file_content_endpoint_template": config.resolved_file_content_endpoint_template,
        "batches_endpoint": config.resolved_batches_endpoint,
        "batch_status_endpoint_template": config.resolved_batch_status_endpoint_template,
        "batch_cancel_endpoint_template": config.resolved_batch_cancel_endpoint_template,
        "batch_request_url": config.resolved_batch_request_url,
        "temperature": config.temperature,
    }
    data.update(config.extra)
    return data
