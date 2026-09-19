from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


_SECRET_KEYS = {"api_key", "apikey", "token", "access_token", "secret", "authorization"}
_SECRET_SUFFIXES = ("_api_key", "_secret", "_token")


def reject_plaintext_secrets(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            child_path = f"{path}.{key}"
            if normalized.endswith("_env"):
                continue
            secret_key = normalized in _SECRET_KEYS or normalized.endswith(_SECRET_SUFFIXES)
            if secret_key and child not in (None, "", False):
                raise ValueError(
                    f"Refusing plaintext credential at {child_path}; reference an environment variable with *_env"
                )
            reject_plaintext_secrets(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_plaintext_secrets(child, f"{path}[{index}]")


def snapshot_run_configs(
    run_dir: str | Path, resolved: dict[str, Any], *,
    config_path: str | Path | None = None,
    input_config: dict[str, Any] | None = None,
    resume: bool = False, run_kind: str,
) -> dict[str, Any]:
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    input_bytes = _input_bytes(config_path, input_config)
    reject_plaintext_secrets(yaml.safe_load(input_bytes.decode("utf-8")) or {})
    reject_plaintext_secrets(resolved)
    manifest_path = directory / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if resume and manifest_path.exists() else {}
    )
    manifest = {key: value for key, value in manifest.items()
                if not key.endswith("_sha256") and key != "resume_config_drift"}
    manifest.setdefault("created_at", _now())
    manifest.update(run_kind=run_kind, python=sys.version.split()[0], platform=platform.platform())
    if resume:
        manifest["resumed_at"] = _now()
    (directory / "config.input.yaml").write_bytes(input_bytes)
    (directory / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest

def update_manifest(run_dir: str | Path, **values: Any) -> None:
    path = Path(run_dir) / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest.update(values)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _input_bytes(path: str | Path | None, value: dict[str, Any] | None) -> bytes:
    if path is not None:
        source = Path(path)
        if source.name == ".env":
            raise ValueError(".env files must never be copied into run artifacts")
        return source.read_bytes()
    return yaml.safe_dump(value or {}, sort_keys=False, allow_unicode=True).encode("utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
