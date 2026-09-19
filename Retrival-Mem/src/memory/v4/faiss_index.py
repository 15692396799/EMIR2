from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from memory.v4.errors import MemoryNotReadyError


def _faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - depends on installation profile.
        raise RuntimeError(
            "The v4 memory backend requires faiss-cpu; install the project dependencies first."
        ) from exc
    return faiss


@dataclass(frozen=True)
class StagedFaissGeneration:
    namespace: str
    generation: str
    node_count: int


class NamespaceFaissIndex:
    """Persistent immutable IndexFlatIP generations keyed by namespace."""

    def __init__(self, directory: str | Path, read_only: bool = False):
        self.directory = Path(directory)
        self.read_only = read_only
        if not read_only:
            self.directory.mkdir(parents=True, exist_ok=True)
        self._cache: dict[tuple[str, str, int], tuple[object, list[str]]] = {}
        self._lock = threading.RLock()
        self.search_calls = 0

    def stage(
        self,
        namespace: str,
        generation: str,
        node_ids: Iterable[str],
        embeddings: Iterable[Iterable[float]],
        node_metadata: Iterable[dict[str, str]] | None = None,
    ) -> StagedFaissGeneration:
        if self.read_only:
            raise RuntimeError("Cannot persist a FAISS index in read-only mode")
        node_ids = list(node_ids)
        metadata = list(node_metadata or [{} for _ in node_ids])
        if len(metadata) != len(node_ids):
            raise ValueError("FAISS node metadata and node ID counts differ")
        matrix = _normalized_matrix(embeddings)
        if len(node_ids) != matrix.shape[0]:
            raise ValueError("FAISS node ID and embedding counts differ")
        if matrix.shape[0] == 0:
            raise ValueError("Cannot create a FAISS index without embeddings")
        index = _faiss().IndexFlatIP(int(matrix.shape[1]))
        index.add(matrix)
        mapping_bytes = _mapping_bytes(namespace, generation, node_ids, metadata)
        index_path, mapping_path = self._paths(namespace, generation)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_index = index_path.with_suffix(index_path.suffix + ".tmp")
        tmp_mapping = mapping_path.with_suffix(mapping_path.suffix + ".tmp")
        try:
            _faiss().write_index(index, str(tmp_index))
            tmp_mapping.write_bytes(mapping_bytes)
            os.replace(tmp_index, index_path)
            os.replace(tmp_mapping, mapping_path)
        except BaseException:
            for path in (tmp_index, tmp_mapping, index_path, mapping_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        return StagedFaissGeneration(namespace, generation, len(node_ids))

    def search(
        self,
        namespace: str,
        query_embeddings: Iterable[Iterable[float]],
        top_k: int,
        *,
        generation: str,
        node_count: int,
    ) -> list[list[tuple[str, float]]]:
        queries = _normalized_matrix(query_embeddings)
        if queries.shape[0] == 0:
            return []
        with self._lock:
            index, node_ids = self._load(
                namespace, generation, node_count, verify_disk=False
            )
            self.search_calls += 1
            if queries.shape[1] != int(index.d):
                raise ValueError(
                    f"Embedding dimension mismatch: queries={queries.shape[1]}, index={index.d}"
                )
            k = max(1, min(int(top_k), len(node_ids)))
            scores, positions = index.search(queries, k)
        output: list[list[tuple[str, float]]] = []
        for score_row, position_row in zip(scores, positions):
            row = []
            for score, position in zip(score_row, position_row):
                if int(position) >= 0:
                    row.append((node_ids[int(position)], float(score)))
            output.append(row)
        return output

    def validate(
        self,
        namespace: str,
        generation: str,
        node_count: int,
    ) -> bool:
        with self._lock:
            self._load(namespace, generation, node_count, verify_disk=True)
        return True

    def vectors(
        self,
        namespace: str,
        node_ids: Iterable[str],
        *,
        generation: str,
        node_count: int,
    ) -> dict[str, list[float]]:
        """Reconstruct stored (L2-normalized) vectors for the given node ids.

        Used by graph expansion to score neighbor relevance via cosine instead
        of token Jaccard. IndexFlatIP stores raw vectors, so reconstruct is cheap.
        """
        ids = [str(node_id) for node_id in node_ids]
        if not ids:
            return {}
        with self._lock:
            index, id_list = self._load(
                namespace, generation, node_count, verify_disk=False
            )
            position_by_id = {node_id: i for i, node_id in enumerate(id_list)}
            out: dict[str, list[float]] = {}
            for node_id in ids:
                position = position_by_id.get(node_id)
                if position is None:
                    continue
                try:
                    vector = index.reconstruct(int(position))
                except Exception:
                    continue
                out[node_id] = [float(value) for value in vector]
            return out

    def discard(self, namespace: str, generation: str) -> None:
        if self.read_only:
            return
        paths = self._paths(namespace, generation)
        with self._lock:
            for key in [key for key in self._cache if key[:2] == (namespace, generation)]:
                self._cache.pop(key, None)
            for path in paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def cleanup(self, namespace: str, keep_generation: str) -> None:
        if self.read_only:
            return
        keep = set(self._paths(namespace, keep_generation))
        prefix = self._namespace_key(namespace)
        for path in self.directory.glob(f"{prefix}-*"):
            if path not in keep and path.suffix in {".faiss", ".json", ".tmp"}:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        # Remove pre-generation V4 artifacts only after a generation was published.
        for path in (self.directory / f"{prefix}.faiss", self.directory / f"{prefix}.json"):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        with self._lock:
            for key in [key for key in self._cache if key[0] == namespace and key[1] != keep_generation]:
                self._cache.pop(key, None)

    def _load(
        self,
        namespace: str,
        generation: str,
        node_count: int,
        *,
        verify_disk: bool,
    ):
        cache_key = (namespace, generation, node_count)
        if not verify_disk:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        index_path, mapping_path = self._paths(namespace, generation)
        try:
            mapping_bytes = mapping_path.read_bytes()
        except OSError as exc:
            raise MemoryNotReadyError(
                f"V4 namespace {namespace!r} generation {generation!r} has no complete FAISS mapping"
            ) from exc
        try:
            mapping = json.loads(mapping_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise MemoryNotReadyError(
                f"V4 namespace {namespace!r} generation {generation!r} has an invalid FAISS mapping"
            ) from exc
        if mapping.get("namespace") != namespace or mapping.get("generation") != generation:
            raise MemoryNotReadyError(
                f"V4 namespace {namespace!r} generation metadata does not match SQLite state"
            )
        node_ids = [str(value.get("node_id")) for value in mapping.get("nodes", [])]
        if len(node_ids) != node_count:
            raise MemoryNotReadyError(
                f"V4 namespace {namespace!r} generation node count does not match SQLite state"
            )
        try:
            index = _faiss().read_index(str(index_path))
        except Exception as exc:
            python_version = ".".join(str(value) for value in sys.version_info[:3])
            raise MemoryNotReadyError(
                f"V4 namespace {namespace!r} generation {generation!r} has no readable FAISS index "
                f"(python={python_version}, executable={sys.executable}, "
                f"cause={type(exc).__name__}: {exc})"
            ) from exc
        if int(index.ntotal) != node_count:
            raise MemoryNotReadyError(
                f"V4 namespace {namespace!r} FAISS index count does not match SQLite state"
            )
        self._cache[cache_key] = (index, node_ids)
        return index, node_ids

    def _paths(self, namespace: str, generation: str) -> tuple[Path, Path]:
        stem = f"{self._namespace_key(namespace)}-{hashlib.sha256(generation.encode('utf-8')).hexdigest()[:24]}"
        return self.directory / f"{stem}.faiss", self.directory / f"{stem}.json"

    @staticmethod
    def _namespace_key(namespace: str) -> str:
        return hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:24]


def _mapping_bytes(
    namespace: str,
    generation: str,
    node_ids: list[str],
    node_metadata: list[dict[str, str]],
) -> bytes:
    nodes = []
    for node_id, metadata in zip(node_ids, node_metadata):
        nodes.append({
            "node_id": node_id,
            "node_type": str(metadata.get("node_type") or ""),
            "scope_id": str(metadata.get("scope_id") or namespace),
            "chain_id": str(metadata.get("chain_id") or ""),
        })
    return json.dumps(
        {"generation": generation, "namespace": namespace, "nodes": nodes},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _normalized_matrix(values: Iterable[Iterable[float]]) -> np.ndarray:
    rows = list(values)
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    dimensions = []
    for row in rows:
        try:
            dimensions.append(len(row))  # type: ignore[arg-type]
        except TypeError:
            dimensions.append(-1)
    if len(set(dimensions)) != 1:
        counts: dict[int, int] = {}
        for dimension in dimensions:
            counts[dimension] = counts.get(dimension, 0) + 1
        raise ValueError(f"Embeddings must have one dimension, got counts {counts}")
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("FAISS cannot index or search zero-norm embeddings")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)
