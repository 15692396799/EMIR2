from __future__ import annotations

import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Mapping, Sequence

from memory.clients import ChatClient, NoopChatClient, parse_json_object
from memory.v4.builder_request import count_builder_input_tokens, get_token_encoding
from memory.v4.config import FailurePolicyConfig
from memory.v4.failure import V4BuildStageError, V4OperationContext, retry_v4_call
from memory.prompt_safety import UNTRUSTED_DATA_INSTRUCTION
from memory.structured_output import (
    messages_with_json_schema,
    object_schema,
)


_USER_ROLES = {"user", "human"}
_RESPONSE_ROLES = {"assistant", "agent", "tool", "function"}
_KNOWN_ROLES = _USER_ROLES | _RESPONSE_ROLES | {"system", "developer"}
_REASON_CODES = {
    "new_topic", "response", "follow_up", "elaboration", "correction",
    "reference", "ambiguous",
}

WINDOW_BOUNDARY_JSON_SCHEMA = object_schema(
    {
        "decision": {"type": "string", "enum": ["split", "continue"]},
        "reason_code": {"type": "string", "enum": sorted(_REASON_CODES)},
        "topic": {"type": "string", "maxLength": 80},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    required=("decision", "reason_code", "topic", "confidence"),
)


@dataclass(frozen=True)
class MemoryWindow:
    window_id: str
    namespace: str
    scope_id: str
    session_id: str
    turn_ids: list[str]
    turns: list[dict[str, Any]]
    order_index: int
    source: str = "conversation"
    planner: str = "rule"
    boundary_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "namespace": self.namespace,
            "scope_id": self.scope_id,
            "session_id": self.session_id,
            "turn_ids": list(self.turn_ids),
            "order_index": self.order_index,
            "source": self.source,
            "planner": self.planner,
            "boundary_metadata": dict(self.boundary_metadata),
        }


@dataclass(frozen=True)
class BoundaryDecision:
    decision: str
    reason_code: str = "ambiguous"
    topic: str = ""
    confidence: float = 0.0
    failed: bool = False
    error: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BoundaryDecision":
        decision = str(value.get("decision") or "continue").strip().lower()
        if decision not in {"split", "continue"}:
            decision = "continue"
        reason_code = str(value.get("reason_code") or "ambiguous").strip().lower()
        if reason_code not in _REASON_CODES:
            reason_code = "ambiguous"
        return cls(
            decision=decision,
            reason_code=reason_code,
            topic=str(value.get("topic") or "")[:80],
            confidence=_float01(value.get("confidence"), 0.0),
        )

    def to_metadata(self, boundary_index: int, token_count: int) -> dict[str, Any]:
        return {
            "boundary_index": boundary_index,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "topic": self.topic,
            "confidence": self.confidence,
            "token_count": token_count,
            "failed": self.failed,
            "error": self.error,
        }


@dataclass(frozen=True)
class _DialogueAtom:
    turns: tuple[dict[str, Any], ...]
    source_start: int
    source_end: int
    fragmented: bool = False


@dataclass
class _Chunk:
    start: int
    end: int
    split_reason: str
    atoms: list[_DialogueAtom]
    overlap_atom_index: int | None = None


class RuleWindowPlanner:
    name = "rule"

    def __init__(self, settings: Any):
        self.token_encoding = str(_setting(settings, "token_encoding", "cl100k_base"))
        self.encoding = get_token_encoding(self.token_encoding)
        self.min_builder_input_tokens = max(
            1, int(_setting(settings, "min_builder_input_tokens", 2048))
        )
        self.target_builder_input_tokens = max(
            self.min_builder_input_tokens,
            int(_setting(settings, "target_builder_input_tokens", 8192)),
        )
        self.max_builder_input_tokens = max(
            self.target_builder_input_tokens + 1,
            int(_setting(settings, "max_builder_input_tokens", 30000)),
        )
        if self.max_builder_input_tokens > 30000:
            raise ValueError("max_builder_input_tokens must be <= 30000")
        self.max_window_atoms = max(1, int(_setting(settings, "max_window_atoms", 64)))
        self.max_window_turns = max(0, int(_setting(settings, "max_window_turns", 0)))
        self.confidence_threshold = max(
            0.0,
            min(1.0, float(_setting(settings, "confidence_threshold", 0.65))),
        )

    def plan(
        self,
        namespace: str,
        scope_id: str,
        turns: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> list[MemoryWindow]:
        request_metadata = dict(metadata or {})
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for source_turn in turns:
            grouped[str(source_turn.get("session_id") or "default")].append(source_turn)

        windows: list[MemoryWindow] = []
        order_index = 0
        for session_id, session_turns in grouped.items():
            atoms = self._fragment_oversized_atoms(
                session_id,
                _dialogue_atoms(session_turns),
                request_metadata,
            )
            if not atoms:
                continue
            full_tokens = self._tokens(session_id, atoms, request_metadata)
            if full_tokens <= self.target_builder_input_tokens and self._within_hard(
                session_id, atoms, request_metadata
            ):
                chunks = [_Chunk(0, len(atoms), "session_end", list(atoms))]
                decisions: dict[int, BoundaryDecision] = {}
            else:
                decisions = self._boundary_decisions(atoms)
                chunks = self._plan_chunks(
                    session_id, atoms, decisions, request_metadata
                )

            for chunk_index, chunk in enumerate(chunks):
                chunk_turns = _atom_turns(chunk.atoms)
                token_count = count_builder_input_tokens(
                    session_id,
                    chunk_turns,
                    request_metadata,
                    self.token_encoding,
                )
                if token_count > self.max_builder_input_tokens:
                    raise ValueError(
                        "planned V4 window exceeds builder input token budget: "
                        f"{token_count} > {self.max_builder_input_tokens}"
                    )
                boundary_metadata = self._chunk_metadata(
                    session_id,
                    chunk_index,
                    chunk,
                    decisions,
                    request_metadata,
                    token_count,
                )
                windows.append(self._window(
                    namespace,
                    scope_id,
                    session_id,
                    chunk_turns,
                    order_index,
                    boundary_metadata,
                ))
                order_index += 1
        return windows

    def _boundary_decisions(
        self, atoms: Sequence[_DialogueAtom]
    ) -> dict[int, BoundaryDecision]:
        return {}

    def _tokens(
        self,
        session_id: str,
        atoms: Sequence[_DialogueAtom],
        metadata: Mapping[str, Any],
    ) -> int:
        return count_builder_input_tokens(
            session_id,
            _atom_turns(atoms),
            metadata,
            self.token_encoding,
        )

    def _within_hard(
        self,
        session_id: str,
        atoms: Sequence[_DialogueAtom],
        metadata: Mapping[str, Any],
    ) -> bool:
        if not atoms or len(atoms) > self.max_window_atoms:
            return False
        if self.max_window_turns and len(_atom_turns(atoms)) > self.max_window_turns:
            return False
        return self._tokens(session_id, atoms, metadata) <= self.max_builder_input_tokens

    def _fragment_oversized_atoms(
        self,
        session_id: str,
        atoms: Sequence[_DialogueAtom],
        metadata: Mapping[str, Any],
    ) -> list[_DialogueAtom]:
        output: list[_DialogueAtom] = []
        for atom in atoms:
            if self._within_hard(session_id, [atom], metadata):
                output.append(atom)
                continue
            output.extend(self._fragment_atom(session_id, atom, metadata))
        return output

    def _fragment_atom(
        self,
        session_id: str,
        atom: _DialogueAtom,
        metadata: Mapping[str, Any],
    ) -> list[_DialogueAtom]:
        expanded_turns: list[dict[str, Any]] = []
        for source_turn in atom.turns:
            one_turn_atom = _DialogueAtom(
                (source_turn,), atom.source_start, atom.source_end, True
            )
            if self._within_hard(session_id, [one_turn_atom], metadata):
                expanded_turns.append(source_turn)
            else:
                expanded_turns.extend(
                    self._fragment_turn(session_id, source_turn, metadata)
                )

        fragments: list[_DialogueAtom] = []
        current: list[dict[str, Any]] = []
        for source_turn in expanded_turns:
            projected = [*current, source_turn]
            projected_atom = _DialogueAtom(
                tuple(projected), atom.source_start, atom.source_end, True
            )
            if current and not self._within_hard(
                session_id, [projected_atom], metadata
            ):
                fragments.append(_DialogueAtom(
                    tuple(current), atom.source_start, atom.source_end, True
                ))
                current = [source_turn]
            else:
                current = projected
        if current:
            fragments.append(_DialogueAtom(
                tuple(current), atom.source_start, atom.source_end, True
            ))
        if not all(self._within_hard(session_id, [item], metadata) for item in fragments):
            raise ValueError("unable to fragment oversized dialogue atom below hard token limit")
        return fragments

    def _fragment_turn(
        self,
        session_id: str,
        source_turn: dict[str, Any],
        metadata: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        content_tokens = self.encoding.encode(str(source_turn.get("content") or ""))
        if not content_tokens:
            raise ValueError(
                "builder request metadata and turn fields exceed the hard token limit"
            )
        token_ranges: list[tuple[int, int]] = []
        start = 0
        fragment_count_hint = len(content_tokens)
        while start < len(content_tokens):
            low = start + 1
            high = len(content_tokens)
            best: int | None = None
            while low <= high:
                middle = (low + high) // 2
                decoded = _decode_token_range(self.encoding, content_tokens, start, middle)
                if decoded is None:
                    high = middle - 1
                    continue
                candidate = _fragment_copy(
                    source_turn,
                    decoded,
                    len(token_ranges),
                    fragment_count_hint,
                )
                candidate_atom = _DialogueAtom((candidate,), 0, 1, True)
                if self._within_hard(session_id, [candidate_atom], metadata):
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best is None:
                raise ValueError(
                    "builder request overhead leaves no room for one content token"
                )
            token_ranges.append((start, best))
            start = best

        fragments = []
        for fragment_index, (start, end) in enumerate(token_ranges):
            decoded = _decode_token_range(self.encoding, content_tokens, start, end)
            if decoded is None:
                raise ValueError("fragment token boundary is not valid UTF-8")
            fragments.append(_fragment_copy(
                source_turn,
                decoded,
                fragment_index,
                len(token_ranges),
            ))
        return fragments

    def _plan_chunks(
        self,
        session_id: str,
        atoms: Sequence[_DialogueAtom],
        decisions: Mapping[int, BoundaryDecision],
        metadata: Mapping[str, Any],
    ) -> list[_Chunk]:
        chunks: list[_Chunk] = []
        start = 0
        while start < len(atoms):
            remainder = atoms[start:]
            remainder_tokens = self._tokens(session_id, remainder, metadata)
            if (
                remainder_tokens <= self.target_builder_input_tokens
                and self._within_hard(session_id, remainder, metadata)
            ):
                chunks.append(_Chunk(start, len(atoms), "session_end", list(remainder)))
                break

            legal: list[tuple[int, int]] = []
            for end in range(start + 1, len(atoms) + 1):
                candidate = atoms[start:end]
                if self._within_hard(session_id, candidate, metadata):
                    legal.append((end, self._tokens(session_id, candidate, metadata)))
                    continue
                break
            if not legal:
                raise ValueError("single dialogue atom exceeds the hard builder token limit")

            eligible = [item for item in legal if item[1] >= self.min_builder_input_tokens]
            choices = eligible or legal
            confirmed = [
                item for item in choices
                if item[0] < len(atoms)
                and (decision := decisions.get(item[0])) is not None
                and decision.decision == "split"
                and decision.confidence >= self.confidence_threshold
            ]
            if confirmed:
                end, _ = min(
                    confirmed,
                    key=lambda item: (
                        abs(item[1] - self.target_builder_input_tokens),
                        -decisions[item[0]].confidence,
                        item[0],
                    ),
                )
                split_reason = "topic_confirmed"
            else:
                end, _ = min(
                    choices,
                    key=lambda item: (
                        abs(item[1] - self.target_builder_input_tokens), item[0]
                    ),
                )
                split_reason = "target_fallback"

            if end < len(atoms):
                projected = atoms[start:end + 1]
                projected_tokens = self._tokens(session_id, projected, metadata)
                if projected_tokens > self.max_builder_input_tokens:
                    split_reason = "hard_limit"
                elif len(projected) > self.max_window_atoms:
                    split_reason = "max_window_atoms"
                elif (
                    self.max_window_turns
                    and len(_atom_turns(projected)) > self.max_window_turns
                ):
                    split_reason = "max_window_turns"
            else:
                split_reason = "session_end"
            chunks.append(_Chunk(start, end, split_reason, list(atoms[start:end])))
            start = end

        self._rebalance_short_tail(session_id, chunks, metadata)
        self._add_hard_limit_overlap(session_id, chunks, metadata)
        return chunks

    def _rebalance_short_tail(
        self,
        session_id: str,
        chunks: list[_Chunk],
        metadata: Mapping[str, Any],
    ) -> None:
        if len(chunks) < 2:
            return
        previous = chunks[-2]
        tail = chunks[-1]
        if self._tokens(session_id, tail.atoms, metadata) >= self.min_builder_input_tokens:
            return
        merged = [*previous.atoms, *tail.atoms]
        if self._within_hard(session_id, merged, metadata):
            previous.end = tail.end
            previous.atoms = merged
            previous.split_reason = "tail_merged"
            chunks.pop()
            return
        while len(previous.atoms) > 1:
            moved = previous.atoms[-1]
            candidate_previous = previous.atoms[:-1]
            candidate_tail = [moved, *tail.atoms]
            if not self._within_hard(session_id, candidate_tail, metadata):
                break
            if (
                self._tokens(session_id, candidate_previous, metadata)
                < self.min_builder_input_tokens
            ):
                break
            previous.atoms = candidate_previous
            previous.end -= 1
            tail.atoms = candidate_tail
            tail.start -= 1
            tail.split_reason = "tail_rebalanced"
            if self._tokens(session_id, tail.atoms, metadata) >= self.min_builder_input_tokens:
                break

    def _add_hard_limit_overlap(
        self,
        session_id: str,
        chunks: list[_Chunk],
        metadata: Mapping[str, Any],
    ) -> None:
        for index in range(len(chunks) - 1):
            current = chunks[index]
            following = chunks[index + 1]
            if current.split_reason != "hard_limit" or not current.atoms:
                continue
            overlap = current.atoms[-1]
            projected = [overlap, *following.atoms]
            if self._within_hard(session_id, projected, metadata):
                following.atoms = projected
                following.overlap_atom_index = current.end - 1

    def _chunk_metadata(
        self,
        session_id: str,
        chunk_index: int,
        chunk: _Chunk,
        decisions: Mapping[int, BoundaryDecision],
        metadata: Mapping[str, Any],
        token_count: int,
    ) -> dict[str, Any]:
        evaluated = []
        decision_start = max(1, chunk.start + 1)
        for boundary_index in range(decision_start, chunk.end + 1):
            decision = decisions.get(boundary_index)
            if decision is None:
                continue
            prefix = chunk.atoms[: max(0, boundary_index - chunk.start)]
            evaluated.append(decision.to_metadata(
                boundary_index,
                self._tokens(session_id, prefix, metadata),
            ))
        fragment_metadata = [
            dict(turn["fragment_metadata"])
            for turn in _atom_turns(chunk.atoms)
            if isinstance(turn.get("fragment_metadata"), dict)
        ]
        selected = decisions.get(chunk.end)
        return {
            "chunk_index": chunk_index,
            "builder_input_tokens": token_count,
            "token_encoding": self.token_encoding,
            "atom_range": {"start": chunk.start, "end_exclusive": chunk.end},
            "atom_count": len(chunk.atoms),
            "turn_count": len(_atom_turns(chunk.atoms)),
            "forced_split_reason": chunk.split_reason,
            "boundary": (
                selected.to_metadata(chunk.end, token_count) if selected else None
            ),
            "evaluated_boundaries": evaluated,
            "fragment_metadata": fragment_metadata,
            "overlap_source": (
                {
                    "atom_index": chunk.overlap_atom_index,
                    "turn_ids": [
                        str(turn.get("turn_id")) for turn in chunk.atoms[0].turns
                    ],
                }
                if chunk.overlap_atom_index is not None
                else None
            ),
        }

    def _window(
        self,
        namespace: str,
        scope_id: str,
        session_id: str,
        turns: list[dict[str, Any]],
        order_index: int,
        boundary_metadata: dict[str, Any],
    ) -> MemoryWindow:
        turn_identity = []
        for turn in turns:
            fragment = turn.get("fragment_metadata")
            turn_identity.append({
                "turn_id": str(turn["turn_id"]),
                "fragment": fragment if isinstance(fragment, dict) else None,
                "content_sha256": sha256(
                    str(turn.get("content") or "").encode("utf-8")
                ).hexdigest(),
            })
        digest = sha256(json.dumps(
            [namespace, scope_id, session_id, order_index, turn_identity],
            sort_keys=True,
        ).encode("utf-8")).hexdigest()[:16]
        return MemoryWindow(
            window_id=f"{scope_id}:{digest}",
            namespace=namespace,
            scope_id=scope_id,
            session_id=session_id,
            turn_ids=[str(turn["turn_id"]) for turn in turns],
            turns=turns,
            order_index=order_index,
            source="conversation",
            planner=self.name,
            boundary_metadata=boundary_metadata,
        )


class ModelWindowPlanner(RuleWindowPlanner):
    name = "model"

    def __init__(
        self,
        settings: Any,
        client: ChatClient | None,
        *,
        failure_policy: FailurePolicyConfig | None = None,
        model_identity: tuple[str | None, str | None] | None = None,
    ):
        super().__init__(settings)
        self.client = client
        self.boundary_batch_size = max(
            1, int(_setting(settings, "boundary_batch_size", 8))
        )
        self.boundary_workers = max(
            1, int(_setting(settings, "boundary_workers", 4))
        )
        self.failure_policy = failure_policy or FailurePolicyConfig()
        self.model_identity = model_identity or (None, None)

    def _boundary_decisions(
        self, atoms: Sequence[_DialogueAtom]
    ) -> dict[int, BoundaryDecision]:
        if self.client is None or isinstance(self.client, NoopChatClient):
            raise ValueError(
                "V4 model window planning requires a configured non-noop client; "
                "use mode=rule for the explicit rule baseline"
            )
        decisions: dict[int, BoundaryDecision] = {}
        boundary_indexes = list(range(1, len(atoms)))
        for batch_start in range(0, len(boundary_indexes), self.boundary_batch_size):
            batch = boundary_indexes[batch_start:batch_start + self.boundary_batch_size]
            if self.boundary_workers > 1 and len(batch) > 1:
                with ThreadPoolExecutor(
                    max_workers=min(self.boundary_workers, len(batch))
                ) as executor:
                    results = list(executor.map(
                        lambda index: self._decide_boundary(atoms, index), batch
                    ))
            else:
                results = [self._decide_boundary(atoms, index) for index in batch]
            decisions.update(zip(batch, results))
        return decisions

    def _decide_boundary(
        self,
        atoms: Sequence[_DialogueAtom],
        boundary_index: int,
    ) -> BoundaryDecision:
        payload = {
            "task": "memory_window_boundary",
            "boundary_index": boundary_index,
            "instructions": [
                "Return split only for an independent new topic, task, event, or user goal.",
                "Answers, follow-ups, clarifications, corrections, examples, implementation details, causal continuation, and referential continuation must continue.",
                "If uncertain, return continue.",
                "Confidence calibration: continuation is <0.65; a defensible split is 0.65–0.89; only an obviously independent topic is >=0.90.",
                "Corrections and follow-ups default to continue even when wording or named entities change.",
                "Do not judge window length or token budgets.",
                UNTRUSTED_DATA_INSTRUCTION,
            ],
            "previous_atom": [_public_turn(turn) for turn in atoms[boundary_index - 1].turns],
            "current_atom": [_public_turn(turn) for turn in atoms[boundary_index].turns],
            "recent_context": [
                [_public_turn(turn) for turn in atom.turns]
                for index, atom in enumerate(atoms)
                if index not in {boundary_index - 1, boundary_index}
                and max(0, boundary_index - 2) <= index <= min(
                    len(atoms) - 1, boundary_index + 1
                )
            ],
        }
        def invoke() -> BoundaryDecision:
            assert self.client is not None
            raw = self.client.chat(
                messages_with_json_schema(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Return only valid JSON for a conservative local "
                                "topic-boundary classification. When uncertain, "
                                "choose continue. "
                                + UNTRUSTED_DATA_INSTRUCTION
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(payload, ensure_ascii=False),
                        },
                    ],
                    WINDOW_BOUNDARY_JSON_SCHEMA,
                ),
                json_mode=True,
                json_schema=WINDOW_BOUNDARY_JSON_SCHEMA,
            )
            data = parse_json_object(raw)
            if str(data.get("decision") or "").strip().lower() not in {
                "split", "continue"
            }:
                raise ValueError("window planner response has no valid decision")
            if "confidence" not in data:
                raise ValueError("window planner response has no confidence")
            return BoundaryDecision.from_mapping(data)

        provider, model = self.model_identity
        return retry_v4_call(
            invoke,
            policy=self.failure_policy,
            context=V4OperationContext(
                stage="window_plan",
                unit_id=f"boundary-{boundary_index}",
                provider=provider,
                model=model,
                checkpoint_key=f"window_plan:boundary-{boundary_index}",
            ),
            error_type=V4BuildStageError,
            validation_errors=(ValueError, TypeError),
        )


def make_window_planner(
    settings: Any,
    client: ChatClient | None = None,
    *,
    failure_policy: FailurePolicyConfig | None = None,
    model_identity: tuple[str | None, str | None] | None = None,
) -> RuleWindowPlanner:
    mode = str(_setting(settings, "mode", "model")).lower()
    if mode == "rule":
        return RuleWindowPlanner(settings)
    if mode == "model":
        return ModelWindowPlanner(
            settings,
            client,
            failure_policy=failure_policy,
            model_identity=model_identity,
        )
    raise ValueError("memory.v4.window_planner.mode must be rule or model")


def _dialogue_atoms(turns: Sequence[dict[str, Any]]) -> list[_DialogueAtom]:
    if not turns:
        return []
    roles = {_role(turn) for turn in turns}
    known_dialogue = bool(roles & _USER_ROLES) and roles <= _KNOWN_ROLES
    atoms: list[_DialogueAtom] = []
    if known_dialogue:
        current: list[dict[str, Any]] = []
        current_start = 0
        for index, turn in enumerate(turns):
            role = _role(turn)
            if role in _USER_ROLES and current:
                atoms.append(_DialogueAtom(tuple(current), current_start, index))
                current = []
                current_start = index
            elif not current:
                current_start = index
            current.append(turn)
        if current:
            atoms.append(_DialogueAtom(tuple(current), current_start, len(turns)))
        return atoms

    for start in range(0, len(turns), 2):
        atoms.append(_DialogueAtom(
            tuple(turns[start:start + 2]), start, min(start + 2, len(turns))
        ))
    return atoms


def _atom_turns(atoms: Sequence[_DialogueAtom]) -> list[dict[str, Any]]:
    return [turn for atom in atoms for turn in atom.turns]


def _fragment_copy(
    source_turn: Mapping[str, Any],
    content: str,
    fragment_index: int,
    fragment_count: int,
) -> dict[str, Any]:
    copied = dict(source_turn)
    copied["content"] = content
    copied["fragment_metadata"] = {
        "source_turn_id": str(source_turn.get("turn_id")),
        "fragment_index": fragment_index,
        "fragment_count": fragment_count,
    }
    return copied


def _decode_token_range(encoding, tokens: Sequence[int], start: int, end: int) -> str | None:
    try:
        raw = b"".join(encoding.decode_single_token_bytes(token) for token in tokens[start:end])
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _setting(settings: Any, key: str, default: Any) -> Any:
    if isinstance(settings, dict):
        return settings.get(key, default)
    return getattr(settings, key, default)


def _role(turn: Mapping[str, Any]) -> str:
    return str(turn.get("role") or "").strip().lower()


def _public_turn(turn: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": turn.get("turn_id"),
        "role": turn.get("role"),
        "speaker": turn.get("speaker"),
        "content": turn.get("content"),
        "timestamp": turn.get("timestamp"),
    }


def _float01(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _canonical_topic(text: str) -> str:
    tokens = set(re.findall(r"[a-z0-9']+", text.casefold()))
    rules = {
        "travel": ("travel", "trip", "journey", "beijing", "shanghai"),
        "health": ("health", "doctor", "hospital", "pain", "allergy"),
        "food_preference": ("food", "eat", "spicy", "restaurant"),
        "work": ("work", "job", "company", "project"),
        "relationship": ("friend", "family", "marry", "relationship"),
        "education": ("school", "university", "study", "degree"),
        "long_term_plan": ("plan", "goal", "will", "intend"),
    }
    for topic, markers in rules.items():
        if tokens.intersection(markers):
            return topic
    return "general"
