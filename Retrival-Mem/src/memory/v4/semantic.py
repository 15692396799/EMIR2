from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

from memory.clients import ChatClient
from memory.v4.config import FailurePolicyConfig
from memory.v4.failure import (
    V4BuildStageError,
    V4OperationContext,
    retry_v4_call,
)
from memory.prompt_safety import UNTRUSTED_DATA_INSTRUCTION
from memory.structured_output import (
    array_schema,
    messages_with_json_schema,
    object_schema,
    string_array,
)


DECISIONS = {"reinforce", "extend", "revise", "uncertain"}
OPERATIONS = {"reinforce", "add", "replace", "retract"}
SEMANTIC_REDUCER_PROMPT_VERSION = "semantic-reducer-v4-prompt-v4"
SEMANTIC_REDUCER_SYSTEM_PROMPT_V1 = (
    "You reconcile a complete semantic fact snapshot. "
    + UNTRUSTED_DATA_INSTRUCTION
    + " Return JSON with status, operations, full_summary, and reason. "
    "status must be apply or uncertain. Use apply when evidence supports "
    "committing the operations. Use uncertain when evidence is conflicting "
    "or insufficient; uncertain operations remain pending and do not change "
    "canonical state. Do not return a decision field; the system derives it "
    "from operations. "
    'Each operations item must use the field "operation" for its '
    "operation name. Operations are reinforce, add, replace, or retract. "
    "Use reinforce to support an existing fact, add to create a new fact, "
    "replace to update an existing fact, and retract to remove an existing fact. "
    "An initial state accepts add operations only. add must use fact_key=null "
    "and provide subject, dimension, aspect, value, and supplied "
    "evidence_event_refs. Other operations must use only known fact keys. "
    "Cite only supplied local event refs. For apply outputs containing add, "
    "replace, or retract, full_summary must be a complete non-empty "
    "natural-language canonical summary with no internal IDs. A "
    "reinforce-only apply does not rewrite the current summary."
)
SEMANTIC_REDUCER_SYSTEM_PROMPT_V2 = (
    "You reconcile a complete semantic fact snapshot. "
    + UNTRUSTED_DATA_INSTRUCTION
    + " Return JSON with status, operations, full_summary, and reason. "
    "status must be apply or uncertain. operations MUST be a non-empty array "
    "for both apply and uncertain. Never return an empty operations array. "
    "Use apply when evidence supports committing the operations. Use uncertain "
    "when evidence is conflicting or insufficient. For uncertain, still provide "
    "the exact candidate operation that would be applied if the uncertainty were "
    "resolved; it remains pending and does not change canonical state. Evidence "
    "insufficiency is not permission to omit the candidate operation. Do not "
    "return a decision field; the system derives it from status and operations. "
    'Each operations item must use the field "operation" for its operation name. '
    "Operations are reinforce, add, replace, or retract. Use reinforce to support "
    "an existing fact, add to create a new fact, replace to update an existing "
    "fact, and retract to remove an existing fact. An initial state accepts add "
    "operations only. add must use fact_key=null and provide subject, dimension, "
    "aspect, value, and supplied evidence_event_refs. Other operations must use "
    "only known fact keys. Cite only supplied local event refs. For apply outputs "
    "containing add, replace, or retract, full_summary must be a complete non-empty "
    "natural-language canonical summary with no internal IDs. A reinforce-only "
    "apply does not rewrite the current summary. For uncertain, preserve the "
    "active canonical summary in full_summary, or use an empty string when no "
    "active state exists."
)
SEMANTIC_REDUCER_SYSTEM_PROMPT_V4 = SEMANTIC_REDUCER_SYSTEM_PROMPT_V2 + (
    " Fact identity is determined by normalized subject + dimension + aspect; "
    "value is NOT part of the fact key. Compare this identity with fact_snapshot "
    "before proposing add. Never add an identity that already exists, and never "
    "invent a different aspect merely to bypass a collision. For genuinely "
    "multi-valued attributes preserve all still-supported "
    "members: use replace with the known fact_key and the complete deduplicated "
    "JSON array of old and new values. Repeated evidence for a member already present "
    "uses reinforce, not add or a duplicate array entry. Remove a member only when "
    "evidence explicitly supports its removal; replace with the remaining members, "
    "or retract if none remain. Do not merge mutually exclusive values of a "
    "single-valued attribute into an array: use supported replacement or uncertain. "
    "For an initial multi-valued fact, emit one add with the complete collection, "
    "not multiple adds for the same identity. Keep full_summary consistent with "
    "the complete resulting snapshot, including unaffected facts."
)
_INTERNAL_TOKEN = re.compile(
    r"\b(?:node|event|state|chain|scope|fact|local)[_-][A-Za-z0-9-]+\b",
    re.IGNORECASE,
)
_T = TypeVar("_T")

_SEMANTIC_OPERATION_COMMON = {
    "subject": {"type": "string"},
    "dimension": {"type": "string"},
    "aspect": {"type": "string"},
    "value": {},
    "confidence": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
    },
    "evidence_event_refs": string_array(),
}

_SEMANTIC_ADD_SCHEMA = object_schema(
    {
        **_SEMANTIC_OPERATION_COMMON,
        "operation": {"type": "string", "enum": ["add"]},
        "fact_key": {"type": "null"},
    },
    required=(
        "operation",
        "fact_key",
        "subject",
        "dimension",
        "aspect",
        "value",
        "evidence_event_refs",
    ),
)

_SEMANTIC_REPLACE_SCHEMA = object_schema(
    {
        **_SEMANTIC_OPERATION_COMMON,
        "operation": {"type": "string", "enum": ["replace"]},
        "fact_key": {"type": "string", "minLength": 1},
    },
    required=("operation", "fact_key", "value", "evidence_event_refs"),
)

_SEMANTIC_REFERENCE_SCHEMA = object_schema(
    {
        "operation": {
            "type": "string",
            "enum": ["reinforce", "retract"],
        },
        "fact_key": {"type": "string", "minLength": 1},
        "evidence_event_refs": string_array(),
    },
    required=("operation", "fact_key", "evidence_event_refs"),
)

SEMANTIC_REDUCER_JSON_SCHEMA = object_schema(
    {
        "status": {"type": "string", "enum": ["apply", "uncertain"]},
        "operations": array_schema(
            {
                "oneOf": [
                    _SEMANTIC_ADD_SCHEMA,
                    _SEMANTIC_REPLACE_SCHEMA,
                    _SEMANTIC_REFERENCE_SCHEMA,
                ]
            },
            min_items=1,
        ),
        "full_summary": {"type": "string"},
        "reason": {"type": "string"},
    },
    required=("status", "operations", "full_summary", "reason"),
)


class SemanticValidationError(ValueError):
    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.details = dict(details or {})


@dataclass
class SemanticFactRecord:
    key: str
    subject: str
    dimension: str
    aspect: str
    value: Any
    valid_from: str | None
    valid_to: str | None = None
    confidence: float = 0.9
    evidence_event_refs: list[str] = field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    history_origin: str | None = None

    def clone(self) -> "SemanticFactRecord":
        return copy.deepcopy(self)


@dataclass
class SemanticEpoch:
    id: str
    chain_id: str
    epoch: int
    summary: str
    facts: dict[str, SemanticFactRecord]
    valid_from: str | None
    valid_to: str | None = None
    is_chain_head: bool = False
    changed_keys: tuple[str, ...] = ()


@dataclass
class PendingConflict:
    chain_id: str
    event_refs: tuple[str, ...]
    candidate_operations: tuple[dict[str, Any], ...]
    reason: str
    boundary_time: str | None


@dataclass
class SemanticTransition:
    decision: str
    state: SemanticEpoch | None
    previous_state: SemanticEpoch | None = None
    changed_keys: tuple[str, ...] = ()
    pending: PendingConflict | None = None
    created_state: bool = False


def fact_key(subject: str, dimension: str, aspect: str) -> str:
    identity = "\u241f".join(_normalize(value) for value in (subject, dimension, aspect))
    return "fact_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def revision_state_id(
    participant_scope: str,
    chain_id: str,
    boundary_time: str | None,
    trigger_event_refs: Sequence[str],
) -> str:
    value = json.dumps(
        [participant_scope, chain_id, boundary_time, sorted(set(trigger_event_refs))],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "state_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


class SemanticEpochMachine:
    """Validate reducer output and apply one deterministic semantic transition."""

    def apply(
        self,
        *,
        participant_scope: str,
        chain_id: str,
        current: SemanticEpoch | None,
        reducer_output: Mapping[str, Any],
        boundary_time: str | None,
        trigger_event_refs: Sequence[str],
    ) -> SemanticTransition:
        decision = str(reducer_output.get("decision") or "").lower()
        if decision not in DECISIONS:
            raise SemanticValidationError(f"unknown semantic decision: {decision!r}")
        operations = reducer_output.get("operations")
        if not isinstance(operations, list) or not operations:
            raise SemanticValidationError("semantic reducer operations must be a non-empty list")
        known = current.facts if current else {}
        normalized = [self._operation(value, known) for value in operations]
        allowed_refs = set(trigger_event_refs)
        for operation in normalized:
            unknown_refs = set(operation["evidence_event_refs"]) - allowed_refs
            if unknown_refs:
                raise SemanticValidationError(
                    "semantic operation references evidence outside trigger_event_refs"
                )
        self._validate_decision_operations(decision, normalized, current)

        if decision == "uncertain":
            return SemanticTransition(
                decision=decision,
                state=current,
                pending=PendingConflict(
                    chain_id,
                    tuple(_stable(trigger_event_refs)),
                    tuple(copy.deepcopy(normalized)),
                    str(reducer_output.get("reason") or "").strip(),
                    boundary_time,
                ),
            )

        proposed = {key: value.clone() for key, value in known.items()}
        changed: list[str] = []
        for operation in normalized:
            operation_name = operation["operation"]
            key = operation["fact_key"]
            evidence = _stable(operation["evidence_event_refs"] or trigger_event_refs)
            if operation_name == "reinforce":
                fact = proposed[key]
                fact.evidence_event_refs = _stable([*fact.evidence_event_refs, *evidence])
                fact.confidence = min(1.0, 1.0 - (1.0 - fact.confidence) * 0.5)
                fact.updated_at = boundary_time or fact.updated_at
                continue
            changed.append(key)
            if operation_name == "retract":
                proposed.pop(key)
                continue
            if operation_name == "replace":
                prior = proposed[key]
                proposed[key] = SemanticFactRecord(
                    key=key,
                    subject=str(operation.get("subject") or prior.subject),
                    dimension=str(operation.get("dimension") or prior.dimension),
                    aspect=str(operation.get("aspect") or prior.aspect),
                    value=operation["value"],
                    valid_from=boundary_time,
                    confidence=float(operation.get("confidence", prior.confidence)),
                    evidence_event_refs=evidence,
                    created_at=boundary_time,
                    updated_at=boundary_time,
                    history_origin=prior.key,
                )
                continue
            proposed[key] = SemanticFactRecord(
                key=key,
                subject=operation["subject"],
                dimension=operation["dimension"],
                aspect=operation["aspect"],
                value=operation["value"],
                valid_from=boundary_time,
                confidence=float(operation.get("confidence", 0.9)),
                evidence_event_refs=evidence,
                created_at=boundary_time,
                updated_at=boundary_time,
                history_origin="add",
            )

        summary_value = reducer_output.get("full_summary")
        summary = summary_value if isinstance(summary_value, str) else ""
        if decision == "reinforce":
            if current is None:
                raise SemanticValidationError("reinforce requires an active state")
            summary = current.summary
        else:
            self._validate_summary(summary, proposed, trigger_event_refs)

        if current is None:
            state = SemanticEpoch(
                id=revision_state_id(participant_scope, chain_id, boundary_time, trigger_event_refs),
                chain_id=chain_id,
                epoch=0,
                summary=summary,
                facts=proposed,
                valid_from=boundary_time,
                is_chain_head=True,
                changed_keys=tuple(_stable(changed)),
            )
            return SemanticTransition(decision="extend", state=state, changed_keys=tuple(changed), created_state=True)
        if decision == "reinforce":
            current.facts = proposed
            return SemanticTransition(decision, current, changed_keys=(), created_state=False)
        if decision == "extend":
            current.facts = proposed
            current.summary = summary
            return SemanticTransition(decision, current, changed_keys=tuple(_stable(changed)), created_state=False)

        previous = current
        state = SemanticEpoch(
            id=revision_state_id(participant_scope, chain_id, boundary_time, trigger_event_refs),
            chain_id=chain_id,
            epoch=current.epoch + 1,
            summary=summary,
            facts=proposed,
            valid_from=boundary_time,
            is_chain_head=False,
            changed_keys=tuple(_stable(changed)),
        )
        previous.valid_to = boundary_time
        return SemanticTransition(
            decision, state, previous_state=previous,
            changed_keys=tuple(_stable(changed)), created_state=True,
        )

    def _operation(self, value: Any, known: Mapping[str, SemanticFactRecord]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise SemanticValidationError("semantic operation must be an object")
        operation = dict(value)
        if "op" in operation:
            raise SemanticValidationError(
                "semantic operation must use 'operation', not 'op'"
            )
        operation_name = str(operation.get("operation") or "").lower()
        if operation_name not in OPERATIONS:
            raise SemanticValidationError(
                f"unknown semantic operation: {operation_name!r}"
            )
        supplied = operation.get("fact_key")
        if operation_name == "add":
            if supplied not in {None, ""}:
                raise SemanticValidationError("add must not supply fact_key")
            for name in ("subject", "dimension", "aspect"):
                if not str(operation.get(name) or "").strip():
                    raise SemanticValidationError(f"add requires {name}")
            key = fact_key(operation["subject"], operation["dimension"], operation["aspect"])
            if key in known:
                raise SemanticValidationError(
                    "add resolves to an existing fact key",
                    details={
                        "conflicting_fact_key": key,
                        "identity": {name: getattr(known[key], name)
                                     for name in ("subject", "dimension", "aspect")},
                        "existing_value": copy.deepcopy(known[key].value),
                        "candidate_value": copy.deepcopy(operation.get("value")),
                        "repair_instruction": (
                            "Use this known fact key. Reinforce if already supported; "
                            "for a coexisting member of a multi-valued attribute, "
                            "replace with the complete deduplicated collection, "
                            "preserving existing members unless evidence removes them. "
                            "Do not blindly overwrite or change identity to evade validation."
                        ),
                    },
                )
        else:
            key = str(supplied or "")
            if key not in known:
                raise SemanticValidationError(
                    f"{operation_name} references unknown fact key"
                )
        operation["operation"] = operation_name
        operation["fact_key"] = key
        refs = operation.get("evidence_event_refs") or []
        if not isinstance(refs, list) or not all(isinstance(item, str) for item in refs):
            raise SemanticValidationError("evidence_event_refs must be a string list")
        operation["evidence_event_refs"] = refs
        return operation

    def _validate_decision_operations(
        self, decision: str, operations: Sequence[Mapping[str, Any]], current: SemanticEpoch | None
    ) -> None:
        operation_types = {value["operation"] for value in operations}
        if current is None and operation_types != {"add"}:
            raise SemanticValidationError("initial state accepts add operations only")
        allowed = {
            "reinforce": {"reinforce"},
            "extend": {"add", "reinforce"},
            "revise": {"replace", "retract", "reinforce", "add"},
            "uncertain": OPERATIONS,
        }[decision]
        if not operation_types <= allowed:
            raise SemanticValidationError("semantic decision and operations are inconsistent")
        if decision == "revise" and not operation_types.intersection({"replace", "retract"}):
            raise SemanticValidationError("revise requires replace or retract")

    def _validate_summary(
        self,
        summary: str,
        proposed: Mapping[str, SemanticFactRecord],
        trigger_event_refs: Sequence[str],
    ) -> None:
        if not summary.strip():
            raise SemanticValidationError("full_summary is required")
        exposes_event_ref = any(
            ref and re.search(rf"(?<!\w){re.escape(ref)}(?!\w)", summary)
            for ref in trigger_event_refs
        )
        if (
            _INTERNAL_TOKEN.search(summary)
            or any(key in summary for key in proposed)
            or exposes_event_ref
        ):
            raise SemanticValidationError("full_summary exposes an internal identifier")


def _derive_reducer_decision(output: dict[str, Any]) -> dict[str, Any]:
    """Translate the model-facing status and operations into an internal decision."""
    status = str(output.get("status") or "").lower()
    if status == "reinforce":
        operations = output.get("operations")
        if (
            isinstance(operations, list)
            and operations
            and all(
                isinstance(operation, Mapping)
                and str(operation.get("operation") or "").lower() == "reinforce"
                for operation in operations
            )
        ):
            # Accept the operation name misplaced as status only for pure reinforcement.
            # All operation and state validation below still applies.
            status = "apply"
    if status:
        if status not in {"apply", "uncertain"}:
            raise SemanticValidationError(
                "semantic reducer status must be apply or uncertain"
            )
    else:
        legacy_decision = str(output.get("decision") or "").lower()
        if legacy_decision not in DECISIONS:
            raise SemanticValidationError(
                "semantic reducer status must be apply or uncertain"
            )
        status = "uncertain" if legacy_decision == "uncertain" else "apply"

    normalized = dict(output)
    normalized["status"] = status
    if status == "uncertain":
        normalized["decision"] = "uncertain"
        return normalized

    operations = output.get("operations")
    if not isinstance(operations, list) or not operations:
        raise SemanticValidationError(
            "semantic reducer operations must be a non-empty list"
        )
    operation_names: set[str] = set()
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise SemanticValidationError("semantic operation must be an object")
        if "op" in operation:
            raise SemanticValidationError(
                "semantic operation must use 'operation', not 'op'"
            )
        operation_name = str(operation.get("operation") or "").lower()
        if operation_name not in OPERATIONS:
            raise SemanticValidationError(
                f"unknown semantic operation: {operation_name!r}"
            )
        operation_names.add(operation_name)

    if operation_names.intersection({"replace", "retract"}):
        decision = "revise"
    elif "add" in operation_names:
        decision = "extend"
    else:
        decision = "reinforce"
    normalized["decision"] = decision
    return normalized


class SemanticReducer:
    """LLM reducer with structural retry and no synthesized-summary fallback."""

    def __init__(
        self,
        client: ChatClient,
        max_retries: int | None = None,
        *,
        failure_policy: FailurePolicyConfig | None = None,
        sleeper=None,
        provider: str | None = None,
        model: str | None = None,
    ):
        self.client = client
        # ``max_retries`` remains accepted for source compatibility but V4 uses
        # only the unified failure policy.
        self.failure_policy = failure_policy or FailurePolicyConfig()
        self.sleeper = sleeper
        self.provider = provider
        self.model = model
        self.machine = SemanticEpochMachine()

    def reconcile(self, *, payload: Mapping[str, Any], apply_kwargs: Mapping[str, Any]) -> SemanticTransition:
        return self.reduce(
            payload=payload,
            validate=lambda output: self.machine.apply(
                reducer_output=output, **apply_kwargs
            ),
            unit_id=str(apply_kwargs.get("chain_id") or "semantic_reducer"),
            checkpoint_key=str(
                apply_kwargs.get("chain_id") or "semantic_reducer"
            ),
        )

    def reduce(
        self,
        *,
        payload: Mapping[str, Any],
        validate: Callable[[dict[str, Any]], _T],
        unit_id: str,
        checkpoint_key: str | None = None,
        on_failure: Callable[[int, Exception], None] | None = None,
    ) -> _T:
        """Run the canonical reducer prompt and validate each attempted output."""
        messages = _reducer_messages(payload)

        def reduce_once() -> _T:
            raw = self.client.chat(
                messages_with_json_schema(
                    messages, SEMANTIC_REDUCER_JSON_SCHEMA
                ),
                json_mode=True,
                json_schema=SEMANTIC_REDUCER_JSON_SCHEMA,
            )
            output = json.loads(raw)
            if (
                isinstance(output, list)
                and len(output) == 1
                and isinstance(output[0], dict)
            ):
                output = output[0]
            if not isinstance(output, dict):
                raise SemanticValidationError(
                    "semantic reducer output must be an object"
                )
            output = _derive_reducer_decision(output)
            return validate(output)

        def add_feedback(attempt: int, error: Exception) -> None:
            messages.append({
                "role": "user",
                "content": json.dumps({
                    "validator_feedback": {
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "details": getattr(error, "details", {}),
                    },
                    "instruction": "Return one corrected JSON object only.",
                }, ensure_ascii=False),
            })
            if on_failure is not None:
                on_failure(attempt, error)

        return retry_v4_call(
            reduce_once,
            policy=self.failure_policy,
            context=V4OperationContext(
                stage="semantic_update",
                unit_id=unit_id,
                provider=self.provider,
                model=self.model,
                checkpoint_key=checkpoint_key,
            ),
            error_type=V4BuildStageError,
            sleeper=self.sleeper,
            on_failure=add_feedback,
            validation_errors=(ValueError, TypeError, json.JSONDecodeError),
            validation_retries=1,
        )


class SemanticTimeline:
    """Chronological epoch container with bounded forward propagation for late facts."""

    def __init__(self, epochs: Sequence[SemanticEpoch] = ()):
        self.epochs = sorted(list(epochs), key=lambda item: (item.valid_from or "", item.epoch, item.id))

    @property
    def current(self) -> SemanticEpoch | None:
        return self.epochs[-1] if self.epochs else None


def _reducer_messages(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SEMANTIC_REDUCER_SYSTEM_PROMPT_V4,
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    return " ".join(text.split())


def _stable(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output
