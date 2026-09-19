"""Answer-only rendering of already delivered evidence; never fetches memories."""
from __future__ import annotations

import json
from typing import Any, Mapping


def prepare_answer_context(bundle: Any, cached_context: str,
                           options: Mapping[str, Any]) -> str:
    """Rebuild optional event-bound input from cached items without mutating them."""
    mode = options.get("answer_context_presentation", "legacy")
    if mode == "legacy":
        return cached_context
    if mode != "event_bound_dedup":
        raise ValueError(f"Unknown answer_context_presentation: {mode}")
    items = getattr(bundle, "episodic_memories", None)
    if not items:
        return cached_context
    # Keep distinct speakers/timestamps/turns even if their words coincide.
    sources: dict[tuple[str, ...], dict[str, Any]] = {}
    events = []
    for index, item in enumerate(items, 1):
        metadata = item.metadata
        event: dict[str, Any] = {
            "event": index, "kind": item.node_type, "title": item.title,
            "description": item.summary or item.text,
        }
        if item.text and item.text != item.summary:
            event["memory_content"] = item.text
        refs = []
        for record in metadata.get("source_evidence") or []:
            if not isinstance(record, dict) or not record.get("content"):
                continue
            identity = tuple(str(record.get(k) or "") for k in (
                "session_id", "turn_id", "role", "participant_id", "observed_at", "content"))
            if identity not in sources:
                sources[identity] = {
                    "passage": len(sources) + 1,
                    "speaker": record.get("role"),
                    "participant": record.get("participant_id"),
                    "conversation_time": record.get("observed_at"),
                    "original": record["content"],
                }
            source = sources[identity]
            if source["passage"] not in refs:
                refs.append(source["passage"])
        event["original_passages"] = refs
        # A source copied as the memory text must not appear a second time.
        originals = {source["original"] for source in sources.values()
                     if source["passage"] in refs}
        for key in ("description", "memory_content"):
            if event.get(key) in originals:
                event.pop(key)
        event["candidate_time"] = {
            key: metadata[key] for key in (
                "raw_time_expression", "absolute_time_start", "absolute_time_end",
                "time_precision", "normalization_status", "normalization_method",
                "observed_at") if metadata.get(key) not in (None, "")
        }
        if item.timestamp_start or item.timestamp_end:
            event["stored_time_bounds"] = [item.timestamp_start, item.timestamp_end]
        if metadata.get("unresolved_conflict"):
            event["unresolved_conflict"] = True
        if item.entities:
            event["entities"] = item.entities
        if item.semantic_memories:
            event["semantic_assertions"] = [
                {"subject": s.subject, "predicate": s.predicate, "object": s.object,
                 "qualifiers": s.qualifiers} for s in item.semantic_memories]
        events.append(event)
    event_context = json.dumps({
        "evidence_note": "Events link to original passages below. Passages are shown once; "
        "shared passages are not independent corroboration. Candidate times and stored bounds "
        "are memory metadata, NOT guaranteed original facts, even if marked builder_explicit. "
        "Use each passage's speaker and conversation time to interpret its event and relative "
        "time. Verify which occurrence each candidate date describes against the original.",
        "events": events,
    }, ensure_ascii=False, separators=(",", ":"))
    # Leave original text unescaped: quote-grounding must work on newlines and
    # quotation marks exactly as it does with the legacy plain-text context.
    passages = []
    for source in sources.values():
        header = json.dumps({k: v for k, v in source.items() if k != "original"},
                            ensure_ascii=False)
        passages.append(f"{header}\n{source['original']}")
    return event_context + "\n\n[Original passages]\n" + "\n\n".join(passages)
