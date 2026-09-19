from __future__ import annotations

import ast
import json
import re
import warnings
from typing import Any


def normalize_locomo_conversation(example: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("conversation", "conversations", "dialogue", "messages"):
        value = example.get(key)
        if isinstance(value, list):
            return _normalize_message_list(value)
        if isinstance(value, dict):
            return _normalize_locomo_session_dict(value)
    sessions = example.get("sessions")
    if isinstance(sessions, list):
        turns: list[dict[str, Any]] = []
        for session_index, session in enumerate(sessions):
            session_turns = session.get("turns") if isinstance(session, dict) else session
            for turn in _normalize_message_list(session_turns or []):
                turn["session_id"] = turn.get("session_id", session_index)
                turns.append(turn)
        return turns
    return []


def normalize_locomo_qas(example: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("qa", "qas", "questions", "question_answer"):
        value = example.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    if "question" in example:
        return [example]
    return []


def _normalize_message_list(messages: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            message = {"content": str(message)}
        role = message.get("role", message.get("speaker", "unknown"))
        speaker = message.get("speaker") or role
        normalized.append({
            "session_id": message.get("session_id", message.get("session", "default")),
            "turn_id": message.get("turn_id", message.get("turn", index + 1)),
            "role": role,
            "content": message.get("content", message.get("text", message.get("message", ""))),
            "timestamp": message.get("timestamp", message.get("time")),
            "speaker": speaker,
            "participant_id": message.get("participant_id") or message.get("speaker_id") or speaker,
            "mentioned_entities": message.get("mentioned_entities", message.get("entities", [])),
            "raw_turn": _json_safe(message),
        })
    return normalized


def _normalize_locomo_session_dict(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    session_keys = [
        key for key, value in conversation.items()
        if re.fullmatch(r"session_\d+", str(key)) and isinstance(value, list)
    ]
    for session_key in sorted(session_keys, key=_session_sort_key):
        session_time = conversation.get(f"{session_key}_date_time")
        for index, message in enumerate(conversation.get(session_key, [])):
            if not isinstance(message, dict):
                message = {"text": str(message)}
            turns.append(_normalize_locomo_turn(message, session_key, index + 1, session_time))
    return turns


def _json_safe(value: Any) -> Any:
    value = _decode_nested(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _session_sort_key(key: str) -> int:
    match = re.search(r"(\d+)$", key)
    return int(match.group(1)) if match else 0


def _normalize_locomo_turn(message: dict[str, Any], session_id: str, turn_index: int, session_time: Any) -> dict[str, Any]:
    content_parts = [str(message.get("content") or message.get("text") or message.get("message") or "")]
    if message.get("blip_caption"):
        content_parts.append(f"Image caption: {message['blip_caption']}")
    if message.get("query"):
        content_parts.append(f"Image query: {message['query']}")
    content = " ".join(part.strip() for part in content_parts if part and part.strip())
    role = message.get("role", message.get("speaker", "unknown"))
    speaker = message.get("speaker") or role
    return {
        "session_id": session_id,
        "turn_id": message.get("turn_id", message.get("dia_id", turn_index)),
        "role": role,
        "content": content,
        "timestamp": message.get("timestamp", message.get("time", session_time)),
        "speaker": speaker,
        "participant_id": message.get("participant_id") or message.get("speaker_id") or speaker,
        "mentioned_entities": message.get("mentioned_entities", message.get("entities", [])),
        "source": {"dia_id": message.get("dia_id"), "img_url": message.get("img_url")},
        "raw_turn": _json_safe(message),
    }


def _decode_nested(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] in "[{":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", SyntaxWarning)
                        return _decode_nested(ast.literal_eval(text))
                except (ValueError, SyntaxError):
                    return value
        return value
    if hasattr(value, "tolist"):
        try:
            return _decode_nested(value.tolist())
        except Exception:
            pass
    if isinstance(value, tuple):
        return [_decode_nested(item) for item in value]
    if isinstance(value, list):
        return [_decode_nested(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _decode_nested(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value
