"""Run the Retrival-Mem (AutoRetri V4) long-term memory system on MemConflict.

This mirrors ``eval_memzero.py`` / ``eval_a_mem.py``: it replays each persona's
``Full_Session_Chain`` session by session, ingests every session dialogue into
the memory system, and answers that session's ``Session_Questions`` from the
retrieved memory context. The output JSONL is consumed unchanged by
``scoring_retrival_mem.py`` -> ``eval_scoring.Generate_User_Evaluation``.

The memory side is the V4 backend from the Retrival-Mem repository. Only the
retrieval/answer protocol is MemConflict's; the answer LLM is the same
``llm_request`` helper used by the other MemConflict runners so that the answer
model is held constant across memory systems.
"""

import argparse
import copy
import json
import os
import shutil
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    import jsonlines
except ImportError:
    jsonlines = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from llm_request import llm_request, calculate_cumulative_cost
except Exception:
    llm_request = None

    def calculate_cumulative_cost(previous_cost: Optional[Dict], current_cost: Dict) -> Dict:
        if not isinstance(previous_cost, dict):
            return copy.deepcopy(current_cost)
        merged = copy.deepcopy(previous_cost)
        for key in ["input_tokens", "output_tokens", "total_tokens"]:
            merged[key] = (merged.get(key, 0) or 0) + (current_cost.get(key, 0) or 0)
        merged["total_cost_usd"] = (merged.get("total_cost_usd", 0.0) or 0.0) + (
            current_cost.get("total_cost_usd", 0.0) or 0.0
        )
        if merged.get("model") is None:
            merged["model"] = current_cost.get("model")
        merged["pricing_available"] = bool(
            merged.get("pricing_available") or current_cost.get("pricing_available")
        )
        return merged


load_dotenv()


RETRIVAL_MEM_ANSWER_SYSTEM_PROMPT = """You answer memory-evaluation questions using only the retrieved memory context.

Rules:
1. Use only the retrieved memories.
2. Do not invent facts that are not supported by the retrieved memories.
3. If the memories are insufficient, say that you cannot confirm.
4. If the memories contain inconsistent statements, briefly mention the inconsistency first and then give the best-supported answer.
5. Keep the answer concise, natural, and directly responsive to the question."""

MEMORY_SYSTEM_NAME = "retrival_mem"
MAX_STORED_RETRIEVED_MEMORIES = 5


def _default_retrival_mem_root() -> str:
    return os.path.abspath(
        os.path.join(CURRENT_DIR, "..", "..", "Retrival-Mem")
    )


def Load_Retrival_Mem_Runtime(retrival_mem_root: str):
    """Import the Retrival-Mem package from its checkout and return its API."""
    source_dir = os.path.join(os.path.abspath(retrival_mem_root), "src")
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(
            f"Retrival-Mem src directory not found: {source_dir}. "
            "Pass --retrival_mem_root pointing at the Retrival-Mem checkout."
        )
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    from memory import MemorySystem as RetrivalMemorySystem
    from memory.config import configure_backend_output_paths, load_config
    from api_history import ApiHistoryLogger

    return RetrivalMemorySystem, load_config, configure_backend_output_paths, ApiHistoryLogger


def load_jsonl_items(input_file: str) -> List[Dict[str, Any]]:
    if jsonlines is not None:
        with jsonlines.open(input_file) as reader:
            return [item for item in reader]
    items = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl_items(output_file: str, items: List[Dict[str, Any]]):
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if jsonlines is not None:
        with jsonlines.open(output_file, "w") as writer:
            for item in items:
                writer.write(item)
        return

    with open(output_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_dialogue_turn_order(key_name: str) -> int:
    """Parse dialogue_turn_12 -> 12 for sorting."""
    try:
        return int(str(key_name).split("_")[-1])
    except Exception:
        return 10**9


def Build_Session_Dialogue_List(session_dialogue: Any) -> List[Dict[str, Any]]:
    """Flatten Session_Dialogue dict into a chronological list of role/content turns."""
    if not isinstance(session_dialogue, dict):
        return []

    flattened_dialogue = []
    ordered_keys = sorted(session_dialogue.keys(), key=extract_dialogue_turn_order)

    for turn_key in ordered_keys:
        turn_value = session_dialogue.get(turn_key, [])
        if not isinstance(turn_value, list):
            continue
        for message_item in turn_value:
            if not isinstance(message_item, dict):
                continue
            role = message_item.get("role")
            content = message_item.get("content")
            if role in ["user", "assistant"] and content not in [None, ""]:
                flattened_dialogue.append({
                    "role": role,
                    "content": str(content),
                })

    return flattened_dialogue


def Build_Retrival_Mem_Namespace(persona_item: Dict[str, Any], version: str) -> str:
    """Build a stable, per-persona namespace inside the isolated memory store."""
    persona_id = str(persona_item.get("ID") or persona_item.get("uuid") or "unknown_persona")
    persona_short = persona_id[-6:] if len(persona_id) >= 6 else persona_id
    return f"memconflict:{persona_short}:{version}"


def Build_Session_Turns(dialogue_messages: List[Dict[str, Any]], session_date: str) -> List[Dict[str, Any]]:
    """Attach turn ids and the session date so V4 can ground time expressions."""
    turns = []
    for turn_index, message in enumerate(dialogue_messages, start=1):
        turns.append({
            "turn_id": turn_index,
            "role": message.get("role"),
            "speaker": message.get("role"),
            "content": message.get("content", ""),
            "timestamp": session_date,
            "session_timestamp": session_date,
        })
    return turns


def _memory_item_text(item: Any) -> str:
    """Render one V4 episodic memory node into a single judge-readable string."""
    title = str(getattr(item, "title", "") or "").strip()
    body = str(getattr(item, "text", "") or getattr(item, "summary", "") or "").strip()
    parts = [part for part in (title, body) if part]
    if title and body and title == body:
        parts = [body]

    semantic_memories = getattr(item, "semantic_memories", None) or []
    semantic = "; ".join(
        " ".join(
            str(value).strip()
            for value in (
                getattr(memory, "subject", ""),
                getattr(memory, "predicate", ""),
                getattr(memory, "object", ""),
            )
            if str(value or "").strip()
        )
        for memory in semantic_memories
    )
    if semantic:
        parts.append(f"semantic: {semantic}")
    return " | ".join(parts)


def _memory_item_created_at(item: Any) -> str:
    metadata = getattr(item, "metadata", None) or {}
    for candidate in (
        metadata.get("absolute_time_start"),
        metadata.get("observed_at"),
        getattr(item, "timestamp_start", None),
    ):
        if candidate not in (None, ""):
            return str(candidate)
    return "Unknown Time"


def Build_Retrieved_Memory_Context(retrieved_memories: List[Dict[str, Any]], namespace: str) -> str:
    """Convert Retrival-Mem retrieval results into answer prompt context."""
    lines = [f"Memories for user {namespace}:"]

    if len(retrieved_memories) == 0:
        lines.append("No relevant memories found.")
    else:
        for idx, item in enumerate(retrieved_memories, start=1):
            memory_text = item.get("memory", "")
            created_at = item.get("created_at", "Unknown Time")
            score = item.get("score", None)
            if score is None:
                lines.append(f"{idx}. [{created_at}] {memory_text}")
            else:
                lines.append(f"{idx}. [{created_at}] {memory_text} (score={score})")

    return "\n".join(lines)


def Setup_Retrival_Mem_System(
    retrival_mem_root: str,
    retrival_mem_config_path: str,
    memory_dir: str,
    reset_memory: bool,
):
    """Create an isolated V4 memory system rooted at ``memory_dir``."""
    RetrivalMemorySystem, load_config, configure_backend_output_paths, ApiHistoryLogger = (
        Load_Retrival_Mem_Runtime(retrival_mem_root)
    )

    if reset_memory and os.path.isdir(memory_dir):
        shutil.rmtree(memory_dir)
    os.makedirs(memory_dir, exist_ok=True)

    root = os.path.abspath(retrival_mem_root)
    config_path = retrival_mem_config_path
    if not os.path.isabs(config_path):
        config_path = os.path.join(root, config_path)
    env_path = os.path.join(root, ".env")
    config = load_config(config_path, env_path=env_path)
    configure_backend_output_paths(
        config,
        os.path.join(memory_dir, "memory.sqlite3"),
        os.path.join(memory_dir, "faiss"),
    )
    api_history_logger = ApiHistoryLogger(memory_dir)
    memory_system = RetrivalMemorySystem(config, api_history_logger=api_history_logger)
    return memory_system, config


def Add_Session_Dialogue_To_Retrival_Mem(
    memory_system: Any,
    namespace: str,
    dialogue_messages: List[Dict[str, Any]],
    session_item: Dict[str, Any],
    session_index: int,
) -> Tuple[float, Dict[str, Any]]:
    """Ingest one session dialogue into the V4 memory store."""
    if len(dialogue_messages) == 0:
        return 0.0, {"Dialogue_Added_To_Memory": False, "Dialogue_Message_Count": 0}

    session_date = str(session_item.get("Date") or "").strip()
    session_id = str(session_item.get("Session_ID", session_index))
    session_blob = {
        "session_id": session_id,
        "timestamp": session_date,
        "session_timestamp": session_date,
        "turns": Build_Session_Turns(dialogue_messages, session_date),
    }

    start_time = time.time()
    memory_system.ingest_conversation(
        namespace,
        [session_blob],
        {
            "dataset": "memconflict",
            "session_id": session_id,
            "session_date": session_date,
            "session_index": session_index,
        },
    )
    duration_ms = (time.time() - start_time) * 1000
    try:
        namespace_ready = bool(memory_system.is_namespace_ready(namespace))
    except Exception as ready_error:
        print(f"[DEBUG] is_namespace_ready failed for {namespace}: {ready_error}")
        namespace_ready = None
    add_result = {
        "Dialogue_Added_To_Memory": True,
        "Dialogue_Message_Count": len(dialogue_messages),
        "Add_Duration_ms": duration_ms,
        "Add_Batch_Count": 1,
        "Add_Batch_Size": len(dialogue_messages),
        "Namespace_Ready_After_Add": namespace_ready,
    }
    return duration_ms, add_result


def Search_Retrival_Mem_For_Question(
    memory_system: Any,
    namespace: str,
    question_text: str,
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """Retrieve memory items for one question through the V4 multi-round runtime."""
    start_time = time.time()
    result = memory_system.retrieve(question_text, namespace)
    duration_ms = (time.time() - start_time) * 1000

    episodic_memories = list(getattr(result, "episodic_memories", []) or [])
    retrieved_memories = [
        {
            "memory": _memory_item_text(item),
            "created_at": _memory_item_created_at(item),
            "score": getattr(item, "score", None),
        }
        for item in episodic_memories
    ]

    retrieval_metadata = {
        "Retrieved_Memory_Count": len(retrieved_memories),
        "Retrieval_Round_Count": len(list(getattr(result, "trace", []) or [])),
        "Retrieval_Node_Types": [
            str(getattr(item, "node_type", "") or "") for item in episodic_memories
        ],
    }
    return retrieved_memories, duration_ms, retrieval_metadata


def Generate_Answer_With_Retrieved_Memory(
    system_prompt: str, context_text: str, question_text: str
) -> Tuple[str, Dict[str, Any], float]:
    """Use the shared MemConflict answer LLM with the retrieved memory context."""
    zero_cost = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "model": None,
        "pricing_available": False,
    }

    if llm_request is None:
        raise ImportError("OpenAI dependencies are not available. Please install the required LLM dependencies.")

    user_prompt = (
        "Retrieved Memory Context:\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"{question_text}\n\n"
        "Answer:"
    )

    start_time = time.time()
    answer_text, cost_info = llm_request(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        return_parsed_json=False,
        extract_json=False,
    )
    duration_ms = (time.time() - start_time) * 1000

    if isinstance(answer_text, tuple):
        answer_text = answer_text[0]

    return str(answer_text).strip(), cost_info or zero_cost, duration_ms


def Answer_Questions_For_One_Session(
    memory_system: Any,
    namespace: str,
    session_item: Dict[str, Any],
    top_k: int,
    system_prompt: str,
    overwrite_existing_answers: bool,
) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
    """Answer all questions in one session using Retrival-Mem retrieval."""
    current_stage_total_cost = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "model": None,
        "pricing_available": False,
        "note": "Retrival-Mem retrieval and question answering",
    }

    answered_question_count = 0
    session_retrieval_time_ms = 0.0
    session_response_time_ms = 0.0
    session_questions = copy.deepcopy(session_item.get("Session_Questions", []))
    session_memory_metadata = copy.deepcopy(session_item.get("Session_Memory_Metadata", {}))
    session_memory_metadata["Memory_System"] = MEMORY_SYSTEM_NAME
    session_memory_metadata["Top_K"] = top_k

    for q_idx, question_item in enumerate(session_questions):
        existing_answer = question_item.get("Model_Answer")
        if (not overwrite_existing_answers) and existing_answer not in [None, ""]:
            continue

        question_text = str(question_item.get("question", "")).strip()
        if not question_text:
            continue

        retrieved_memories, search_duration_ms, retrieval_metadata = Search_Retrival_Mem_For_Question(
            memory_system=memory_system,
            namespace=namespace,
            question_text=question_text,
        )
        answer_context_text = Build_Retrieved_Memory_Context(retrieved_memories[:top_k], namespace)

        answer_text, cost_info, response_duration_ms = Generate_Answer_With_Retrieved_Memory(
            system_prompt=system_prompt,
            context_text=answer_context_text,
            question_text=question_text,
        )

        question_item["Retrieved_Memory_Context"] = answer_context_text
        question_item["Retrieved_Memories"] = retrieved_memories
        question_item["Model_Answer"] = answer_text
        question_item["Memory_Search_Duration_ms"] = search_duration_ms
        question_item["Retrieval_Round_Count"] = retrieval_metadata.get("Retrieval_Round_Count")
        question_item["Retrieved_Memory_Count"] = retrieval_metadata.get("Retrieved_Memory_Count")
        question_item["Retrieval_Node_Types"] = retrieval_metadata.get("Retrieval_Node_Types")
        question_item["Response_Duration_ms"] = response_duration_ms
        question_item["Actual_Top_K"] = top_k
        question_item["Memory_System"] = MEMORY_SYSTEM_NAME
        session_questions[q_idx] = question_item

        session_retrieval_time_ms += search_duration_ms
        session_response_time_ms += response_duration_ms
        answered_question_count += 1
        current_stage_total_cost["input_tokens"] += cost_info.get("input_tokens", 0) or 0
        current_stage_total_cost["output_tokens"] += cost_info.get("output_tokens", 0) or 0
        current_stage_total_cost["total_tokens"] += cost_info.get("total_tokens", 0) or 0
        current_stage_total_cost["total_cost_usd"] += cost_info.get("total_cost_usd", 0.0) or 0.0
        if current_stage_total_cost["model"] is None:
            current_stage_total_cost["model"] = cost_info.get("model")
        if cost_info.get("pricing_available") is True:
            current_stage_total_cost["pricing_available"] = True

    session_memory_metadata["Session_Retrieval_Time_ms"] = session_retrieval_time_ms
    session_memory_metadata["Session_Response_Time_ms"] = session_response_time_ms
    session_memory_metadata["Session_Answered_Question_Count"] = answered_question_count
    session_item["Session_Questions"] = session_questions
    session_item["Session_Memory_Metadata"] = session_memory_metadata
    return session_item, answered_question_count, current_stage_total_cost


def place_session_memory_metadata_before_event_types(session_item: Dict[str, Any]) -> Dict[str, Any]:
    """Place Session_Memory_Metadata before Event_Types for readability."""
    target_key = "Session_Memory_Metadata"
    if target_key not in session_item:
        return session_item
    metadata = session_item.pop(target_key)
    reordered: Dict[str, Any] = {}
    for key, value in session_item.items():
        if key == "Event_Types":
            reordered[target_key] = metadata
        reordered[key] = value
    if target_key not in reordered:
        reordered[target_key] = metadata
    return reordered


def Build_Compact_Retrival_Mem_Question(question_item: Dict[str, Any], keep_top_k: int) -> Dict[str, Any]:
    compact_question = {
        "question_id": question_item.get("question_id"),
        "question": question_item.get("question"),
        "answer": question_item.get("answer"),
        "conflict_type": question_item.get("conflict_type"),
        "ability_target": question_item.get("ability_target"),
        "difficulty": question_item.get("difficulty"),
        "Model_Answer": question_item.get("Model_Answer"),
        "Memory_Search_Duration_ms": question_item.get("Memory_Search_Duration_ms"),
        "Response_Duration_ms": question_item.get("Response_Duration_ms"),
        "Actual_Top_K": question_item.get("Actual_Top_K"),
        "Memory_System": question_item.get("Memory_System"),
        "Retrieval_Round_Count": question_item.get("Retrieval_Round_Count"),
        "Retrieved_Memory_Count": question_item.get("Retrieved_Memory_Count"),
        "Retrieval_Node_Types": question_item.get("Retrieval_Node_Types"),
    }

    retrieved_memories = question_item.get("Retrieved_Memories", [])
    if isinstance(retrieved_memories, list):
        compact_question["Retrieved_Memories"] = retrieved_memories[
            : max(keep_top_k, MAX_STORED_RETRIEVED_MEMORIES)
        ]
    else:
        compact_question["Retrieved_Memories"] = []

    return compact_question


def Build_Compact_Retrival_Mem_Session(session_item: Dict[str, Any], keep_top_k: int) -> Dict[str, Any]:
    compact_session = {
        "Session_ID": session_item.get("Session_ID"),
        "Date": session_item.get("Date"),
        "Question_Trigger_Types": copy.deepcopy(session_item.get("Question_Trigger_Types", [])),
        "Session_Question_Count": session_item.get("Session_Question_Count", 0),
        "Session_Memory_Metadata": copy.deepcopy(session_item.get("Session_Memory_Metadata", {})),
        "Session_Questions": [],
    }

    session_questions = session_item.get("Session_Questions", [])
    if isinstance(session_questions, list):
        compact_session["Session_Questions"] = [
            Build_Compact_Retrival_Mem_Question(question_item, keep_top_k)
            for question_item in session_questions
            if isinstance(question_item, dict)
        ]

    return compact_session


def Build_Compact_Retrival_Mem_Result_Item(
    persona_item: Dict[str, Any],
    updated_chain: List[Dict[str, Any]],
    total_answered_question_count: int,
    answered_session_count: int,
    final_cost: Dict[str, Any],
    namespace: str,
    eval_top_k: int,
    persona_runtime_summary: Dict[str, Any],
    observable_token_cost_summary: Dict[str, Any],
) -> Dict[str, Any]:
    compact_result = {
        "ID": persona_item.get("ID"),
        "Memory_System": MEMORY_SYSTEM_NAME,
        "Retrival_Mem_Namespace": namespace,
        "Eval_Top_K": eval_top_k,
        "Answered_Session_Count": answered_session_count,
        "Answered_Question_Count": total_answered_question_count,
        "Retrival_Mem_Runtime_Summary": persona_runtime_summary,
        "Observable_Token_Cost_Summary": observable_token_cost_summary,
        "token_cost": final_cost,
        "Full_Session_Chain": [
            Build_Compact_Retrival_Mem_Session(session_item, eval_top_k)
            for session_item in updated_chain
        ],
    }
    return compact_result


def Generate_Single_Persona_Retrival_Mem_Eval(
    persona_item: Dict[str, Any],
    retrival_mem_root: str,
    retrival_mem_config_path: str,
    output_dir: str,
    system_prompt: str,
    top_k: int,
    version: str,
    overwrite_existing_answers: bool,
    reset_memory: bool,
):
    """Run the Retrival-Mem evaluation for one persona."""
    memory_system = None
    try:
        persona_id = str(persona_item.get("ID") or persona_item.get("uuid") or "unknown_persona")
        persona_short = persona_id[-6:] if len(persona_id) >= 6 else persona_id
        memory_dir = os.path.join(output_dir, "Memory", f"{MEMORY_SYSTEM_NAME}_{persona_short}_{version}")
        memory_system, _config = Setup_Retrival_Mem_System(
            retrival_mem_root=retrival_mem_root,
            retrival_mem_config_path=retrival_mem_config_path,
            memory_dir=memory_dir,
            reset_memory=reset_memory,
        )

        previous_cost = persona_item.get("token_cost", None)
        full_session_chain = copy.deepcopy(persona_item["Full_Session_Chain"])
        persona_start_time = time.time()
        namespace = Build_Retrival_Mem_Namespace(persona_item, version)

        total_answered_question_count = 0
        answered_session_count = 0
        persona_add_time_ms = 0.0
        persona_retrieval_time_ms = 0.0
        persona_response_time_ms = 0.0
        session_total_runtime_ms_sum = 0.0
        current_stage_total_cost = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "model": None,
            "pricing_available": False,
            "note": "Retrival-Mem retrieval and question answering",
        }

        for current_idx, session_item in enumerate(full_session_chain):
            print(
                f"[DEBUG] Processing session {current_idx + 1}/{len(full_session_chain)} for Retrival-Mem"
            )
            session_start_time = time.time()

            dialogue_messages = Build_Session_Dialogue_List(session_item.get("Session_Dialogue", {}))
            add_duration_ms, add_result = Add_Session_Dialogue_To_Retrival_Mem(
                memory_system=memory_system,
                namespace=namespace,
                dialogue_messages=dialogue_messages,
                session_item=session_item,
                session_index=current_idx,
            )
            persona_add_time_ms += add_duration_ms
            session_item["Session_Memory_Metadata"] = {
                "Memory_System": MEMORY_SYSTEM_NAME,
                "Top_K": top_k,
                **add_result,
            }

            session_questions = session_item.get("Session_Questions", [])
            if not isinstance(session_questions, list) or len(session_questions) == 0:
                session_item["Session_Memory_Metadata"]["Session_Retrieval_Time_ms"] = 0.0
                session_item["Session_Memory_Metadata"]["Session_Response_Time_ms"] = 0.0
                session_item["Session_Memory_Metadata"]["Session_Answered_Question_Count"] = 0
                session_item["Session_Memory_Metadata"]["Session_Total_Runtime_ms"] = (
                    time.time() - session_start_time
                ) * 1000
                session_total_runtime_ms_sum += session_item["Session_Memory_Metadata"]["Session_Total_Runtime_ms"]
                full_session_chain[current_idx] = place_session_memory_metadata_before_event_types(
                    session_item
                )
                continue

            answered_session_count += 1
            updated_session_item, answered_question_count, call_cost = Answer_Questions_For_One_Session(
                memory_system=memory_system,
                namespace=namespace,
                session_item=session_item,
                top_k=top_k,
                system_prompt=system_prompt,
                overwrite_existing_answers=overwrite_existing_answers,
            )

            total_answered_question_count += answered_question_count
            current_stage_total_cost["input_tokens"] += call_cost.get("input_tokens", 0) or 0
            current_stage_total_cost["output_tokens"] += call_cost.get("output_tokens", 0) or 0
            current_stage_total_cost["total_tokens"] += call_cost.get("total_tokens", 0) or 0
            current_stage_total_cost["total_cost_usd"] += call_cost.get("total_cost_usd", 0.0) or 0.0
            if current_stage_total_cost["model"] is None:
                current_stage_total_cost["model"] = call_cost.get("model")
            if call_cost.get("pricing_available") is True:
                current_stage_total_cost["pricing_available"] = True

            session_metadata = updated_session_item.get("Session_Memory_Metadata", {})
            persona_retrieval_time_ms += session_metadata.get("Session_Retrieval_Time_ms", 0.0) or 0.0
            persona_response_time_ms += session_metadata.get("Session_Response_Time_ms", 0.0) or 0.0
            session_metadata["Session_Total_Runtime_ms"] = (time.time() - session_start_time) * 1000
            updated_session_item["Session_Memory_Metadata"] = session_metadata
            session_total_runtime_ms_sum += session_metadata["Session_Total_Runtime_ms"]

            full_session_chain[current_idx] = place_session_memory_metadata_before_event_types(
                updated_session_item
            )
            print(
                f"[DEBUG] Session {session_item.get('Session_ID', current_idx)} completed - "
                f"questions answered: {answered_question_count}"
            )

        persona_total_runtime_ms = (time.time() - persona_start_time) * 1000
        persona_runtime_summary = {
            "Persona_Add_Time_ms": persona_add_time_ms,
            "Persona_Retrieval_Time_ms": persona_retrieval_time_ms,
            "Persona_Response_Time_ms": persona_response_time_ms,
            "Persona_Total_Runtime_ms": persona_total_runtime_ms,
            "Average_Add_Time_Per_Session_ms": (persona_add_time_ms / len(full_session_chain))
            if len(full_session_chain) > 0
            else 0.0,
            "Average_Retrieval_Time_Per_Session_ms": (persona_retrieval_time_ms / answered_session_count)
            if answered_session_count > 0
            else 0.0,
            "Average_Response_Time_Per_Session_ms": (persona_response_time_ms / answered_session_count)
            if answered_session_count > 0
            else 0.0,
            "Average_Total_Runtime_Per_Session_ms": (session_total_runtime_ms_sum / len(full_session_chain))
            if len(full_session_chain) > 0
            else 0.0,
        }

        observable_token_cost_summary = {
            "Stage_Name": "retrival_mem_answer_generation",
            "Input_Tokens": current_stage_total_cost.get("input_tokens", 0) or 0,
            "Output_Tokens": current_stage_total_cost.get("output_tokens", 0) or 0,
            "Total_Tokens": current_stage_total_cost.get("total_tokens", 0) or 0,
            "Total_Cost_USD": current_stage_total_cost.get("total_cost_usd", 0.0) or 0.0,
            "Model": current_stage_total_cost.get("model"),
            "Pricing_Available": bool(current_stage_total_cost.get("pricing_available")),
        }
        final_cost = calculate_cumulative_cost(previous_cost, current_stage_total_cost)
        return (
            full_session_chain,
            total_answered_question_count,
            answered_session_count,
            final_cost,
            namespace,
            persona_runtime_summary,
            observable_token_cost_summary,
        )

    except Exception as e:
        print(f"[DEBUG] Generate_Single_Persona_Retrival_Mem_Eval failed: {e}:{traceback.format_exc()}")
        raise
    finally:
        if memory_system is not None:
            try:
                memory_system.close()
            except Exception as close_error:
                print(f"[DEBUG] Failed to close Retrival-Mem system: {close_error}")


def Generate_User_Retrival_Mem_Eval(
    input_jsonl_path: str,
    output_jsonl_path: str,
    output_json_path: str,
    retrival_mem_root: str,
    retrival_mem_config_path: str,
    system_prompt: str,
    top_k: int,
    start_idx: int,
    end_idx: Optional[int],
    version: str,
    overwrite_existing_answers: bool,
    reset_memory: bool,
):
    """Batch entry for the Retrival-Mem MemConflict evaluation."""
    print(f"Processing file: {input_jsonl_path}")
    print(f"Output file: {output_jsonl_path}")

    try:
        print("[DEBUG] Using built-in Retrival-Mem answer prompt")
        print(f"[DEBUG] Top-K retrieval for answer context: {top_k}")
        print(f"[DEBUG] Retrival-Mem root: {retrival_mem_root}")

        all_personas = load_jsonl_items(input_jsonl_path)
        selected_items = all_personas[start_idx:end_idx] if end_idx is not None else all_personas[start_idx:]
        print(f"[DEBUG] Read {len(all_personas)} personas")

        output_dir = os.path.dirname(output_jsonl_path) or CURRENT_DIR
        all_results = []
        for item_idx, persona_item in enumerate(selected_items):
            absolute_idx = start_idx + item_idx
            print(f"[DEBUG] Processing persona {absolute_idx + 1}/{len(all_personas)}")

            (
                updated_chain,
                total_answered_question_count,
                answered_session_count,
                final_cost,
                namespace,
                persona_runtime_summary,
                observable_token_cost_summary,
            ) = Generate_Single_Persona_Retrival_Mem_Eval(
                persona_item=persona_item,
                retrival_mem_root=retrival_mem_root,
                retrival_mem_config_path=retrival_mem_config_path,
                output_dir=output_dir,
                system_prompt=system_prompt,
                top_k=top_k,
                version=version,
                overwrite_existing_answers=overwrite_existing_answers,
                reset_memory=reset_memory,
            )

            result_item = Build_Compact_Retrival_Mem_Result_Item(
                persona_item=persona_item,
                updated_chain=updated_chain,
                total_answered_question_count=total_answered_question_count,
                answered_session_count=answered_session_count,
                final_cost=final_cost,
                namespace=namespace,
                eval_top_k=top_k,
                persona_runtime_summary=persona_runtime_summary,
                observable_token_cost_summary=observable_token_cost_summary,
            )
            all_results.append(result_item)

            print(
                f"[DEBUG] Persona {absolute_idx + 1} completed - "
                f"Answered sessions: {answered_session_count}, Answered questions: {total_answered_question_count}"
            )

        write_jsonl_items(output_jsonl_path, all_results)

        output_json_dir = os.path.dirname(output_json_path)
        if output_json_dir:
            os.makedirs(output_json_dir, exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as f:
            if len(all_results) == 1:
                json.dump(all_results[0], f, ensure_ascii=False, indent=4)
            else:
                json.dump(all_results, f, ensure_ascii=False, indent=4)

        print(f"[DEBUG] Wrote {len(all_results)} persona results to {output_jsonl_path}")

    except Exception as e:
        print(f"[DEBUG] Generate_User_Retrival_Mem_Eval failed: {e}:{traceback.format_exc()}")
        raise


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Retrival-Mem on the MemConflict benchmark.")
    parser.add_argument(
        "--input_jsonl_path",
        type=str,
        default=os.path.join(CURRENT_DIR, "..", "Data", "Step4_4.jsonl"),
    )
    parser.add_argument(
        "--output_jsonl_path",
        type=str,
        default=os.path.join(CURRENT_DIR, "Results", "retrival_mem_results.jsonl"),
    )
    parser.add_argument(
        "--output_json_path",
        type=str,
        default=os.path.join(CURRENT_DIR, "Results", "retrival_mem_results.json"),
    )
    parser.add_argument(
        "--retrival_mem_root",
        type=str,
        default=os.getenv("RETRIVAL_MEM_ROOT") or _default_retrival_mem_root(),
        help="Path to the Retrival-Mem checkout containing src/ and configs/.",
    )
    parser.add_argument(
        "--retrival_mem_config",
        type=str,
        default=os.getenv("RETRIVAL_MEM_CONFIG") or "configs/default.yaml",
        help="Config path passed to Retrival-Mem load_config (relative to its root).",
    )
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--overwrite_existing_answers", action="store_true")
    parser.add_argument(
        "--keep_memory",
        action="store_true",
        help="Reuse an existing per-persona memory store instead of rebuilding it.",
    )
    args = parser.parse_args()
    args.reset_memory = not args.keep_memory
    return args


if __name__ == "__main__":
    args = build_args()
    Generate_User_Retrival_Mem_Eval(
        input_jsonl_path=args.input_jsonl_path,
        output_jsonl_path=args.output_jsonl_path,
        output_json_path=args.output_json_path,
        retrival_mem_root=args.retrival_mem_root,
        retrival_mem_config_path=args.retrival_mem_config,
        system_prompt=RETRIVAL_MEM_ANSWER_SYSTEM_PROMPT,
        top_k=args.top_k,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        version=args.version,
        overwrite_existing_answers=args.overwrite_existing_answers,
        reset_memory=args.reset_memory,
    )
