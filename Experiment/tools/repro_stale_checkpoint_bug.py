"""Reproduce the upstream V4 stale-build-cache crash in seconds, offline.

The crash under investigation is::

    SemanticValidationError: reinforce references unknown fact key

which killed persona ``90e98aa7`` of the 2026-09-20 one-hour run while it was
ingesting its 7th session (``Session_ID 6``). Nothing in this demo talks to a
model or the network: it drives upstream's own code over a **copy** of that
run's memory store and reproduces the same exception deterministically.

Three upstream design facts are demonstrated, each with the responsible line:

1. window planning caches on the **turn count only** (``builder.py:505``):
   ``unit_id = f"conversation:{len(turns)}"``, so two different sessions with
   the same number of messages share one cache entry inside a scope;
2. the checkpoint loader matches on ``checkpoint_key`` + ``status='succeeded'``
   and ignores the recorded ``scope_revision`` (``storage.py:700-712``), so an
   output validated against an older semantic state is replayed as-is;
3. the cached-replay path applies that output with ``raise_on_validation=True``
   and never re-asks the model (``builder.py:2023-2031`` -> ``builder.py:2187``
   -> ``semantic.py:265`` -> ``semantic.py:422``), so one stale entry is fatal.

Usage::

    python Experiment/tools/repro_stale_checkpoint_bug.py
    python Experiment/tools/repro_stale_checkpoint_bug.py --store <memory.sqlite3>
    python Experiment/tools/repro_stale_checkpoint_bug.py --persona-index 14
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from memconflict_eval import runtime  # noqa: E402
from memconflict_eval.data import load_personas  # noqa: E402

PERSONA_PREFIX = "90e98aa7"
WINDOW_PLAN_STAGE = "window_plan"
SEMANTIC_STAGE = "semantic_update"


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def find_store(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"[error] store not found: {explicit}")
        return explicit
    root = EXPERIMENT_DIR / "runs"
    matches = sorted(root.glob(f"*/Memory/retrival_mem_v4_{PERSONA_PREFIX}*/memory.sqlite3"))
    if not matches:
        raise SystemExit(
            "[error] no store found for persona "
            f"{PERSONA_PREFIX} under {root}; pass --store <memory.sqlite3>"
        )
    if len(matches) > 1:
        print("[info] several stores for this persona; picking the most complete one:")
        for path in matches:
            print(f"        {path}")
    # A store that still has a published scope and semantic facts is the one
    # that carries the failing state; the earliest crashed attempt has neither.
    best_score, best = None, matches[-1]
    for path in matches:
        try:
            connection = sqlite3.connect(path)
            scopes = int(connection.execute("SELECT COUNT(*) FROM v4_participant_scopes").fetchone()[0])
            facts = int(connection.execute("SELECT COUNT(*) FROM v4_semantic_facts").fetchone()[0])
            connection.close()
        except sqlite3.Error:
            scopes, facts = 0, 0
        score = (scopes > 0, facts, path.stat().st_size)
        if best_score is None or score > best_score:
            best_score, best = score, path
    return best


def copy_store(store: Path) -> tuple[Path, Path]:
    """Copy the store (sqlite + faiss) so the demo never touches the original."""
    temp_root = Path(tempfile.mkdtemp(prefix="v4-repro-"))
    target_dir = temp_root / store.parent.name
    target_dir.mkdir(parents=True)
    shutil.copy2(store, target_dir / store.name)
    faiss_src = store.parent / "faiss"
    if faiss_src.is_dir():
        shutil.copytree(faiss_src, target_dir / "faiss")
    return target_dir, target_dir / store.name


def check_window_plan_key(store, namespace: str, dataset: Path, persona_index: int) -> None:
    """Fact 1: the plan cache key is the message count, so sessions collide."""
    banner("CHECK 1 / 3 - the window-plan cache key is the turn count (builder.py:505)")
    rows = list(
        store.conn.execute(
            "SELECT checkpoint_key, unit_id, scope_revision, status, updated_at"
            " FROM v4_build_checkpoints WHERE stage=? ORDER BY updated_at",
            (WINDOW_PLAN_STAGE,),
        )
    )
    print(f"window_plan cache entries in this scope: {len(rows)}")
    for row in rows:
        print(
            "  key=...:{unit:<18} revision={rev:<3} status={status:<9} written={at}".format(
                unit=row["unit_id"], rev=row["scope_revision"], status=row["status"], at=row["updated_at"]
            )
        )
    print()
    print("upstream builds that key as:  unit_id = f\"conversation:{len(turns)}\"")
    print("  Retrival-Mem/src/memory/v4/builder.py:505")
    print()

    personas = load_personas(dataset, start_index=persona_index, end_index=persona_index + 1)
    persona = personas[0]
    sizes: dict[int, list[int]] = {}
    for session in persona.sessions[:11]:
        sizes.setdefault(len(session.dialogue), []).append(session.session_id)
    collisions = {size: ids for size, ids in sizes.items() if len(ids) > 1}
    print(f"messages per session of persona {persona.persona_id[:8]} (first 11 sessions):")
    for size in sorted(sizes):
        marker = "  <-- COLLISION" if len(sizes[size]) > 1 else ""
        print(f"  {size:>4} messages: sessions {sizes[size]}{marker}")
    if collisions:
        print()
        for size, ids in sorted(collisions.items()):
            print(
                f"  => sessions {ids} all have {size} messages, so they share the single "
                f"cache key 'conversation:{size}'"
            )
            print(
                "     the second session's plan is served from the first session's cache "
                "(builder.py:515-517),\n"
                "     and the plan payload carries the FIRST session's turns"
            )
    print()
    print(f"namespace checked: {namespace}")


def check_loader_ignores_revision(store, namespace: str) -> None:
    """Fact 2: a succeeded row is returned no matter how old its revision is."""
    banner("CHECK 2 / 3 - the loader ignores scope_revision (storage.py:700-712)")
    scopes = list(store.conn.execute("SELECT id, revision FROM v4_participant_scopes"))
    if not scopes:
        print("no participant scope in this store; skipping")
        return
    scope_id, live_revision = scopes[0]["id"], scopes[0]["revision"]
    stale_revision = max(0, int(live_revision) - 1)
    key = f"{scope_id}:{SEMANTIC_STAGE}:repro-stale-key"
    payload = {
        "decision": "reinforce",
        "operations": [
            {
                "operation": "reinforce",
                "fact_key": "fact_" + "0" * 20,
                "evidence_event_refs": ["e_001"],
            }
        ],
    }
    # Write the row through upstream's own API; this is exactly the shape the
    # crashed run left behind (validated output, older revision, succeeded).
    store.record_build_checkpoint(
        checkpoint_key=key,
        namespace=namespace,
        scope_id=scope_id,
        scope_revision=stale_revision,
        stage=SEMANTIC_STAGE,
        unit_id="repro-stale-key",
        status="succeeded",
        attempt=1,
        provider="repro",
        model="repro",
        prompt_version="repro",
        output=payload,
    )
    loaded = store.load_build_checkpoint(key)
    print(f"scope {scope_id}")
    print(f"  live revision                        : {live_revision}")
    print(f"  revision recorded on the cached row   : {stale_revision}")
    print(f"  load_build_checkpoint() returned      : {'a payload (stale row accepted)' if loaded else 'None'}")
    if loaded:
        print("  => the loader only checks checkpoint_key + status='succeeded';")
        print("     scope_revision / provider / model / prompt_version are stored but never compared.")
    print()
    print("Retrival-Mem/src/memory/v4/storage.py:700")
    print('  "SELECT * FROM v4_build_checkpoints WHERE checkpoint_key=? AND status=\'succeeded\'"')


def reproduce_exception(store, namespace: str) -> None:
    """Fact 3: replaying a cached output validates against the current state."""
    banner("CHECK 3 / 3 - the cached output is replayed against the current state")
    from memory.v4.semantic import (  # noqa: E402 - upstream import
        SemanticEpoch,
        SemanticEpochMachine,
        SemanticFactRecord,
        SemanticValidationError,
    )

    candidates = []
    for candidate in store.conn.execute(
        "SELECT checkpoint_key, unit_id, scope_revision, status, provider, output_json"
        " FROM v4_build_checkpoints WHERE stage=?",
        (SEMANTIC_STAGE,),
    ):
        if str(candidate["provider"] or "") == "repro":
            continue  # the row written by CHECK 2, not a real cached output
        payload = json_loads(candidate["output_json"])
        operations = (payload or {}).get("operations") or []
        reinforce = [op for op in operations if str(op.get("operation")) == "reinforce"]
        if reinforce:
            candidates.append((int(candidate["scope_revision"] or 0), candidate, payload, reinforce[0]))
    if not candidates:
        print("no cached semantic_update payload with a 'reinforce' operation in this store")
        return
    # The latest such entry is the one closest to the crash.
    _revision, candidate, payload, operation = max(candidates, key=lambda item: item[0])
    fact_key = str(operation.get("fact_key"))
    topic_id = str(payload.get("topic_id") or "")
    routed = store.conn.execute(
        "SELECT id FROM v4_topic_chains WHERE topic_id=?", (topic_id,)
    ).fetchone()
    routed_chain = routed["id"] if routed else ""
    owner = store.conn.execute(
        "SELECT chain_id FROM v4_semantic_facts WHERE fact_key=?", (fact_key,)
    ).fetchone()
    owner_chain = owner["chain_id"] if owner else ""
    scope_id = scopes_of(store)[0] if scopes_of(store) else "scope_repro"

    def epoch_for(chain_id: str, label: str) -> SemanticEpoch:
        facts: dict[str, SemanticFactRecord] = {}
        if chain_id:
            for fact in store.conn.execute(
                "SELECT fact_key, subject, dimension, aspect, value_json, valid_from, confidence"
                " FROM v4_semantic_facts WHERE chain_id=? AND valid_to IS NULL",
                (chain_id,),
            ):
                facts[fact["fact_key"]] = SemanticFactRecord(
                    key=fact["fact_key"],
                    subject=fact["subject"],
                    dimension=fact["dimension"],
                    aspect=fact["aspect"],
                    value=json_loads(fact["value_json"]),
                    valid_from=fact["valid_from"],
                    confidence=float(fact["confidence"] or 0.9),
                )
        return SemanticEpoch(
            id=label,
            chain_id=chain_id or "chain_repro",
            epoch=1,
            summary="",
            facts=facts,
            valid_from=payload.get("valid_from"),
        )

    current = epoch_for(routed_chain, "repro-routed")
    owner_epoch = epoch_for(owner_chain, "repro-owner")
    print(f"cached payload from checkpoint : {candidate['checkpoint_key']}")
    print(f"  recorded status/revision     : {candidate['status']} / {candidate['scope_revision']}")
    print(f"  decision                     : {payload.get('decision')}")
    print(f"  operation                    : reinforce {fact_key}")
    print(f"  topic the payload routes to  : {topic_id!r} -> chain {routed_chain or '(no such topic now)'}")
    print(f"  chain that owns the fact key : {owner_chain or '(fact key no longer stored)'}")
    print(f"  facts in the routed epoch    : {len(current.facts)}")
    print(f"  facts in the owner epoch     : {len(owner_epoch.facts)}")
    print(f"  fact key in the routed epoch : {fact_key in current.facts}")
    print(f"  fact key in the owner epoch  : {fact_key in owner_epoch.facts}")
    print()
    refs = sorted(
        {
            str(ref)
            for item in payload.get("operations") or []
            for ref in (item.get("evidence_event_refs") or [])
        }
    )

    print("A) the cached output against the chain that currently owns the fact key:")
    try:
        SemanticEpochMachine().apply(
            participant_scope=scope_id,
            chain_id=owner_epoch.chain_id,
            current=owner_epoch,
            reducer_output=payload,
            boundary_time=payload.get("valid_from"),
            trigger_event_refs=refs,
        )
        print("   accepted -> this cached output still fits this state")
    except SemanticValidationError as error:
        print(f"   rejected too -> {type(error).__name__}: {error}")
    print()
    print("B) the same cached output against the epoch the replay routes it to:")
    try:
        SemanticEpochMachine().apply(
            participant_scope=scope_id,
            chain_id=current.chain_id,
            current=current,
            reducer_output=payload,
            boundary_time=payload.get("valid_from"),
            trigger_event_refs=refs,
        )
        print("   accepted (no mismatch in this store)")
    except SemanticValidationError as error:
        print(f"   REPRODUCED -> {type(error).__name__}: {error}")
        print()
        print("this is the call the cached-replay path makes:")
        print("  builder.py:2026  _apply_reducer_update(..., raise_on_validation=True)")
        print("  builder.py:2187  SemanticEpochMachine().apply(...)")
        print("  semantic.py:422  raise SemanticValidationError(")
        print("                       f'{operation_name} references unknown fact key')")
        print()
        print("the *fresh* path (builder.py:2034+) instead feeds the validation error back")
        print("to the model and retries, which is why this is fatal only on the cache path.")
    print()
    print("a cached reducer output is only valid in the exact state that produced it.")
    print("upstream replays it regardless of how far the state has moved, and raises")
    print("instead of recomputing -- that is the whole bug.")
    print()
    print("note: the rows carry status='failed' because the harness guard has already")
    print("      invalidated them on a later resume; at crash time they were 'succeeded'.")
    print()
    print(f"namespace: {namespace}")


def scopes_of(store) -> list[str]:
    return [str(row["id"]) for row in store.conn.execute("SELECT id FROM v4_participant_scopes")]


def json_loads(value):
    import json

    if value in (None, ""):
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=None, help="memory.sqlite3 to inspect")
    parser.add_argument(
        "--dataset", type=Path, default=runtime.default_dataset_path(), help="MemConflict dataset"
    )
    parser.add_argument(
        "--persona-index",
        type=int,
        default=14,
        help="dataset index of the persona that crashed (default 14 = 90e98aa7)",
    )
    args = parser.parse_args(argv)
    runtime.ensure_utf8_console()

    original = find_store(args.store)
    print(f"original store : {original}")
    store_dir, store_path = copy_store(original)
    print(f"working copy   : {store_path}")
    print("(the original is never modified)")
    recorded = original.parent.parent.parent / "errors.jsonl"
    if recorded.is_file():
        import json

        for line in recorded.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            print(f"recorded crash : {recorded} -> {row.get('Error')}")

    runtime.import_retrival_mem()  # Retrival-Mem/src on sys.path
    from memory.v4.storage import V4SQLiteStore  # noqa: E402 - upstream import

    store = V4SQLiteStore(str(store_path), read_only=False)
    try:
        namespaces = [
            str(row["namespace"])
            for row in store.conn.execute("SELECT DISTINCT namespace FROM v4_build_checkpoints")
        ]
        namespace = namespaces[0] if namespaces else ""
        check_window_plan_key(store, namespace, args.dataset, args.persona_index)
        check_loader_ignores_revision(store, namespace)
        reproduce_exception(store, namespace)
    finally:
        store.close()
        shutil.rmtree(store_dir.parent, ignore_errors=True)

    banner("SUMMARY - all three design facts live in upstream Retrival-Mem")
    print("1. builder.py:505   plan cache key = f\"conversation:{len(turns)}\"  -> cross-session collision")
    print("2. storage.py:700   loader matches key + status only               -> stale entries replay")
    print("3. builder.py:2023  replay path validates with raise_on_validation  -> one stale entry is fatal")
    print()
    print("The experiment harness (Experiment/) only calls the public")
    print("MemorySystem.ingest_conversation() API and never writes these code paths;")
    print("Experiment/memconflict_eval/memory.py drops the stale rows before each")
    print("session ingest so the replay cannot happen (see Stale_Checkpoints_Dropped")
    print("in sessions.jsonl).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
