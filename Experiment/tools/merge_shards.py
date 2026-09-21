"""P1-3: merge persona shards into one run directory, then score that.

The full benchmark (30 personas, 1,579 sessions) is far too long for a single
run, and a single run is also a single point of failure. The intended shape is:

* split the personas into shards with ``--start-index`` / ``--end-index`` (one
  shard per machine or per GPU group),
* run each shard into its own ``--output-dir``,
* merge the shards here, and score the merged directory once.

Every shard writes the same per-persona records as a normal run, so merging is
a concatenation with three guarantees this tool checks:

1. no persona appears in two shards (a re-run of an overlapping range would
   otherwise double-count its questions),
2. every persona of the requested range is present somewhere, and
3. the merged ``run_meta.json`` keeps the per-shard wall clocks and unit
   assignments, so the sharded run stays auditable.

Usage::

    python Experiment\\tools\\merge_shards.py --out Experiment\\runs\\full_merged ^
      --shards Experiment\\runs\\shard_0 Experiment\\runs\\shard_1
    python Experiment\\run_scoring.py --run-dir Experiment\\runs\\full_merged
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memconflict_eval import runtime  # noqa: E402
from memconflict_eval.data import load_personas  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_meta(shard: Path) -> dict[str, Any]:
    path = shard / "run_meta.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def merge(
    shard_dirs: list[Path],
    *,
    dataset: Path | None,
    allow_partial: bool = False,
) -> tuple[dict[str, Any], int]:
    personas: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    shard_meta: list[dict[str, Any]] = []

    for shard in shard_dirs:
        if not shard.is_dir():
            raise SystemExit(f"[error] shard directory not found: {shard}")
        rows = _read_jsonl(shard / "results.jsonl")
        sessions.extend(_read_jsonl(shard / "sessions.jsonl"))
        errors.extend(_read_jsonl(shard / "errors.jsonl"))
        meta = _load_meta(shard)
        shard_meta.append(
            {
                "Shard": str(shard),
                "Personas": len(rows),
                "Wall_Clock_s": meta.get("Wall_Clock_s"),
                "Persona_Workers": meta.get("Persona_Workers"),
                "Ollama_Units": meta.get("Ollama_Units"),
                "Answer_Workers": meta.get("Answer_Workers"),
                "Config": meta.get("Config"),
                "Models": meta.get("Models"),
                "Failed_Personas": meta.get("Failed_Personas") or [],
            }
        )
        for row in rows:
            persona_id = str(row.get("Persona_ID") or "")
            if not persona_id:
                continue
            if persona_id in seen:
                duplicates.append(persona_id)
                continue
            seen[persona_id] = str(shard)
            row["Source_Shard"] = str(shard)
            personas.append(row)

    order: dict[str, int] = {}
    if dataset is not None and Path(dataset).is_file():
        order = {
            persona.persona_id: index
            for index, persona in enumerate(load_personas(dataset))
        }
    personas.sort(key=lambda row: order.get(str(row.get("Persona_ID") or ""), 10**9))

    missing = [persona_id for persona_id in order if persona_id not in seen]
    units: list[str] = []
    assignments: dict[str, Any] = {}
    for meta in shard_meta:
        for unit in meta.get("Ollama_Units") or []:
            if unit not in units:
                units.append(unit)
    for row in personas:
        assignments[str(row.get("Persona_ID"))] = row.get("Ollama_Unit")

    configs = sorted(
        {str(meta.get("Config")) for meta in shard_meta if meta.get("Config")}
    )
    model_sets = sorted(
        {
            json.dumps(meta.get("Models") or {}, sort_keys=True, ensure_ascii=True)
            for meta in shard_meta
            if meta.get("Models")
        }
    )
    summary = {
        "Memory_System": personas[0].get("Memory_System") if personas else None,
        "Merged_At": datetime.now().isoformat(timespec="seconds"),
        "Dataset": str(Path(dataset).resolve()) if dataset else None,
        "Persona_Count": len(personas),
        "Personas_Expected_From_Dataset": len(order) or None,
        "Partial_Merge": bool(missing),
        "Shards_Merged": [str(shard) for shard in shard_dirs],
        "Shard_Configs": configs,
        "Mixed_Configs": len(configs) > 1 or len(model_sets) > 1,
        "Session_Count": len(sessions),
        "Failed_Personas": sorted({str(row.get("Persona_ID")) for row in errors}),
        "Duplicate_Personas": sorted(set(duplicates)),
        "Missing_Personas": missing,
        "Ollama_Units": units,
        "Persona_Unit_Assignments": assignments,
        "Shards": shard_meta,
        "Shard_Wall_Clock_Sum_s": round(
            sum(float(m.get("Wall_Clock_s") or 0.0) for m in shard_meta), 3
        ),
    }
    return {
        "personas": personas,
        "sessions": sessions,
        "errors": errors,
        "summary": summary,
    }, (1 if duplicates or (missing and not allow_partial) else 0)


def main(argv: list[str] | None = None) -> int:
    runtime.ensure_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Merged run directory.")
    parser.add_argument(
        "--shards",
        type=Path,
        nargs="+",
        required=True,
        help="Shard run directories (each with results.jsonl).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset used to check coverage and restore dataset order.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Point 27: accept a merge of only the shards that have finished, so "
            "the tables can be refreshed after every shard. The merged run_meta "
            "records Personas_Expected_From_Dataset / Partial_Merge."
        ),
    )
    args = parser.parse_args(argv)

    dataset = args.dataset or runtime.default_dataset_path()
    merged, status = merge(
        list(args.shards), dataset=dataset, allow_partial=args.allow_partial
    )
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "results.jsonl", merged["personas"])
    _write_jsonl(out / "sessions.jsonl", merged["sessions"])
    _write_jsonl(out / "errors.jsonl", merged["errors"])
    (out / "run_meta.json").write_text(
        json.dumps(merged["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = merged["summary"]
    print(
        f"[merge] {summary['Persona_Count']} persona(s), "
        f"{summary['Session_Count']} session row(s) from {len(args.shards)} shard(s)"
    )
    if summary.get("Partial_Merge"):
        print(
            f"[merge] partial: {summary['Persona_Count']}/"
            f"{summary['Personas_Expected_From_Dataset']} personas "
            f"from {len(args.shards)} shard(s); the tables describe this subset only",
            file=sys.stderr,
        )
        if not args.allow_partial:
            print(
                "        re-run with --allow-partial to treat this as a valid "
                "incremental merge (exit code 0)",
                file=sys.stderr,
            )
        else:
            print(
                "        not run yet: "
                + (", ".join(summary["Missing_Personas"]) or "none"),
                file=sys.stderr,
            )
    print(f"[merge] units: {', '.join(summary['Ollama_Units']) or '(config default)'}")
    print(f"[merge] shard wall clock sum: {summary['Shard_Wall_Clock_Sum_s'] / 3600.0:.2f} h")
    if summary.get("Mixed_Configs"):
        print(
            "[warn] the shards were produced by different configs/models; the "
            "merged tables are not a single-model result:",
            file=sys.stderr,
        )
        for config in summary.get("Shard_Configs") or []:
            print(f"       config: {config}", file=sys.stderr)
        for models in sorted(
            {
                json.dumps(meta.get("Models") or {}, ensure_ascii=True, sort_keys=True)
                for meta in summary["Shards"]
                if meta.get("Models")
            }
        ):
            print(f"       models: {models}", file=sys.stderr)
    if summary["Duplicate_Personas"]:
        print(
            "[fail] personas present in more than one shard: "
            + ", ".join(summary["Duplicate_Personas"]),
            file=sys.stderr,
        )
    if summary["Missing_Personas"] and not args.allow_partial:
        print(
            "[fail] personas missing from every shard: "
            + ", ".join(summary["Missing_Personas"]),
            file=sys.stderr,
        )
    if summary["Failed_Personas"]:
        print(
            "[warn] personas that failed during their shard: "
            + ", ".join(summary["Failed_Personas"]),
            file=sys.stderr,
        )
    print(f"[done] merged run directory: {out}")
    print(f"       next: python Experiment/run_scoring.py --run-dir {out}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
