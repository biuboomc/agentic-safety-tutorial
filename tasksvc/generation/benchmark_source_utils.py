import re
import json
import copy
from collections import Counter
from pathlib import Path


def benchmark_slugify(value):
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def normalize_source_task_id(source, *parts):
    pieces = [benchmark_slugify(source)]
    for part in parts:
        slug = benchmark_slugify(part)
        if slug:
            pieces.append(slug)
    return "__".join(piece for piece in pieces if piece)


def make_function_schema(name, description, parameters=None):
    return {
        "type": "function",
        "function": {
            "name": str(name or "").strip(),
            "description": str(description or "").strip(),
            "parameters": parameters or {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }


def make_tool(
    name,
    description,
    *,
    domain="general",
    parameters=None,
    reads_state_keys=None,
    writes_state_keys=None,
    labels=None,
    metadata=None,
):
    return {
        "name": str(name or "").strip(),
        "domain": str(domain or "general"),
        "labels": list(labels or []),
        "schema": make_function_schema(name, description, parameters=parameters),
        "simulator_requirements": {
            "reads_state_keys": list(reads_state_keys or []),
            "writes_state_keys": list(writes_state_keys or []),
        },
        **({"metadata": metadata} if metadata else {}),
    }


def dedupe_strings(values):
    seen = set()
    ordered = []
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def preview_text(value, limit=240):
    text = str(value or "").replace("\r\n", "\n").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def make_text_only_response_tool(domain, *, benchmark_name):
    return make_tool(
        "respond_directly",
        "Return the final answer directly when the benchmark exposes no callable tools.",
        domain=domain,
        parameters={
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "Final answer to present to the user.",
                }
            },
            "required": ["response"],
        },
        labels=["utility"],
        metadata={
            "synthetic_tool": True,
            "synthetic_tool_reason": "text_only_benchmark_task",
            "benchmark": benchmark_name,
        },
    )


def _stable_json_dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _counter_to_dict(counter):
    return {key: counter[key] for key in sorted(counter)}


def build_tool_pool_from_source_tasks(source_tasks):
    tool_pool = []
    seen = set()
    for source_task in source_tasks or []:
        for tool in source_task.get("tool_pool") or source_task.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            key = _stable_json_dumps(tool)
            if key in seen:
                continue
            seen.add(key)
            tool_pool.append(copy.deepcopy(tool))
    return tool_pool


def summarize_benchmark_accounting(source_tasks, tool_pool=None):
    tasks = list(source_tasks or [])
    resolved_tool_pool = list(tool_pool or build_tool_pool_from_source_tasks(tasks))

    domains = Counter()
    evaluation_tracks = Counter()
    suites = Counter()
    attack_names = Counter()
    bundle_kinds = Counter()
    sources = Counter()
    domain_track_pairs = Counter()

    tasks_with_environment_snapshot = 0
    tasks_with_content_artifacts = 0
    tasks_with_ground_truth_calls = 0
    tasks_with_injection_ground_truth_calls = 0

    user_task_ids = set()
    injection_task_ids = set()
    attack_spec_ids = set()

    for task in tasks:
        if not isinstance(task, dict):
            continue
        domain = str(task.get("domain") or "").strip() or "unknown"
        domains[domain] += 1

        source_name = str(task.get("source") or "").strip() or "unknown"
        sources[source_name] += 1

        metadata = task.get("metadata") or {}
        track = str(metadata.get("evaluation_track") or "").strip() or "unknown"
        evaluation_tracks[track] += 1
        domain_track_pairs[f"{domain}::{track}"] += 1

        suite = str(metadata.get("suite") or "").strip()
        if suite:
            suites[suite] += 1

        attack_name = str(metadata.get("attack_name") or "").strip()
        if attack_name:
            attack_names[attack_name] += 1

        bundle_kind = str(metadata.get("bundle_kind") or "").strip()
        if bundle_kind:
            bundle_kinds[bundle_kind] += 1

        user_task_id = str(metadata.get("user_task_id") or "").strip()
        if user_task_id:
            user_task_ids.add(user_task_id)

        injection_task_id = str(metadata.get("injection_task_id") or "").strip()
        if injection_task_id:
            injection_task_ids.add(injection_task_id)

        attack_spec_id = str(metadata.get("attack_spec_id") or "").strip()
        if attack_spec_id:
            attack_spec_ids.add(attack_spec_id)

        if task.get("environment_snapshot"):
            tasks_with_environment_snapshot += 1
        if task.get("content_artifacts"):
            tasks_with_content_artifacts += 1
        if metadata.get("user_ground_truth_calls"):
            tasks_with_ground_truth_calls += 1
        if metadata.get("injection_ground_truth_calls"):
            tasks_with_injection_ground_truth_calls += 1

    return {
        "source_task_count": len(tasks),
        "tool_pool_count": len(resolved_tool_pool),
        "tool_reference_count": sum(len((task or {}).get("tools") or []) for task in tasks),
        "domain_counts": _counter_to_dict(domains),
        "evaluation_track_counts": _counter_to_dict(evaluation_tracks),
        "suite_counts": _counter_to_dict(suites),
        "attack_name_counts": _counter_to_dict(attack_names),
        "bundle_kind_counts": _counter_to_dict(bundle_kinds),
        "source_counts": _counter_to_dict(sources),
        "domain_track_counts": _counter_to_dict(domain_track_pairs),
        "tasks_with_environment_snapshot": tasks_with_environment_snapshot,
        "tasks_with_content_artifacts": tasks_with_content_artifacts,
        "tasks_with_user_ground_truth_calls": tasks_with_ground_truth_calls,
        "tasks_with_injection_ground_truth_calls": tasks_with_injection_ground_truth_calls,
        "unique_user_task_count": len(user_task_ids),
        "unique_injection_task_count": len(injection_task_ids),
        "unique_attack_spec_count": len(attack_spec_ids),
    }


def build_benchmark_extraction_payload(
    *,
    benchmark,
    benchmark_version,
    extraction_track,
    source_tasks,
    tool_pool=None,
    extra_fields=None,
):
    resolved_tool_pool = list(tool_pool or build_tool_pool_from_source_tasks(source_tasks))
    payload = {
        "benchmark": benchmark,
        "benchmark_version": benchmark_version,
        "extraction_track": extraction_track,
        "source_task_count": len(list(source_tasks or [])),
        "source_tasks": list(source_tasks or []),
        "tool_pool_count": len(resolved_tool_pool),
        "tool_pool": resolved_tool_pool,
        "benchmark_accounting_summary": summarize_benchmark_accounting(
            source_tasks,
            tool_pool=resolved_tool_pool,
        ),
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload
