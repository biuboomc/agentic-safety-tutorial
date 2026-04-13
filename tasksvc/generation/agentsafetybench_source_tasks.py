import copy
import json
from pathlib import Path

from tasksvc.generation.benchmark_source_utils import (
    benchmark_slugify,
    dedupe_strings,
    make_tool,
    preview_text,
)


DEFAULT_AGENTSAFETYBENCH_VERSION = "released_data_main"
DEFAULT_AGENTSAFETYBENCH_ACCOUNTING_MODE = "release_manifest"
DEFAULT_AGENTDYN_ACCOUNTING_MODE = "derived_dialog_subset"
DEFAULT_AGENTSAFETYBENCH_ACCOUNTING_CLASS = "flat_bundle_benchmark"
DEFAULT_AGENTDYN_ACCOUNTING_CLASS = "reusable_attack_bundle_candidate"


def _load_released_rows(repo_path):
    data_path = Path(repo_path) / "data" / "released_data.json"
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {data_path}")
    return payload


def _load_env_tool_catalog(repo_path, env_name):
    env_path = Path(repo_path) / "environments" / f"{env_name}.json"
    if not env_path.exists():
        return []
    payload = json.loads(env_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    tools = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        tools.append(
            make_tool(
                name,
                item.get("description") or f"{env_name} benchmark tool",
                domain=env_name,
                parameters=item.get("parameters"),
                reads_state_keys=["environments"],
                writes_state_keys=["environments"] if name.startswith(("send_", "click_", "block_", "delete_", "create_", "write_")) else [],
                labels=["benchmark_env_tool"],
                metadata={"source_env_json": str(env_path)},
            )
        )
    return tools


def _collect_env_names(row):
    envs = row.get("environments") or []
    return dedupe_strings((env or {}).get("name") for env in envs if isinstance(env, dict))


def _infer_track(row):
    has_dialog = bool(row.get("dialog"))
    if has_dialog:
        return "attacked_dynamic"
    env_names = _collect_env_names(row)
    has_real_env = any(name for name in env_names)
    if not has_real_env:
        return "harm_query_text_only"
    return "harm_query_with_tools"


def _build_content_artifacts(environments):
    artifacts = []
    for env in environments or []:
        if not isinstance(env, dict):
            continue
        env_name = str(env.get("name") or "").strip()
        parameters = env.get("parameters") or {}
        if not isinstance(parameters, dict):
            continue
        for key, value in parameters.items():
            if isinstance(value, str) and value.strip():
                artifacts.append(
                    {
                        "state_path": f"environments.{env_name or 'unknown'}.parameters",
                        "resource_kind": "text_parameter",
                        "locator": f"{env_name or 'unknown'}::{key}",
                        "content_field": key,
                        "content_preview": preview_text(value),
                    }
                )
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                serialized = json.dumps(value[0], ensure_ascii=False)
                artifacts.append(
                    {
                        "state_path": f"environments.{env_name or 'unknown'}.parameters",
                        "resource_kind": "record_list",
                        "locator": f"{env_name or 'unknown'}::{key}",
                        "content_field": key,
                        "content_preview": preview_text(serialized),
                    }
                )
    return artifacts


def _dedupe_tool_pool(tools):
    pool = []
    seen = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        key = json.dumps(
            {
                "name": tool.get("name"),
                "domain": tool.get("domain"),
                "schema": tool.get("schema"),
                "reads": tool.get("simulator_requirements", {}).get("reads_state_keys"),
                "writes": tool.get("simulator_requirements", {}).get("writes_state_keys"),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        if key in seen:
            continue
        seen.add(key)
        pool.append(copy.deepcopy(tool))
    return pool


def build_agentsafetybench_tool_pool(source_tasks, *, benchmark_name="agent-safety-bench", benchmark_version=DEFAULT_AGENTSAFETYBENCH_VERSION):
    all_tools = []
    for item in source_tasks or []:
        if not isinstance(item, dict):
            continue
        all_tools.extend(item.get("tools") or [])
    return _dedupe_tool_pool(all_tools)


def _build_accounting_summary(
    rows,
    selected_rows,
    source_tasks,
    tool_pool,
    *,
    benchmark_name,
    benchmark_version,
    accounting_mode,
    selected_track,
):
    rows = [row for row in rows or [] if isinstance(row, dict)]
    selected_rows = [row for row in selected_rows or [] if isinstance(row, dict)]
    source_tasks = [task for task in source_tasks or [] if isinstance(task, dict)]
    raw_env_names = []
    selected_env_names = []
    track_counts = {}
    selected_track_counts = {}
    for row in rows:
        for env_name in _collect_env_names(row):
            if env_name not in raw_env_names:
                raw_env_names.append(env_name)
        track_name = _infer_track(row)
        track_counts[track_name] = track_counts.get(track_name, 0) + 1
    for row in selected_rows:
        for env_name in _collect_env_names(row):
            if env_name not in selected_env_names:
                selected_env_names.append(env_name)
        track_name = _infer_track(row)
        selected_track_counts[track_name] = selected_track_counts.get(track_name, 0) + 1
    bundle_kind_counts = {}
    for task in source_tasks:
        bundle_kind = str((task.get("metadata") or {}).get("bundle_kind") or "unspecified").strip()
        bundle_kind_counts[bundle_kind] = bundle_kind_counts.get(bundle_kind, 0) + 1
    return {
        "benchmark": benchmark_name,
        "benchmark_version": benchmark_version,
        "accounting_mode": accounting_mode,
        "accounting_class": (
            DEFAULT_AGENTDYN_ACCOUNTING_CLASS if benchmark_name == "agentdyn" else DEFAULT_AGENTSAFETYBENCH_ACCOUNTING_CLASS
        ),
        "inventory_source": "data/released_data.json",
        "selected_track": selected_track,
        "raw_row_count": len(rows),
        "selected_row_count": len(selected_rows),
        "source_task_count": len(source_tasks),
        "raw_env_count": len(raw_env_names),
        "selected_env_count": len(selected_env_names),
        "tool_count": len(tool_pool or []),
        "dialog_row_count": sum(1 for row in rows if row.get("dialog")),
        "track_counts": track_counts,
        "selected_track_counts": selected_track_counts,
        "bundle_kind_counts": bundle_kind_counts,
    }


def build_agentsafetybench_source_tasks_payload(
    repo_path,
    *,
    benchmark_name="agent-safety-bench",
    benchmark_version=DEFAULT_AGENTSAFETYBENCH_VERSION,
    track="all",
    limit=None,
    accounting_mode=DEFAULT_AGENTSAFETYBENCH_ACCOUNTING_MODE,
    row_filter=None,
):
    rows = _load_released_rows(repo_path)
    selected_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row_filter is not None and not row_filter(row):
            continue
        inferred_track = _infer_track(row)
        if inferred_track == "excluded":
            continue
        if track != "all" and inferred_track != track:
            continue
        selected_rows.append(row)
    source_tasks = _extract_agentsafetybench_source_tasks_from_rows(
        repo_path,
        selected_rows,
        benchmark_name=benchmark_name,
        benchmark_version=benchmark_version,
        track=track,
        limit=limit,
        accounting_mode=accounting_mode,
    )
    tool_pool = build_agentsafetybench_tool_pool(
        source_tasks,
        benchmark_name=benchmark_name,
        benchmark_version=benchmark_version,
    )
    return {
        "benchmark": benchmark_name,
        "benchmark_version": benchmark_version,
        "extraction_track": track,
        "accounting_summary": _build_accounting_summary(
            rows,
            selected_rows,
            source_tasks,
            tool_pool,
            benchmark_name=benchmark_name,
            benchmark_version=benchmark_version,
            accounting_mode=accounting_mode,
            selected_track=track,
        ),
        "tool_pool": tool_pool,
        "source_task_count": len(source_tasks),
        "source_tasks": source_tasks,
    }


def build_agentdyn_source_tasks_payload(
    repo_path,
    *,
    benchmark_version=DEFAULT_AGENTSAFETYBENCH_VERSION,
    track="attacked_dynamic",
    limit=None,
):
    return build_agentsafetybench_source_tasks_payload(
        repo_path,
        benchmark_name="agentdyn",
        benchmark_version=benchmark_version,
        track=track,
        limit=limit,
        accounting_mode=DEFAULT_AGENTDYN_ACCOUNTING_MODE,
        row_filter=lambda row: bool(row.get("dialog")),
    )


def _extract_agentsafetybench_source_tasks_from_rows(
    repo_path,
    rows,
    *,
    benchmark_name,
    benchmark_version,
    track,
    limit,
    accounting_mode,
):
    source_tasks = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        inferred_track = _infer_track(row)
        environments = copy.deepcopy(row.get("environments") or [])
        env_names = _collect_env_names(row)
        domain = env_names[0] if env_names else "text_only"
        tools = []
        for env_name in env_names:
            tools.extend(_load_env_tool_catalog(repo_path, env_name))
        bundle_kind = "attacked_dynamic" if inferred_track == "attacked_dynamic" else "utility_clean"
        source_task_id = f"{benchmark_slugify(benchmark_name)}__{benchmark_slugify(domain)}__task_{row.get('id')}"
        metadata = {
            "benchmark": benchmark_name,
            "benchmark_version": benchmark_version,
            "suite": domain,
            "evaluation_track": inferred_track,
            "accounting_mode": accounting_mode,
            "accounting_class": (
                DEFAULT_AGENTDYN_ACCOUNTING_CLASS if benchmark_name == "agentdyn" else DEFAULT_AGENTSAFETYBENCH_ACCOUNTING_CLASS
            ),
            "inventory_mode": "release_manifest_curated_subset" if benchmark_name == "agentdyn" else "release_manifest",
            "inventory_source": "data/released_data.json",
            "bundle_kind": bundle_kind,
            "benchmark_task_id": row.get("id"),
            "row_origin": "released_data.json",
            "risks": copy.deepcopy(row.get("risks") or []),
            "failure_modes": row.get("failure_modes"),
            "fulfillable": row.get("fulfillable"),
            "prior_dialog": copy.deepcopy(row.get("dialog") or []),
            "benchmark_semantics": {
                "task_source": "data/released_data.json",
                "env_manager_source": "evaluation/eval.py + environments/EnvManager.py",
                "safety_judge_source": "score/eval_with_shield.py",
                "note": "Benchmark safety outcome is judged downstream over full trajectory; extractor preserves task/env/dialog evidence only.",
            },
            "user_ground_truth_calls": [],
            "injection_ground_truth_calls": [],
        }
        source_tasks.append(
            {
                "source": benchmark_name,
                "source_task_id": source_task_id,
                "domain": domain,
                "task": str(row.get("instruction") or "").strip(),
                "persona": "tool-using assistant",
                "notes": "Agent-SafetyBench task normalized from released_data.json.",
                "metadata": metadata,
                "environment_snapshot": {"environments": environments},
                "content_artifacts": _build_content_artifacts(environments),
                "tools": tools or [
                    make_tool(
                        "respond",
                        "Text-only benchmark task without explicit tools.",
                        domain=domain,
                        parameters={"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
                        labels=["text_only"],
                    )
                ],
            }
        )
        if limit is not None and len(source_tasks) >= int(limit):
            break
    return source_tasks


def extract_agentsafetybench_source_tasks(
    repo_path,
    *,
    benchmark_name="agent-safety-bench",
    benchmark_version=DEFAULT_AGENTSAFETYBENCH_VERSION,
    track="all",
    limit=None,
):
    return build_agentsafetybench_source_tasks_payload(
        repo_path,
        benchmark_name=benchmark_name,
        benchmark_version=benchmark_version,
        track=track,
        limit=limit,
    )["source_tasks"]


def extract_agentdyn_source_tasks(
    repo_path,
    *,
    benchmark_version=DEFAULT_AGENTSAFETYBENCH_VERSION,
    track="attacked_dynamic",
    limit=None,
):
    return build_agentdyn_source_tasks_payload(
        repo_path,
        benchmark_version=benchmark_version,
        track=track,
        limit=limit,
    )["source_tasks"]
