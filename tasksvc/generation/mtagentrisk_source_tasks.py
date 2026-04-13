import csv
import copy
import json
from pathlib import Path

from tasksvc.generation.benchmark_source_utils import (
    make_tool,
    normalize_source_task_id,
    preview_text,
)


MT_AGENTRISK_NAME = "mtagentrisk"
DEFAULT_MTAGENTRISK_VERSION = "toolshield_repo"
DEFAULT_MTAGENTRISK_ACCOUNTING_CLASS = "manifest_driven_benchmark"
_BENIGN_SPECS_FILENAME = "benign_task_specs.json"
_SINGLE_MANIFEST_NAME = "single_dataset.csv"
_MULTI_MANIFEST_NAME = "multi_dataset.csv"


def _read_text(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _read_json(path):
    return json.loads(_read_text(path))


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_simple_yaml_list(path):
    items = []
    for line in _read_text(path).splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return items


def _first_nonempty(mapping, keys, default=None):
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
        elif value != "":
            return value
    return default


def _candidate_existing_paths(root, *relative_paths):
    for rel_path in relative_paths:
        candidate = Path(rel_path)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        joined = root / rel_path
        if joined.exists():
            return joined
    return None


def _find_single_manifest_root(root):
    direct = _candidate_existing_paths(root, _SINGLE_MANIFEST_NAME, f"MT-AgentRisk/{_SINGLE_MANIFEST_NAME}")
    if direct is not None:
        return direct
    for candidate in (
        root / "MT-AgentRisk" / "workspaces",
        root / "workspaces",
        root,
    ):
        if (candidate / _SINGLE_MANIFEST_NAME).exists():
            return candidate
    return None


def _find_multi_manifest_root(root):
    direct = _candidate_existing_paths(root, _MULTI_MANIFEST_NAME, f"MT-AgentRisk/{_MULTI_MANIFEST_NAME}")
    if direct is not None:
        return direct
    for candidate in (
        root / "MT-AgentRisk" / "workspaces",
        root / "workspaces",
        root,
    ):
        if (candidate / _MULTI_MANIFEST_NAME).exists():
            return candidate
    return None


def _find_benign_root(root):
    candidates = [
        root / "workspaces" / "benign_tasks",
        root / "MT-AgentRisk" / "workspaces" / "benign_tasks",
        root / "benign_tasks",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_inventory_roots(repo_path):
    root = Path(repo_path)
    single_manifest_root = _find_single_manifest_root(root)
    multi_manifest_root = _find_multi_manifest_root(root)
    benign_root = _find_benign_root(root)
    dataset_root = None
    for candidate in (
        root / "MT-AgentRisk" / "workspaces",
        root / "workspaces",
        root,
    ):
        if candidate.exists():
            dataset_root = candidate
            break
    if dataset_root is None and single_manifest_root is None and multi_manifest_root is None and benign_root is None:
        raise ValueError(f"Could not locate MT-AgentRisk inventory under {repo_path}")
    return {
        "root": root,
        "dataset_root": dataset_root,
        "single_manifest_root": single_manifest_root,
        "multi_manifest_root": multi_manifest_root,
        "benign_root": benign_root,
    }


def _load_manifest_rows(root, manifest_name):
    manifest_path = _candidate_existing_paths(root, manifest_name, f"MT-AgentRisk/{manifest_name}")
    if manifest_path is None:
        return None, []
    rows = _read_csv(manifest_path)
    return manifest_path, rows


def _load_function_sets_from_benign_specs(benign_root):
    function_sets = {}
    if benign_root is None:
        return function_sets
    for specs_path in benign_root.glob("benign_tasks_*/benign_task_specs.json"):
        category = specs_path.parent.name.replace("benign_tasks_", "")
        try:
            rows = _read_json(specs_path)
        except Exception:
            continue
        names = []
        for row in rows or []:
            for function_name in row.get("functions") or []:
                function_name = str(function_name or "").strip()
                if function_name and function_name not in names:
                    names.append(function_name)
        function_sets[category] = names
    return function_sets


def _iter_manifest_function_names(rows, preferred_keys):
    names = []
    for row in rows or []:
        for key in preferred_keys:
            raw_value = row.get(key)
            if raw_value is None:
                continue
            if isinstance(raw_value, str):
                values = [part.strip() for part in raw_value.split(",")]
            elif isinstance(raw_value, list):
                values = [str(part).strip() for part in raw_value]
            else:
                values = [str(raw_value).strip()]
            for value in values:
                if value and value not in names:
                    names.append(value)
    return names


def _category_tools(category, function_names, *, source_note, inventory_source):
    names = list(function_names or [])
    if not names:
        names = [f"mcp-{category}"]
    tools = []
    for name in names:
        tools.append(
            make_tool(
                name,
                f"MT-AgentRisk benchmark tool or function for the {category} workspace.",
                domain=category,
                labels=["utility"],
                reads_state_keys=[category],
                metadata={
                    "tool_recovery": source_note,
                    "category": category,
                    "inventory_source": inventory_source,
                },
            )
        )
    return tools


def _tool_pool_from_records(records):
    seen = set()
    tools = []
    for record in records or []:
        for tool in record.get("tools") or []:
            name = tool.get("name")
            domain = tool.get("domain")
            key = (domain, name)
            if not name or key in seen:
                continue
            seen.add(key)
            tools.append(copy.deepcopy(tool))
    return tools


def _build_accounting_summary(records, *, inventory_mode):
    inventory_sources = []
    for record in records or []:
        source = str(record.get("metadata", {}).get("inventory_source") or "").strip()
        if source and source not in inventory_sources:
            inventory_sources.append(source)
    summary = {
        "env_count": len(
            {
                record.get("metadata", {}).get("suite")
                for record in records or []
                if record.get("metadata", {}).get("suite")
            }
        ),
        "tool_count": len(_tool_pool_from_records(records)),
        "user_task_count": sum(1 for record in records or [] if record.get("metadata", {}).get("task_kind") == "benign"),
        "injection_task_count": sum(
            1
            for record in records or []
            if record.get("metadata", {}).get("task_kind") in {"single_turn", "multi_turn"}
        ),
        "benign_bundle_count": sum(
            1 for record in records or [] if record.get("metadata", {}).get("evaluation_track") == "utility_clean"
        ),
        "unsafe_bundle_count": sum(
            1 for record in records or [] if record.get("metadata", {}).get("evaluation_track") != "utility_clean"
        ),
        "accounting_mode": "canonical_contract" if inventory_mode != "filesystem_fallback" else "repo_raw",
        "accounting_class": DEFAULT_MTAGENTRISK_ACCOUNTING_CLASS,
        "inventory_mode": inventory_mode,
        "inventory_sources": inventory_sources,
    }
    return summary


def _attach_accounting_metadata(record, *, inventory_mode, inventory_source):
    metadata = record.setdefault("metadata", {})
    metadata["accounting_mode"] = "canonical_contract" if inventory_mode != "filesystem_fallback" else "repo_raw"
    metadata["accounting_class"] = DEFAULT_MTAGENTRISK_ACCOUNTING_CLASS
    metadata["inventory_mode"] = inventory_mode
    metadata["inventory_source"] = inventory_source
    metadata["bundle_kind"] = "benign_bundle" if metadata.get("evaluation_track") == "utility_clean" else "unsafe_bundle"
    return record


def _resolve_manifest_path_from_row(root, row, kind):
    path_keys = [
        "task_dir",
        "task_path",
        "path",
        "dir",
        "folder",
        "workspace_dir",
        "task_folder",
        "task_root",
    ]
    raw_path = _first_nonempty(row, path_keys)
    if raw_path:
        candidate = Path(str(raw_path))
        if candidate.is_absolute() and candidate.exists():
            return candidate
        joined = root / str(raw_path)
        if joined.exists():
            return joined
    category = _first_nonempty(row, ["category", "workspace", "workspace_name", "suite", "tool", "domain"], "general")
    task_name = _first_nonempty(row, ["task_name", "name", "task", "title", "id", "task_id"], "unknown")
    task_number = _first_nonempty(row, ["task_number", "task_no", "number", "index"], None)
    if kind == "single":
        path_candidates = []
        if task_number is not None:
            path_candidates.extend(
                [
                    root / category / "single-turn-tasks" / f"task.{task_number}",
                    root / category / "single-turn-tasks" / task_name,
                ]
            )
        path_candidates.extend(
            [
                root / category / "single-turn-tasks" / task_name,
                root / category / "single-turn-tasks" / str(task_name).replace(" ", "_"),
            ]
        )
    else:
        path_candidates = []
        if task_number is not None:
            path_candidates.extend(
                [
                    root / category / "multi-turn-tasks" / f"task.{task_number}",
                    root / category / "multi-turn-tasks" / task_name,
                ]
            )
        path_candidates.extend(
            [
                root / category / "multi-turn-tasks" / task_name,
                root / category / "multi-turn-tasks" / str(task_name).replace(" ", "_"),
            ]
        )
    for candidate in path_candidates:
        if candidate.exists():
            return candidate
    return None


def _single_turn_task_record(task_dir, category, function_names, *, inventory_mode, inventory_source, row=None):
    task_path = task_dir / "task.md"
    task_text = _read_text(task_path).strip() if task_path.exists() else str(_first_nonempty(row or {}, ["task", "instruction", "prompt", "description"], "")).strip()
    dependencies = _read_simple_yaml_list(task_dir / "utils" / "dependencies.yml") if (task_dir / "utils" / "dependencies.yml").exists() else []
    workspace_dir = task_dir / "workspace"
    tools = _category_tools(
        category,
        function_names,
        source_note="manifest_or_benign_function_set",
        inventory_source=inventory_source,
    )
    record = {
        "source": MT_AGENTRISK_NAME,
        "source_task_id": normalize_source_task_id(
            MT_AGENTRISK_NAME,
            category,
            "single",
            _first_nonempty(row or {}, ["task_name", "name", "task_id", "id"], task_dir.name),
        ),
        "domain": category,
        "task": task_text,
        "persona": "tool-using coding agent",
        "notes": "MT-AgentRisk single-turn attacked task extracted from manifest or task files.",
        "metadata": {
            "benchmark": MT_AGENTRISK_NAME,
            "benchmark_version": DEFAULT_MTAGENTRISK_VERSION,
            "suite": category,
            "evaluation_track": "attacked_single_turn",
            "task_name": _first_nonempty(row or {}, ["task_name", "name", "task_id", "id"], task_dir.name),
            "task_kind": "single_turn",
            "native_tool_dependencies": dependencies,
            "inventory_mode": inventory_mode,
            "inventory_source": inventory_source,
            "benchmark_semantics": {
                "task_source": "MT-AgentRisk single-turn manifest or task.md",
                "notes": "Static extraction preserves task text, dependency list, and source path information.",
            },
        },
        "environment_snapshot": {
            "task_dir": str(task_dir),
            "workspace_dir": str(workspace_dir) if workspace_dir.exists() else None,
            "dependencies": dependencies,
            "inventory_source": inventory_source,
        },
        "content_artifacts": [
            {
                "resource_kind": "task_markdown",
                "locator": str(task_path),
                "content_field": "body",
                "content_preview": preview_text(task_text),
            }
        ],
        "tools": tools,
    }
    return _attach_accounting_metadata(record, inventory_mode=inventory_mode, inventory_source=inventory_source)


def _multi_turn_task_record(task_dir, category, function_names, *, inventory_mode, inventory_source, row=None):
    turns_path = task_dir / "turns.yml"
    turns_text = _read_text(turns_path) if turns_path.exists() else str(_first_nonempty(row or {}, ["turns", "task", "prompt", "description"], "")).strip()
    turn_files = sorted(task_dir.glob("task-turn-*.md"))
    turn_texts = [path.read_text(encoding="utf-8", errors="ignore").strip() for path in turn_files]
    dependencies = _read_simple_yaml_list(task_dir / "utils" / "dependencies.yml") if (task_dir / "utils" / "dependencies.yml").exists() else []
    content_artifacts = [
        {
            "resource_kind": "turn_manifest",
            "locator": str(turns_path),
            "content_field": "body",
            "content_preview": preview_text(turns_text),
        }
    ]
    for path, text in zip(turn_files, turn_texts):
        content_artifacts.append(
            {
                "resource_kind": "turn_instruction",
                "locator": str(path),
                "content_field": "body",
                "content_preview": preview_text(text),
            }
        )
    attack_chain_path = task_dir / "attack_chain.json"
    task_name = _first_nonempty(row or {}, ["task_name", "name", "task_id", "id"], task_dir.name)
    metadata = {
        "benchmark": MT_AGENTRISK_NAME,
        "benchmark_version": DEFAULT_MTAGENTRISK_VERSION,
        "suite": category,
        "evaluation_track": "attacked_multi_turn",
        "task_name": task_name,
        "task_kind": "multi_turn",
        "native_tool_dependencies": dependencies,
        "turn_instruction_files": [path.name for path in turn_files],
        "inventory_mode": inventory_mode,
        "inventory_source": inventory_source,
        "benchmark_semantics": {
            "task_source": "MT-AgentRisk multi-turn manifest or turns.yml",
            "notes": "Multi-turn task preserves ordered turn prompts and dependency list for later scenario-aware replay.",
        },
    }
    if attack_chain_path.exists():
        try:
            metadata["attack_chain"] = _read_json(attack_chain_path)
        except Exception:
            metadata["attack_chain_path"] = str(attack_chain_path)
    record = {
        "source": MT_AGENTRISK_NAME,
        "source_task_id": normalize_source_task_id(MT_AGENTRISK_NAME, category, "multi", task_name),
        "domain": category,
        "task": "\n\n".join(turn_texts).strip(),
        "persona": "tool-using coding agent",
        "notes": "MT-AgentRisk multi-turn attacked task extracted from manifest or turn files.",
        "metadata": metadata,
        "environment_snapshot": {
            "task_dir": str(task_dir),
            "dependencies": dependencies,
            "turn_count": len(turn_files),
            "inventory_source": inventory_source,
        },
        "content_artifacts": content_artifacts,
        "tools": _category_tools(
            category,
            function_names,
            source_note="manifest_or_benign_function_set",
            inventory_source=inventory_source,
        ),
    }
    return _attach_accounting_metadata(record, inventory_mode=inventory_mode, inventory_source=inventory_source)


def _benign_task_record(task_dir, category, spec_row, *, inventory_mode, inventory_source, function_names=None):
    task_brief = str(spec_row.get("task_brief") or spec_row.get("task") or "").strip()
    names = [str(name or "").strip() for name in (function_names or spec_row.get("functions") or []) if str(name or "").strip()]
    tools = [
        make_tool(
            name,
            f"Benign {category} benchmark function.",
            domain=category,
            labels=["utility"],
            reads_state_keys=[category],
            metadata={
                "tool_recovery": inventory_mode,
                "category": category,
                "inventory_source": inventory_source,
            },
        )
        for name in names
    ]
    task_number = _first_nonempty(spec_row, ["task_number", "task_no", "number", "index"], None)
    task_id = _first_nonempty(spec_row, ["task_id", "id", "name"], f"task_{task_number}")
    record = {
        "source": MT_AGENTRISK_NAME,
        "source_task_id": normalize_source_task_id(MT_AGENTRISK_NAME, category, "benign", task_id),
        "domain": category,
        "task": task_brief,
        "persona": "tool-using coding agent",
        "notes": "ToolShield benign seed task extracted from benchmark manifest or specs.",
        "metadata": {
            "benchmark": MT_AGENTRISK_NAME,
            "benchmark_version": DEFAULT_MTAGENTRISK_VERSION,
            "suite": category,
            "evaluation_track": "utility_clean",
            "task_number": task_number,
            "title": spec_row.get("title"),
            "difficulty": spec_row.get("difficulty"),
            "inventory_mode": inventory_mode,
            "inventory_source": inventory_source,
            "benchmark_semantics": {
                "task_source": "benign task manifest or benign_task_specs.json",
                "notes": "Benign seed tasks are provided as static task briefs plus named functions.",
            },
        },
        "environment_snapshot": {
            "task_dir": str(task_dir) if task_dir and task_dir.exists() else None,
            "tool_category": category,
            "inventory_source": inventory_source,
        },
        "content_artifacts": [
            {
                "resource_kind": "task_brief",
                "locator": f"{category}:task_{task_id}",
                "content_field": "task_brief",
                "content_preview": preview_text(task_brief),
            }
        ],
        "tools": tools,
    }
    return _attach_accounting_metadata(record, inventory_mode=inventory_mode, inventory_source=inventory_source)


def _load_benign_specs_from_manifests(benign_root):
    records = []
    if benign_root is None:
        return records
    for spec_path in sorted(benign_root.glob("benign_tasks_*/benign_task_specs.json")):
        category = spec_path.parent.name.replace("benign_tasks_", "")
        try:
            rows = _read_json(spec_path)
        except Exception:
            continue
        function_names = _iter_manifest_function_names(rows, ["functions", "function_names", "tools"])
        for row in rows or []:
            task_dir = spec_path.parent / f"task.{_first_nonempty(row, ['task_number', 'task_no', 'number', 'index'], '0')}"
            records.append(
                _benign_task_record(
                    task_dir,
                    category,
                    row,
                    inventory_mode="manifest",
                    inventory_source=str(spec_path),
                    function_names=function_names,
                )
            )
    return records


def _record_inventory_mode(records):
    modes = []
    for record in records or []:
        mode = record.get("metadata", {}).get("inventory_mode")
        if mode and mode not in modes:
            modes.append(mode)
    if not modes:
        return "filesystem_fallback"
    if len(modes) == 1:
        return modes[0]
    return "mixed"


def _load_benign_specs_from_csv(root):
    manifest_path, rows = _load_manifest_rows(root, "benign_task_specs.csv")
    if manifest_path is None:
        return []
    records = []
    grouped = {}
    for row in rows or []:
        category = str(_first_nonempty(row, ["category", "workspace", "suite", "tool", "domain"], "general")).strip()
        grouped.setdefault(category, []).append(row)
    for category, category_rows in grouped.items():
        function_names = _iter_manifest_function_names(category_rows, ["functions", "function_names", "tools", "function"])
        for row in category_rows:
            task_dir = _resolve_manifest_path_from_row(root, row, "benign")
            records.append(
                _benign_task_record(
                    task_dir or root,
                    category,
                    row,
                    inventory_mode="csv_manifest",
                    inventory_source=str(manifest_path),
                    function_names=function_names,
                )
            )
    return records


def _load_single_turn_from_manifest(root, function_names):
    manifest_path, rows = _load_manifest_rows(root, _SINGLE_MANIFEST_NAME)
    if manifest_path is None:
        return []
    records = []
    for row in rows or []:
        category = str(_first_nonempty(row, ["category", "workspace", "suite", "tool", "domain"], "general")).strip()
        task_dir = _resolve_manifest_path_from_row(root, row, "single")
        if task_dir is None:
            continue
        records.append(
            _single_turn_task_record(
                task_dir,
                category,
                function_names.get(category, []),
                inventory_mode="csv_manifest",
                inventory_source=str(manifest_path),
                row=row,
            )
        )
    return records


def _load_multi_turn_from_manifest(root, function_names):
    manifest_path, rows = _load_manifest_rows(root, _MULTI_MANIFEST_NAME)
    if manifest_path is None:
        return []
    records = []
    for row in rows or []:
        category = str(_first_nonempty(row, ["category", "workspace", "suite", "tool", "domain"], "general")).strip()
        task_dir = _resolve_manifest_path_from_row(root, row, "multi")
        if task_dir is None:
            continue
        records.append(
            _multi_turn_task_record(
                task_dir,
                category,
                function_names.get(category, []),
                inventory_mode="csv_manifest",
                inventory_source=str(manifest_path),
                row=row,
            )
        )
    return records


def _fallback_dataset_scan(dataset_root, function_names, *, track):
    records = []
    categories = [
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name not in {"benign_tasks", "Subset_100"}
    ]
    if track in {"all", "utility_clean"}:
        benign_root = dataset_root / "benign_tasks"
        if benign_root.exists():
            for specs_path in sorted(benign_root.glob("benign_tasks_*/benign_task_specs.json")):
                category = specs_path.parent.name.replace("benign_tasks_", "")
                specs = _read_json(specs_path)
                names = _iter_manifest_function_names(specs, ["functions", "function_names", "tools"])
                for row in specs or []:
                    task_dir = specs_path.parent / f"task.{_first_nonempty(row, ['task_number', 'task_no', 'number', 'index'], '0')}"
                    records.append(
                        _benign_task_record(
                            task_dir,
                            category,
                            row,
                            inventory_mode="filesystem_fallback",
                            inventory_source=str(specs_path),
                            function_names=names,
                        )
                    )
    if track in {"all", "attacked_single_turn"}:
        for category_dir in sorted(categories):
            category = category_dir.name.replace("-", "_")
            single_root = category_dir / "single-turn-tasks"
            if not single_root.exists():
                continue
            for task_dir in sorted(path for path in single_root.iterdir() if path.is_dir() and (path / "task.md").exists()):
                records.append(
                    _single_turn_task_record(
                        task_dir,
                        category,
                        function_names.get(category, []),
                        inventory_mode="filesystem_fallback",
                        inventory_source=str(single_root),
                    )
                )
    if track in {"all", "attacked_multi_turn"}:
        for category_dir in sorted(categories):
            category = category_dir.name.replace("-", "_")
            multi_root = category_dir / "multi-turn-tasks"
            if not multi_root.exists():
                continue
            for task_dir in sorted(path for path in multi_root.iterdir() if path.is_dir() and (path / "turns.yml").exists()):
                records.append(
                    _multi_turn_task_record(
                        task_dir,
                        category,
                        function_names.get(category, []),
                        inventory_mode="filesystem_fallback",
                        inventory_source=str(multi_root),
                    )
                )
    return records


def extract_mtagentrisk_source_tasks(repo_path, *, benchmark_version=DEFAULT_MTAGENTRISK_VERSION, track="all", limit=None):
    inventory = _resolve_inventory_roots(repo_path)
    function_sets = _load_function_sets_from_benign_specs(inventory["benign_root"])
    normalized_track = str(track or "all").strip().lower()
    if normalized_track not in {"all", "utility_clean", "attacked_single_turn", "attacked_multi_turn"}:
        raise ValueError(f"Unsupported MT-AgentRisk track: {track}")

    tasks = []
    inventory_sources = []

    def append(item):
        tasks.append(item)
        return limit is not None and len(tasks) >= int(limit)

    # Prefer manifests when present, then fall back to filesystem scanning.
    if normalized_track in {"all", "utility_clean"}:
        manifest_records = _load_benign_specs_from_manifests(inventory["benign_root"])
        if not manifest_records:
            manifest_records = _load_benign_specs_from_csv(inventory["root"])
        if manifest_records:
            inventory_sources.extend(sorted({record["metadata"]["inventory_source"] for record in manifest_records}))
            for record in manifest_records:
                record["metadata"]["benchmark_version"] = benchmark_version
                if append(record):
                    return tasks

    if normalized_track in {"all", "attacked_single_turn"}:
        manifest_records = _load_single_turn_from_manifest(inventory["root"], function_sets)
        if manifest_records:
            inventory_sources.extend(sorted({record["metadata"]["inventory_source"] for record in manifest_records}))
            for record in manifest_records:
                record["metadata"]["benchmark_version"] = benchmark_version
                if append(record):
                    return tasks

    if normalized_track in {"all", "attacked_multi_turn"}:
        manifest_records = _load_multi_turn_from_manifest(inventory["root"], function_sets)
        if manifest_records:
            inventory_sources.extend(sorted({record["metadata"]["inventory_source"] for record in manifest_records}))
            for record in manifest_records:
                record["metadata"]["benchmark_version"] = benchmark_version
                if append(record):
                    return tasks

    if not tasks:
        dataset_root = inventory["dataset_root"]
        if dataset_root is None:
            raise ValueError(f"Could not locate MT-AgentRisk workspaces under {repo_path}")
        fallback_records = _fallback_dataset_scan(dataset_root, function_sets, track=normalized_track)
        for record in fallback_records:
            record["metadata"]["benchmark_version"] = benchmark_version
            if append(record):
                return tasks

    if tasks:
        inventory_mode = _record_inventory_mode(tasks)
        accounting_summary = _build_accounting_summary(tasks, inventory_mode=inventory_mode)
        for record in tasks:
            record["metadata"]["accounting_summary"] = accounting_summary
            record["metadata"]["inventory_mode"] = inventory_mode
            if inventory_sources:
                record["metadata"]["inventory_sources"] = list(dict.fromkeys(inventory_sources))
    return tasks


def build_mtagentrisk_tool_pool(repo_path, *, benchmark_version=DEFAULT_MTAGENTRISK_VERSION, track="all"):
    records = extract_mtagentrisk_source_tasks(repo_path, benchmark_version=benchmark_version, track=track)
    tools = _tool_pool_from_records(records)
    inventory_mode = _record_inventory_mode(records)
    return {
        "benchmark": MT_AGENTRISK_NAME,
        "benchmark_version": benchmark_version,
        "extraction_track": track,
        "tool_count": len(tools),
        "tool_pool": tools,
        "accounting_summary": _build_accounting_summary(records, inventory_mode=inventory_mode),
    }


def build_mtagentrisk_source_task_payload(repo_path, *, benchmark_version=DEFAULT_MTAGENTRISK_VERSION, track="all", limit=None):
    source_tasks = extract_mtagentrisk_source_tasks(repo_path, benchmark_version=benchmark_version, track=track, limit=limit)
    tool_pool = _tool_pool_from_records(source_tasks)
    inventory_mode = _record_inventory_mode(source_tasks)
    accounting_summary = _build_accounting_summary(source_tasks, inventory_mode=inventory_mode)
    return {
        "benchmark": MT_AGENTRISK_NAME,
        "benchmark_version": benchmark_version,
        "extraction_track": track,
        "source_task_count": len(source_tasks),
        "tool_pool_count": len(tool_pool),
        "accounting_summary": accounting_summary,
        "source_tasks": source_tasks,
        "tool_pool": tool_pool,
    }
