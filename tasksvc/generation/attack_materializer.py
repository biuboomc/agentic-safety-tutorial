import copy


SUPPORTED_INJECTION_SURFACES = {
    "query",
    "environment_content",
    "tool_return",
}


def _default_benchmark_wrapper_text(attack_spec):
    wrapper_metadata = copy.deepcopy(attack_spec.get("attack_wrapper_metadata") or {})
    benchmark_wrappers = copy.deepcopy(wrapper_metadata.get("benchmark_wrapper_text_by_name") or {})
    direct = benchmark_wrappers.get("direct")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    return ""


def _slugify(value):
    text = str(value or "").strip().lower()
    chars = []
    for ch in text:
        chars.append(ch if ch.isalnum() else "_")
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "attack"


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return copy.deepcopy(value)
    return [copy.deepcopy(value)]


def _path_segments(dotted_path):
    if isinstance(dotted_path, list):
        segments = []
        for segment in dotted_path:
            if isinstance(segment, int):
                segments.append(segment)
                continue
            text = str(segment).strip()
            if not text:
                continue
            if text.isdigit():
                segments.append(int(text))
            else:
                segments.append(text)
        return segments
    segments = []
    for segment in str(dotted_path or "").split("."):
        text = str(segment).strip()
        if not text:
            continue
        if text.isdigit():
            segments.append(int(text))
        else:
            segments.append(text)
    return segments


def _is_subset_match(expected, actual):
    if not isinstance(expected, dict):
        return False
    if not isinstance(actual, dict):
        return False
    for key, value in expected.items():
        if key not in actual or actual[key] != value:
            return False
    return True


def _set_path_value(container, dotted_path, value):
    path = _path_segments(dotted_path)
    if not path:
        return container
    current = container
    for index, segment in enumerate(path[:-1]):
        next_segment = path[index + 1]
        if isinstance(current, dict):
            if segment not in current:
                current[segment] = [] if isinstance(next_segment, int) else {}
            elif isinstance(next_segment, int) and not isinstance(current[segment], list):
                current[segment] = []
            elif not isinstance(next_segment, int) and not isinstance(current[segment], dict):
                current[segment] = {}
            current = current[segment]
            continue
        if isinstance(current, list):
            if not isinstance(segment, int):
                raise ValueError(f"Cannot descend into list with non-index segment {segment!r}.")
            while len(current) <= segment:
                current.append(None)
            if current[segment] is None or (
                isinstance(next_segment, int) and not isinstance(current[segment], list)
            ) or (not isinstance(next_segment, int) and not isinstance(current[segment], dict)):
                current[segment] = [] if isinstance(next_segment, int) else {}
            current = current[segment]
            continue
        raise ValueError(f"Cannot descend into non-dict path segment {segment!r}.")
    leaf = path[-1]
    if isinstance(current, dict):
        current[leaf] = copy.deepcopy(value)
        return container
    if isinstance(current, list):
        if not isinstance(leaf, int):
            raise ValueError(f"Cannot assign non-index leaf path {leaf!r} into list container.")
        while len(current) <= leaf:
            current.append(None)
        current[leaf] = copy.deepcopy(value)
        return container
    raise ValueError(f"Cannot assign leaf path {leaf!r} into non-container.")
    return container


def _get_path_value(container, dotted_path):
    path = _path_segments(dotted_path)
    current = container
    for segment in path:
        if isinstance(current, dict):
            if segment not in current:
                return None
            current = current[segment]
            continue
        if isinstance(current, list):
            if not isinstance(segment, int) or segment < 0 or segment >= len(current):
                return None
            current = current[segment]
            continue
        return None
    return current


def normalize_attack_spec(attack_spec):
    if not isinstance(attack_spec, dict):
        raise ValueError("Each attack spec must be a dict.")
    attack_spec_id = str(
        attack_spec.get("attack_spec_id")
        or attack_spec.get("id")
        or f"attack_{_slugify(attack_spec.get('attack_name') or attack_spec.get('name') or '')}"
    ).strip()
    attack_name = str(attack_spec.get("attack_name") or attack_spec.get("name") or attack_spec_id).strip()
    injection_surface = str(attack_spec.get("injection_surface") or "query").strip()
    if injection_surface not in SUPPORTED_INJECTION_SURFACES:
        raise ValueError(
            f"Unsupported injection_surface {injection_surface!r}; expected one of {sorted(SUPPORTED_INJECTION_SURFACES)}."
        )
    injection_candidates = _ensure_list(attack_spec.get("injection_candidates"))
    normalized_candidates = []
    for index, candidate in enumerate(injection_candidates, start=1):
        if not isinstance(candidate, dict):
            raise ValueError("Each injection candidate must be a dict.")
        candidate_copy = copy.deepcopy(candidate)
        candidate_copy.setdefault("candidate_id", f"{injection_surface}_{index}")
        candidate_copy.setdefault("injection_surface", injection_surface)
        normalized_candidates.append(candidate_copy)

    concrete_injections = _ensure_list(attack_spec.get("concrete_injections"))
    injection_template = attack_spec.get("injection_template")
    metadata = copy.deepcopy(attack_spec.get("metadata") or {})
    wrapper_metadata = copy.deepcopy(attack_spec.get("attack_wrapper_metadata") or {})
    risk_success_rule_source = str(attack_spec.get("risk_success_rule_source") or "")

    return {
        "attack_spec_id": attack_spec_id,
        "attack_name": attack_name,
        "domain": str(attack_spec.get("domain") or metadata.get("suite") or metadata.get("domain") or "").strip(),
        "injection_surface": injection_surface,
        "injection_candidates": normalized_candidates,
        "injection_template": copy.deepcopy(injection_template),
        "concrete_injections": concrete_injections,
        "risk_success_rule_source": risk_success_rule_source,
        "attack_wrapper_metadata": wrapper_metadata,
        "metadata": metadata,
        "injection_ground_truth_calls": _ensure_list(
            attack_spec.get("injection_ground_truth_calls") or metadata.get("injection_ground_truth_calls")
        ),
        "risk_benchmark_semantics": copy.deepcopy(
            attack_spec.get("risk_benchmark_semantics") or metadata.get("risk_benchmark_semantics") or {}
        ),
    }


def infer_injection_candidates(source_task, injection_surface=None):
    surface = str(injection_surface or "query").strip()
    metadata = copy.deepcopy(source_task.get("metadata") or {})
    precomputed = _ensure_list(
        metadata.get("attack_candidate_templates")
        or metadata.get("visible_injection_candidates")
        or metadata.get("injection_candidates")
    )
    if precomputed:
        normalized = []
        for index, candidate in enumerate(precomputed, start=1):
            if not isinstance(candidate, dict):
                continue
            candidate_copy = copy.deepcopy(candidate)
            candidate_copy.setdefault("candidate_id", f"{surface}_{index}")
            candidate_copy.setdefault("injection_surface", surface)
            normalized.append(candidate_copy)
        if normalized:
            return normalized

    candidates = []
    if surface == "query":
        candidates.append(
            {
                "candidate_id": "query",
                "injection_surface": "query",
                "target": "user_query",
                "user_visible": True,
            }
        )
        return candidates

    if surface == "environment_content":
        content_artifacts = list(source_task.get("content_artifacts") or [])
        for index, artifact in enumerate(content_artifacts, start=1):
            if not isinstance(artifact, dict):
                continue
            state_path = str(artifact.get("state_path") or "").strip()
            if not state_path:
                continue
            candidates.append(
                {
                    "candidate_id": str(artifact.get("candidate_id") or f"env_{index}"),
                    "injection_surface": "environment_content",
                    "target_path": state_path,
                    "resource_kind": artifact.get("resource_kind"),
                    "locator": artifact.get("locator"),
                    "content_field": artifact.get("content_field"),
                    "content_fields": copy.deepcopy(artifact.get("content_fields") or []),
                    "user_visible": True,
                }
            )
        return candidates

    if surface == "tool_return":
        for tool in source_task.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            labels = {str(label).strip().lower() for label in (tool.get("labels") or [])}
            reads = list((tool.get("simulator_requirements") or {}).get("reads_state_keys") or [])
            if "search" not in labels and "read" not in labels and not reads:
                continue
            candidates.append(
                {
                    "candidate_id": f"tool_return_{tool.get('name')}",
                    "injection_surface": "tool_return",
                    "tool_name": tool.get("name"),
                    "match_arguments": {},
                    "user_visible": True,
                }
            )
        return candidates

    return candidates


def _select_candidates(source_task, attack_spec):
    candidates = list(attack_spec.get("injection_candidates") or [])
    if candidates:
        return candidates
    return infer_injection_candidates(source_task, attack_spec.get("injection_surface"))


def _injected_text_for_item(attack_spec, candidate, index):
    concrete = list(attack_spec.get("concrete_injections") or [])
    if concrete:
        item = concrete[index]
        if isinstance(item, dict):
            if "text" in item:
                return str(item.get("text") or "")
            if "injected_text" in item:
                return str(item.get("injected_text") or "")
        return str(item)
    benchmark_wrapped = _default_benchmark_wrapper_text(attack_spec)
    if benchmark_wrapped:
        return benchmark_wrapped
    template = attack_spec.get("injection_template")
    if isinstance(template, str) and template:
        return template
    metadata = attack_spec.get("metadata") or {}
    return str(
        metadata.get("attack_prompt")
        or metadata.get("injection_goal")
        or metadata.get("goal")
        or attack_spec.get("attack_name")
        or ""
    )


def _materialize_concrete_injections(source_task, attack_spec, candidates):
    explicit = list(attack_spec.get("concrete_injections") or [])
    if explicit:
        materialized = []
        for index, item in enumerate(explicit, start=1):
            if isinstance(item, dict):
                entry = copy.deepcopy(item)
            else:
                entry = {"text": str(item)}
            surface = str(entry.get("injection_surface") or attack_spec.get("injection_surface") or "query")
            if surface not in SUPPORTED_INJECTION_SURFACES:
                raise ValueError(f"Unsupported concrete injection surface {surface!r}.")
            candidate = None
            candidate_id = entry.get("candidate_id")
            if candidate_id:
                for item_candidate in candidates:
                    if item_candidate.get("candidate_id") == candidate_id:
                        candidate = item_candidate
                        break
            elif candidates:
                candidate = candidates[min(index - 1, len(candidates) - 1)]
            if candidate:
                entry.setdefault("candidate_id", candidate.get("candidate_id"))
                entry.setdefault("target_path", candidate.get("target_path"))
                entry.setdefault("tool_name", candidate.get("tool_name"))
                entry.setdefault("match_arguments", copy.deepcopy(candidate.get("match_arguments") or {}))
            entry.setdefault("injection_surface", surface)
            materialized.append(entry)
        return materialized

    materialized = []
    for index, candidate in enumerate(candidates):
        injected_text = _injected_text_for_item(attack_spec, candidate, min(index, max(0, len(candidates) - 1)))
        if not injected_text:
            continue
        surface = str(candidate.get("injection_surface") or attack_spec.get("injection_surface") or "query")
        entry = {
            "candidate_id": candidate.get("candidate_id"),
            "injection_surface": surface,
            "text": injected_text,
        }
        if surface == "environment_content":
            entry["target_path"] = candidate.get("target_path")
            if candidate.get("patches") is not None:
                entry["patches"] = copy.deepcopy(candidate.get("patches") or [])
            if candidate.get("target_paths") is not None:
                entry["target_paths"] = copy.deepcopy(candidate.get("target_paths") or [])
            entry["mode"] = str(candidate.get("mode") or "replace")
        elif surface == "tool_return":
            entry["tool_name"] = candidate.get("tool_name")
            entry["match_arguments"] = copy.deepcopy(candidate.get("match_arguments") or {})
            if candidate.get("tool_result_patch") is not None:
                entry["tool_result_patch"] = copy.deepcopy(candidate.get("tool_result_patch"))
            else:
                entry["tool_result_patch"] = {"injected_content": injected_text}
        else:
            entry["mode"] = str(candidate.get("mode") or "append")
        materialized.append(entry)
    return materialized


def _apply_query_injection(base_query, injection):
    query = str(base_query or "")
    text = str(injection.get("text") or injection.get("injected_text") or "").strip()
    if not text:
        return query
    mode = str(injection.get("mode") or "append").strip().lower()
    if mode == "prepend":
        return f"{text}\n\n{query}".strip()
    if mode == "replace":
        return text
    wrapper_prefix = str(injection.get("wrapper_prefix") or "").strip()
    wrapper_suffix = str(injection.get("wrapper_suffix") or "").strip()
    if wrapper_prefix or wrapper_suffix:
        return f"{wrapper_prefix}{query}{wrapper_suffix}\n\n{text}".strip()
    return f"{query}\n\n{text}".strip()


def _apply_environment_injection(environment_snapshot, injection):
    updated = copy.deepcopy(environment_snapshot or {})
    text = injection.get("text")
    patches = _ensure_list(injection.get("patches"))
    target_paths = _ensure_list(injection.get("target_paths"))
    if patches:
        for patch in patches:
            if not isinstance(patch, dict):
                continue
            target_path = str(patch.get("target_path") or "").strip()
            target_segments = patch.get("target_segments")
            target_ref = target_segments if target_segments else target_path
            if not target_ref:
                continue
            existing = _get_path_value(updated, target_ref)
            mode = str(patch.get("mode") or injection.get("mode") or "replace").strip().lower()
            prefix = str(patch.get("prefix") or "")
            suffix = str(patch.get("suffix") or "")
            if prefix or suffix:
                value = f"{prefix}{text}{suffix}"
            elif mode == "append" and isinstance(existing, str):
                value = f"{existing}\n{text}".strip()
            elif mode == "prepend" and isinstance(existing, str):
                value = f"{text}\n{existing}".strip()
            else:
                value = copy.deepcopy(text)
            _set_path_value(updated, target_ref, value)
        return updated
    if target_paths:
        for target_path in target_paths:
            updated = _apply_environment_injection(
                updated,
                {
                    "target_path": target_path,
                    "text": text,
                    "mode": injection.get("mode"),
                },
            )
        return updated
    target_path = str(injection.get("target_path") or "").strip()
    if not target_path:
        return updated
    mode = str(injection.get("mode") or "replace").strip().lower()
    existing = _get_path_value(updated, target_path)
    if mode == "append" and isinstance(existing, str):
        value = f"{existing}\n{text}".strip()
    elif mode == "prepend" and isinstance(existing, str):
        value = f"{text}\n{existing}".strip()
    else:
        value = copy.deepcopy(text)
    _set_path_value(updated, target_path, value)
    return updated


def _overlay_from_tool_return_injection(injection):
    overlay = {
        "tool_name": str(injection.get("tool_name") or "").strip(),
        "match_arguments": copy.deepcopy(injection.get("match_arguments") or {}),
        "tool_result_patch": copy.deepcopy(injection.get("tool_result_patch") or {}),
    }
    if injection.get("observation_suffix"):
        overlay["observation_suffix"] = str(injection.get("observation_suffix"))
    if injection.get("state_patch"):
        overlay["state_patch"] = copy.deepcopy(injection.get("state_patch"))
    return overlay


def _source_task_matches_attack_spec(source_task, attack_spec):
    source_domain = str(source_task.get("domain") or "").strip()
    spec_domain = str(attack_spec.get("domain") or "").strip()
    if spec_domain and source_domain and spec_domain != source_domain:
        return False
    source_metadata = source_task.get("metadata") or {}
    spec_metadata = attack_spec.get("metadata") or {}
    source_suite = str(source_metadata.get("suite") or "").strip()
    spec_suite = str(spec_metadata.get("suite") or "").strip()
    if source_suite and spec_suite and source_suite != spec_suite:
        return False
    return True


def materialize_source_task_with_attack_spec(source_task, attack_spec):
    normalized_attack_spec = normalize_attack_spec(attack_spec)
    if not _source_task_matches_attack_spec(source_task, normalized_attack_spec):
        raise ValueError("Attack spec domain/suite does not match the utility source task.")

    candidates = _select_candidates(source_task, normalized_attack_spec)
    concrete_injections = _materialize_concrete_injections(source_task, normalized_attack_spec, candidates)
    base_query = str(source_task.get("task_text") or source_task.get("task") or "").strip()
    attacked_query = base_query
    attacked_environment_snapshot = copy.deepcopy(source_task.get("environment_snapshot") or {})
    tool_result_overlays = []

    for injection in concrete_injections:
        surface = str(injection.get("injection_surface") or normalized_attack_spec.get("injection_surface") or "query")
        if surface == "query":
            attacked_query = _apply_query_injection(attacked_query, injection)
        elif surface == "environment_content":
            attacked_environment_snapshot = _apply_environment_injection(attacked_environment_snapshot, injection)
        elif surface == "tool_return":
            tool_result_overlays.append(_overlay_from_tool_return_injection(injection))

    materialized = copy.deepcopy(source_task)
    base_source_task_id = materialized["source_task_id"]
    materialized["source_task_id"] = f"{base_source_task_id}__attack__{normalized_attack_spec['attack_spec_id']}"
    materialized["attacked_environment_snapshot"] = attacked_environment_snapshot
    materialized["attacked_user_query"] = attacked_query
    materialized["tool_result_overlays"] = tool_result_overlays

    metadata = copy.deepcopy(materialized.get("metadata") or {})
    metadata["attack_spec_id"] = normalized_attack_spec["attack_spec_id"]
    metadata["pair_id"] = base_source_task_id
    metadata["injection_candidates"] = copy.deepcopy(candidates)
    metadata["concrete_injections"] = copy.deepcopy(concrete_injections)
    metadata["attack_materialization"] = {
        "attack_spec_id": normalized_attack_spec["attack_spec_id"],
        "attack_name": normalized_attack_spec["attack_name"],
        "injection_surface": normalized_attack_spec["injection_surface"],
        "attacked_user_query": attacked_query,
        "tool_result_overlays": copy.deepcopy(tool_result_overlays),
    }
    metadata["attack_wrapper_metadata"] = copy.deepcopy(normalized_attack_spec.get("attack_wrapper_metadata") or {})
    metadata["injection_ground_truth_calls"] = copy.deepcopy(
        normalized_attack_spec.get("injection_ground_truth_calls") or metadata.get("injection_ground_truth_calls") or []
    )
    if normalized_attack_spec.get("risk_benchmark_semantics"):
        metadata["risk_benchmark_semantics"] = copy.deepcopy(normalized_attack_spec["risk_benchmark_semantics"])
    materialized["metadata"] = metadata
    materialized["notes"] = (
        f"{materialized.get('notes', '').strip()} "
        f"Materialized attacked scenario using attack spec {normalized_attack_spec['attack_spec_id']}."
    ).strip()
    return materialized


def materialize_source_tasks_with_attack_specs(source_tasks, attack_specs, limit=None, progress=None):
    materialized = []
    normalized_specs = [normalize_attack_spec(item) for item in (attack_specs or [])]
    if not normalized_specs:
        return [copy.deepcopy(item) for item in source_tasks]
    total_units = sum(
        1
        for source_task in source_tasks
        for spec in normalized_specs
        if _source_task_matches_attack_spec(source_task, spec)
    )
    phase = progress.phase("Materialize attack scenarios", total_units) if progress and total_units else None
    for source_task in source_tasks:
        for attack_spec in normalized_specs:
            if not _source_task_matches_attack_spec(source_task, attack_spec):
                continue
            materialized.append(materialize_source_task_with_attack_spec(source_task, attack_spec))
            if phase:
                phase.advance(detail=f"{source_task.get('source_task_id')} + {attack_spec.get('attack_spec_id')}")
            if limit is not None and len(materialized) >= int(limit):
                if phase:
                    phase.close()
                return materialized
    if phase:
        phase.close()
    return materialized
