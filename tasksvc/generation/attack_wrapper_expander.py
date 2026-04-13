import copy

from tasksvc.assembly.bundle_validator import validate_runtime_bundle, validate_task_draft
from tasksvc.generation.agentdojo_source_tasks import (
    DEFAULT_AGENTDOJO_ATTACK_VARIANTS,
    render_agentdojo_benchmark_wrapper_text,
)
from tasksvc.generation.attack_materializer import (
    _apply_environment_injection,
    _apply_query_injection,
    _overlay_from_tool_return_injection,
)


def _slugify(value):
    text = str(value or "").strip().lower()
    chars = []
    for ch in text:
        chars.append(ch if ch.isalnum() else "_")
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "attack"


def _variant_id(variant):
    return str(variant.get("attack_name") or "attack").strip() or "attack"


def _source_task_query_text(source_task):
    return str((source_task or {}).get("task_text") or (source_task or {}).get("task") or "").strip()


def _attack_goal_text(source_task):
    metadata = copy.deepcopy(source_task.get("metadata") or {})
    if metadata.get("injection_goal"):
        return str(metadata["injection_goal"])
    for injection in metadata.get("concrete_injections") or []:
        text = str((injection or {}).get("text") or "").strip()
        if text:
            return text
    attack_materialization = copy.deepcopy(metadata.get("attack_materialization") or {})
    if attack_materialization.get("attack_name"):
        return str(attack_materialization["attack_name"])
    return str(metadata.get("injection_goal") or metadata.get("attack_prompt") or "").strip()


def _wrapped_injection_text(source_task, variant):
    metadata = copy.deepcopy(source_task.get("metadata") or {})
    wrapper_metadata = copy.deepcopy(metadata.get("attack_wrapper_metadata") or {})
    benchmark_wrappers = copy.deepcopy(wrapper_metadata.get("benchmark_wrapper_text_by_name") or {})
    attack_name = _variant_id(variant)
    wrapped = benchmark_wrappers.get(attack_name)
    if isinstance(wrapped, str) and wrapped:
        return wrapped
    return render_agentdojo_benchmark_wrapper_text(
        attack_name,
        _attack_goal_text(source_task),
        user_name=str(wrapper_metadata.get("user_name") or ""),
        model_name=str(wrapper_metadata.get("model_name") or ""),
        user_task_prompt=str(metadata.get("user_task_prompt") or _source_task_query_text(source_task)),
    )


def _wrapped_task_id(base_task_id, variant):
    return f"{base_task_id}__attack__{_slugify(_variant_id(variant))}"


def _rewrite_attack_materialization(source_task, variant):
    wrapped_text = _wrapped_injection_text(source_task, variant)
    metadata = copy.deepcopy(source_task.get("metadata") or {})
    base_wrapper_metadata = copy.deepcopy(metadata.get("attack_wrapper_metadata") or {})
    concrete_injections = copy.deepcopy(metadata.get("concrete_injections") or [])
    base_query = _source_task_query_text(source_task)
    attacked_query = base_query
    attacked_environment_snapshot = copy.deepcopy(source_task.get("environment_snapshot") or {})
    tool_result_overlays = []
    rewritten_injections = []

    for injection in concrete_injections:
        updated = copy.deepcopy(injection)
        surface = str(updated.get("injection_surface") or "query").strip()
        if surface == "query":
            updated["text"] = wrapped_text
            attacked_query = _apply_query_injection(attacked_query, updated)
        elif surface == "environment_content":
            updated["text"] = wrapped_text
            attacked_environment_snapshot = _apply_environment_injection(attacked_environment_snapshot, updated)
        elif surface == "tool_return":
            patch = copy.deepcopy(updated.get("tool_result_patch") or {})
            if patch:
                if isinstance(patch.get("injected_content"), str):
                    patch["injected_content"] = wrapped_text
                elif isinstance(patch.get("content"), str):
                    patch["content"] = wrapped_text
                else:
                    patch["injected_content"] = wrapped_text
            else:
                patch = {"injected_content": wrapped_text}
            updated["tool_result_patch"] = patch
            tool_result_overlays.append(_overlay_from_tool_return_injection(updated))
        rewritten_injections.append(updated)

    attack_wrapper_metadata = copy.deepcopy(base_wrapper_metadata)
    attack_wrapper_metadata.update(
        {
            "attack_name": _variant_id(variant),
            "wrapped_attack_text": wrapped_text,
        }
    )
    attack_materialization = copy.deepcopy(metadata.get("attack_materialization") or {})
    attack_materialization.update(
        {
            "attack_name": attack_wrapper_metadata["attack_name"],
            "attack_wrapper_id": attack_wrapper_metadata["attack_name"],
            "wrapped_attack_text": wrapped_text,
            "attacked_user_query": attacked_query,
            "tool_result_overlays": copy.deepcopy(tool_result_overlays),
        }
    )
    return {
        "attacked_user_query": attacked_query,
        "attacked_environment_snapshot": attacked_environment_snapshot,
        "tool_result_overlays": tool_result_overlays,
        "concrete_injections": rewritten_injections,
        "attack_wrapper_metadata": attack_wrapper_metadata,
        "attack_materialization": attack_materialization,
    }


def expand_paired_source_task_to_wrapper_variant(source_task, variant):
    expanded = copy.deepcopy(source_task)
    expanded["source_task_id"] = _wrapped_task_id(source_task["source_task_id"], variant)
    rewritten = _rewrite_attack_materialization(source_task, variant)
    expanded["attacked_user_query"] = rewritten["attacked_user_query"]
    expanded["attacked_environment_snapshot"] = rewritten["attacked_environment_snapshot"]
    expanded["tool_result_overlays"] = copy.deepcopy(rewritten["tool_result_overlays"])
    metadata = copy.deepcopy(expanded.get("metadata") or {})
    metadata["evaluation_track"] = "attacked_4x"
    metadata["attack_wrapper_metadata"] = copy.deepcopy(rewritten["attack_wrapper_metadata"])
    metadata["attack_materialization"] = copy.deepcopy(rewritten["attack_materialization"])
    metadata["concrete_injections"] = copy.deepcopy(rewritten["concrete_injections"])
    expanded["metadata"] = metadata
    expanded["notes"] = (
        f"{str(expanded.get('notes') or '').strip()} Expanded attacked wrapper "
        f"{rewritten['attack_wrapper_metadata']['attack_name']}."
    ).strip()
    return expanded


def _rewrite_draft_scenarios(draft, wrapped_source_task):
    attacked_query = str(wrapped_source_task.get("attacked_user_query") or _source_task_query_text(wrapped_source_task))
    overlays = copy.deepcopy(wrapped_source_task.get("tool_result_overlays") or [])
    for container_key in ("task_plan_spec", "agent_input_view", "state_draft"):
        scenarios = copy.deepcopy((draft.get(container_key) or {}).get("scenarios") or {})
        attacked = copy.deepcopy(scenarios.get("attacked") or {})
        attacked["scenario"] = "attacked"
        attacked["risk_enabled"] = True
        attacked["user_query"] = attacked_query
        attacked["tool_result_overlays"] = overlays
        if container_key == "state_draft":
            attacked["initial_state_template"] = copy.deepcopy(
                wrapped_source_task.get("attacked_environment_snapshot") or {}
            )
        scenarios["attacked"] = attacked
        draft[container_key]["scenarios"] = scenarios


def expand_paired_draft_to_wrapper_variant(draft, source_task_variant):
    expanded = copy.deepcopy(draft)
    new_task_id = source_task_variant["source_task_id"]
    expanded["source_task_input"] = copy.deepcopy(source_task_variant)
    expanded["planned_task"]["task_id"] = new_task_id
    expanded["task_plan_spec"]["task_id"] = new_task_id
    task_metadata = copy.deepcopy(expanded["planned_task"].get("task_metadata") or {})
    task_metadata["source_task_id"] = new_task_id
    benchmark_metadata = copy.deepcopy(task_metadata.get("benchmark_metadata") or {})
    benchmark_metadata.update(
        {
            "evaluation_track": "attacked_4x",
            "attack_wrapper_metadata": copy.deepcopy(
                (source_task_variant.get("metadata") or {}).get("attack_wrapper_metadata") or {}
            ),
            "attack_materialization": copy.deepcopy(
                (source_task_variant.get("metadata") or {}).get("attack_materialization") or {}
            ),
            "concrete_injections": copy.deepcopy(
                (source_task_variant.get("metadata") or {}).get("concrete_injections") or []
            ),
        }
    )
    task_metadata["benchmark_metadata"] = benchmark_metadata
    expanded["planned_task"]["task_metadata"] = task_metadata
    _rewrite_draft_scenarios(expanded, source_task_variant)
    validate_task_draft(expanded)
    return expanded


def expand_paired_bundle_to_wrapper_variant(bundle, source_task_variant):
    expanded = copy.deepcopy(bundle)
    new_task_id = source_task_variant["source_task_id"]
    expanded["task_spec"]["task_id"] = new_task_id
    task_metadata = copy.deepcopy(expanded["task_spec"].get("task_metadata") or {})
    task_metadata["source_task_id"] = new_task_id
    benchmark_metadata = copy.deepcopy(task_metadata.get("benchmark_metadata") or {})
    benchmark_metadata.update(
        {
            "evaluation_track": "attacked_4x",
            "attack_wrapper_metadata": copy.deepcopy(
                (source_task_variant.get("metadata") or {}).get("attack_wrapper_metadata") or {}
            ),
            "attack_materialization": copy.deepcopy(
                (source_task_variant.get("metadata") or {}).get("attack_materialization") or {}
            ),
            "concrete_injections": copy.deepcopy(
                (source_task_variant.get("metadata") or {}).get("concrete_injections") or []
            ),
        }
    )
    task_metadata["benchmark_metadata"] = benchmark_metadata
    expanded["task_spec"]["task_metadata"] = task_metadata
    scenarios = copy.deepcopy(expanded["execution_bundle"].get("scenarios") or {})
    attacked = copy.deepcopy(scenarios.get("attacked") or {})
    attacked["scenario"] = "attacked"
    attacked["risk_enabled"] = True
    attacked["user_query"] = str(source_task_variant.get("attacked_user_query") or _source_task_query_text(source_task_variant))
    attacked["tool_result_overlays"] = copy.deepcopy(source_task_variant.get("tool_result_overlays") or [])
    attacked["initial_state_template"] = copy.deepcopy(source_task_variant.get("attacked_environment_snapshot") or {})
    scenarios["attacked"] = attacked
    expanded["execution_bundle"]["scenarios"] = scenarios
    validate_runtime_bundle(expanded)
    return expanded


def iter_expanded_wrapper_variants(
    source_tasks,
    task_drafts,
    runtime_catalog,
    attack_variants=None,
):
    variants = copy.deepcopy(attack_variants or DEFAULT_AGENTDOJO_ATTACK_VARIANTS)
    draft_by_id = {
        str((draft.get("source_task_input") or {}).get("source_task_id") or ""): draft
        for draft in task_drafts or []
    }
    bundle_by_id = {
        str(task_id): bundle
        for task_id, bundle in (runtime_catalog or {}).items()
    }
    for source_task in source_tasks or []:
        source_task_id = str(source_task.get("source_task_id") or "")
        base_draft = draft_by_id.get(source_task_id)
        base_bundle = bundle_by_id.get(source_task_id)
        if base_draft is None or base_bundle is None:
            raise ValueError(f"Missing paired draft or bundle for source_task_id={source_task_id}")
        for variant in variants:
            wrapped_source_task = expand_paired_source_task_to_wrapper_variant(source_task, variant)
            wrapped_draft = expand_paired_draft_to_wrapper_variant(base_draft, wrapped_source_task)
            wrapped_bundle = expand_paired_bundle_to_wrapper_variant(base_bundle, wrapped_source_task)
            yield wrapped_source_task, wrapped_draft, wrapped_bundle


def expand_paired_artifacts_to_attack_wrapped_variants(
    source_tasks,
    task_drafts,
    runtime_catalog,
    attack_variants=None,
    limit=None,
    progress=None,
):
    variants = copy.deepcopy(attack_variants or DEFAULT_AGENTDOJO_ATTACK_VARIANTS)
    total = len(source_tasks or []) * len(variants)
    phase = progress.phase("Expand attacked wrappers", total) if progress and total else None
    expanded_source_tasks = []
    expanded_drafts = []
    expanded_bundles = {}
    for wrapped_source_task, wrapped_draft, wrapped_bundle in iter_expanded_wrapper_variants(
        source_tasks,
        task_drafts,
        runtime_catalog,
        attack_variants=variants,
    ):
        expanded_source_tasks.append(wrapped_source_task)
        expanded_drafts.append(wrapped_draft)
        expanded_bundles[wrapped_bundle["task_spec"]["task_id"]] = wrapped_bundle
        if phase:
            phase.advance(detail=wrapped_bundle["task_spec"]["task_id"])
        if limit is not None and len(expanded_source_tasks) >= int(limit):
            if phase:
                phase.close()
            return expanded_source_tasks, expanded_drafts, expanded_bundles
    if phase:
        phase.close()
    return expanded_source_tasks, expanded_drafts, expanded_bundles
