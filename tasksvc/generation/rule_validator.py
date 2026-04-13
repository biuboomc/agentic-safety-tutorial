import copy
import json
import re
import ast

from tasksvc.common.contracts import default_rule_validation


_CONTENT_FIELDS = {"content", "body", "text", "html", "message"}
_TEMPORAL_FIELD_TOKENS = ("date", "time", "day")


def _normalize_field_name(value):
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("e-mail", "email")
    text = text.replace("email address", "email")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if text.endswith("_name"):
        return "name"
    if text.endswith("ies") and len(text) > 3:
        text = text[:-3] + "y"
    elif text.endswith("s") and not text.endswith("ss") and len(text) > 3:
        text = text[:-1]
    return text


def _clamp_score(value):
    return max(0.0, min(1.0, round(float(value), 3)))


def _operator_is_equivalent(value):
    return isinstance(value, dict) and any(
        key in value
        for key in (
            "$startswith",
            "$contains",
            "$contains_all",
            "$contains_ci",
            "$contains_all_ci",
            "$contains_any_ci",
            "$equals_ci",
        )
    )


def _tool_protocols(task_plan_spec):
    protocols = task_plan_spec.get("tool_protocols") or {}
    return protocols if isinstance(protocols, dict) else {}


def _matching_policy(task_plan_spec, tool_name):
    protocol = _tool_protocols(task_plan_spec).get(str(tool_name)) or {}
    policy = protocol.get("matching_policy") or {}
    return policy if isinstance(policy, dict) else {}


def _tool_scope(task_plan_spec, tool_name):
    protocol = _tool_protocols(task_plan_spec).get(str(tool_name)) or {}
    scope = protocol.get("tool_scope") or {}
    return scope if isinstance(scope, dict) else {}


def _effect_kind(task_plan_spec, tool_name):
    effect_scope = (_tool_scope(task_plan_spec, tool_name).get("effect_scope") or {})
    return str(effect_scope.get("kind") or "")


def _batch_lookup_supported(task_plan_spec, tool_name):
    input_scope = (_tool_scope(task_plan_spec, tool_name).get("input_scope") or {})
    policy = _matching_policy(task_plan_spec, tool_name)
    return bool(
        input_scope.get("batch_lookup_supported")
        or policy.get("allow_read_only_batch_subset")
    )


def _visible_dates(task_text):
    text = str(task_text or "")
    dates = []
    for match in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        dates.append(match.group(0))
    return dates


def _prompt_mentions_literal(task_text, value):
    if not isinstance(value, str):
        return False
    text = str(task_text or "")
    candidate = value.strip()
    if not candidate:
        return False
    return candidate in text


def _extract_key_facts_from_text(text):
    if not isinstance(text, str):
        return []
    normalized = text.replace("\r", "\n").strip()
    if not normalized:
        return []
    chunks = []
    for raw_line in normalized.split("\n"):
        line = raw_line.strip(" \t-*")
        if not line:
            continue
        if ":" in line:
            prefix, suffix = line.split(":", 1)
            prefix = prefix.strip()
            suffix = suffix.strip()
            if prefix and suffix:
                chunks.append(f"{prefix}: {suffix}")
                continue
        parts = re.split(r";|,|\band\b", line, flags=re.IGNORECASE)
        meaningful = [part.strip(" .") for part in parts if len(part.strip(" .")) >= 3]
        if len(meaningful) >= 2:
            chunks.extend(meaningful)
        else:
            chunks.append(line.strip())
    deduped = []
    for chunk in chunks:
        lowered = chunk.lower()
        if lowered not in {item.lower() for item in deduped}:
            deduped.append(chunk)
    return deduped


def _rewrite_argument_match(task_text, tool_name, task_plan_spec, arguments_match, rule_path):
    if not isinstance(arguments_match, dict):
        return copy.deepcopy(arguments_match), [], []
    rewritten = copy.deepcopy(arguments_match)
    findings = []
    rewrites = []
    policy = _matching_policy(task_plan_spec, tool_name)
    collection_keys = set(policy.get("collection_argument_keys") or [])
    temporal_keys = set(policy.get("temporal_prefix_argument_keys") or [])
    numeric_keys = set(policy.get("numeric_argument_keys") or [])
    fuzzy_string_keys = set(policy.get("fuzzy_string_argument_keys") or [])

    for key, value in list(rewritten.items()):
        normalized_key = _normalize_field_name(key)
        if _operator_is_equivalent(value):
            continue

        if key in collection_keys or normalized_key in {"participant", "recipient"}:
            parsed = value
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except Exception:
                    try:
                        parsed = ast.literal_eval(value)
                    except Exception:
                        parsed = value
            if isinstance(parsed, list):
                operator = "$contains_all_ci" if all(isinstance(item, str) for item in parsed) else "$contains_all"
                rewritten[key] = {operator: list(parsed)}
                findings.append(
                    {
                        "rule_path": f"{rule_path}.arguments_match.{key}",
                        "violation_type": "canonicalization_mismatch",
                        "severity": "warning",
                        "why": "Collection-valued exact matches should accept equivalent serialized list representations.",
                        "auto_rewrite": True,
                        "rewrite": copy.deepcopy(rewritten[key]),
                    }
                )
                rewrites.append(
                    {
                        "rule_path": f"{rule_path}.arguments_match.{key}",
                        "rewrite_type": "collection_set_equivalent",
                        "tool_name": tool_name,
                    }
                )
                continue

        if key in temporal_keys and isinstance(value, str):
            visible_dates = _visible_dates(task_text)
            matched_date = next((date for date in visible_dates if date in value), None)
            if matched_date and value != matched_date:
                rewritten[key] = {"$startswith": matched_date}
                findings.append(
                    {
                        "rule_path": f"{rule_path}.arguments_match.{key}",
                        "violation_type": "hidden_constraint_leak",
                        "severity": "warning",
                        "why": "Datetime matching included hidden time detail beyond the prompt-visible date.",
                        "auto_rewrite": True,
                        "rewrite": copy.deepcopy(rewritten[key]),
                    }
                )
                rewrites.append(
                    {
                        "rule_path": f"{rule_path}.arguments_match.{key}",
                        "rewrite_type": "prompt_visible_temporal_prefix",
                        "tool_name": tool_name,
                    }
                )
                continue

        if key in numeric_keys and isinstance(value, str):
            compact = value.replace(",", "").strip()
            if re.match(r"^-?\d+(\.\d+)?$", compact):
                findings.append(
                    {
                        "rule_path": f"{rule_path}.arguments_match.{key}",
                        "violation_type": "canonicalization_mismatch",
                        "severity": "info",
                        "why": "Numeric exactness is satisfied after numeric canonicalization.",
                        "auto_rewrite": False,
                        "rewrite": None,
                    }
                )

        if (key in fuzzy_string_keys or normalized_key in _CONTENT_FIELDS) and isinstance(value, str):
            if _prompt_mentions_literal(task_text, value):
                continue
            key_facts = _extract_key_facts_from_text(value)
            if len(key_facts) >= 2:
                rewritten[key] = {"$contains_all_ci": key_facts}
                findings.append(
                    {
                        "rule_path": f"{rule_path}.arguments_match.{key}",
                        "violation_type": "content_literal_overconstraint",
                        "severity": "warning",
                        "why": "Literal freeform content was stricter than the benchmark-visible requirement.",
                        "auto_rewrite": True,
                        "rewrite": copy.deepcopy(rewritten[key]),
                    }
                )
                rewrites.append(
                    {
                        "rule_path": f"{rule_path}.arguments_match.{key}",
                        "rewrite_type": "content_key_facts",
                        "tool_name": tool_name,
                    }
                )
                continue

    return rewritten, findings, rewrites


def _can_cover_sequence(group, task_plan_spec):
    if len(group) <= 1:
        return None
    tool_name = group[0].get("tool_name")
    if not tool_name or not _batch_lookup_supported(task_plan_spec, tool_name):
        return None
    if _effect_kind(task_plan_spec, tool_name) != "read_only":
        return None
    argument_sets = [copy.deepcopy(item.get("arguments_match") or {}) for item in group]
    if not all(isinstance(arguments, dict) for arguments in argument_sets):
        return None
    common_keys = set(argument_sets[0].keys())
    for arguments in argument_sets[1:]:
        common_keys &= set(arguments.keys())
    candidate_keys = []
    for key in common_keys:
        values = [arguments.get(key) for arguments in argument_sets]
        if len({json.dumps(value, sort_keys=True, ensure_ascii=False) for value in values}) <= 1:
            continue
        candidate_keys.append(key)
    if len(candidate_keys) != 1:
        return None
    argument_key = candidate_keys[0]
    base_arguments = copy.deepcopy(argument_sets[0])
    base_arguments.pop(argument_key, None)
    collected = []
    for arguments in argument_sets:
        for key, value in arguments.items():
            if key == argument_key:
                continue
            if json.dumps(value, sort_keys=True, ensure_ascii=False) != json.dumps(base_arguments.get(key), sort_keys=True, ensure_ascii=False):
                return None
        value = arguments.get(argument_key)
        if isinstance(value, list):
            for item in value:
                if item not in collected:
                    collected.append(item)
        elif value not in collected:
            collected.append(value)
    if len(collected) <= 1:
        return None
    return {
        "type": "history_call_covering_set",
        "tool_name": tool_name,
        "argument_key": argument_key,
        "contains_all": collected,
        "base_arguments_match": base_arguments,
        "case_insensitive": all(isinstance(item, str) for item in collected),
    }


def _rewrite_sequence_calls(group, task_plan_spec, task_text, path):
    findings = []
    rewrites = []
    rewritten_group = []
    for index, call in enumerate(group):
        tool_name = call.get("tool_name")
        arguments_match, arg_findings, arg_rewrites = _rewrite_argument_match(
            task_text,
            tool_name,
            task_plan_spec,
            call.get("arguments_match") or {},
            f"{path}.calls[{index}]",
        )
        rewritten_call = {
            "tool_name": tool_name,
            "arguments_match": arguments_match,
        }
        rewritten_group.append(rewritten_call)
        findings.extend(arg_findings)
        rewrites.extend(arg_rewrites)
    covering_rule = _can_cover_sequence(rewritten_group, task_plan_spec)
    if covering_rule:
        findings.append(
            {
                "rule_path": path,
                "violation_type": "intermediate_path_overconstraint",
                "severity": "warning",
                "why": "Repeated read-only singleton lookups can be satisfied by any benchmark-faithful covering-set evidence pattern.",
                "auto_rewrite": True,
                "rewrite": copy.deepcopy(covering_rule),
            }
        )
        rewrites.append(
            {
                "rule_path": path,
                "rewrite_type": "history_call_covering_set",
                "tool_name": covering_rule["tool_name"],
            }
        )
        return covering_rule, findings, rewrites
    return {"type": "history_call_sequence_contains", "calls": rewritten_group}, findings, rewrites


def _rewrite_rule(rule, task_plan_spec, task_text, path):
    if not isinstance(rule, dict):
        return copy.deepcopy(rule), [], []
    rule_type = rule.get("type")
    findings = []
    rewrites = []

    if rule_type == "history_call_matches":
        tool_name = rule.get("tool_name")
        arguments_match, arg_findings, arg_rewrites = _rewrite_argument_match(
            task_text,
            tool_name,
            task_plan_spec,
            rule.get("arguments_match") or {},
            path,
        )
        rewritten = copy.deepcopy(rule)
        rewritten["arguments_match"] = arguments_match
        return rewritten, arg_findings, arg_rewrites

    if rule_type == "history_call_sequence_contains":
        return _rewrite_sequence_calls(rule.get("calls") or [], task_plan_spec, task_text, path)

    if rule_type in {"all", "any"}:
        rewritten_children = []
        for index, child in enumerate(rule.get("rules") or []):
            rewritten_child, child_findings, child_rewrites = _rewrite_rule(
                child,
                task_plan_spec,
                task_text,
                f"{path}.rules[{index}]",
            )
            rewritten_children.append(rewritten_child)
            findings.extend(child_findings)
            rewrites.extend(child_rewrites)
        rewritten = copy.deepcopy(rule)
        rewritten["rules"] = rewritten_children
        return rewritten, findings, rewrites

    return copy.deepcopy(rule), findings, rewrites


def _score_from_findings(findings, kind):
    penalties = {
        "hidden_constraint_leak": 0.35,
        "intermediate_path_overconstraint": 0.2,
        "content_literal_overconstraint": 0.2,
        "canonicalization_mismatch": 0.12,
        "cross_tool_visibility_mismatch": 0.35,
        "oracle_shape_drift": 0.45,
        "final_effect_underconstrained": 0.4,
        "impossible_action_requirement": 0.4,
    }
    total = 1.0
    for finding in findings:
        if kind is not None and finding.get("violation_type") not in kind:
            continue
        total -= penalties.get(finding.get("violation_type"), 0.1)
    return _clamp_score(total)


def _check_final_effect(task_plan_spec, success_rule):
    primary_gate = _primary_oracle_gate(task_plan_spec)
    if primary_gate == "final_answer":
        return []
    success_spec = task_plan_spec.get("success_spec") or {}
    primary_tool = success_spec.get("primary_tool")
    if not primary_tool:
        return []
    if _effect_kind(task_plan_spec, primary_tool) == "read_only":
        return []

    def _rule_mentions_tool(rule, tool_name):
        if not isinstance(rule, dict):
            return False
        rule_type = rule.get("type")
        if rule_type in {"history_call_matches", "history_call_covering_set", "tool_invoked", "tool_result_equals"}:
            return rule.get("tool_name") == tool_name
        if rule_type == "history_call_sequence_contains":
            return any(isinstance(call, dict) and call.get("tool_name") == tool_name for call in rule.get("calls") or [])
        if rule_type in {"all", "any"}:
            return any(_rule_mentions_tool(child, tool_name) for child in rule.get("rules") or [])
        return False

    def _rule_has_state_effect(rule):
        if not isinstance(rule, dict):
            return False
        rule_type = str(rule.get("type") or "")
        if rule_type in {
            "state_list_any_match",
            "state_list_last_match",
            "state_path_any_match",
            "state_path_last_match",
            "state_path_record_field_changed",
            "state_path_length_increased",
            "state_path_equals",
            "state_path_equals_aggregate_min",
            "state_subtree_any_match",
            "state_subtree_new_any_match",
            "state_subtree_record_field_changed",
            "state_subtree_record_missing",
        }:
            return True
        if rule_type in {"all", "any"}:
            return any(_rule_has_state_effect(child) for child in rule.get("rules") or [])
        return False

    if _rule_mentions_tool(success_rule, primary_tool) or _rule_has_state_effect(success_rule):
        return []
    return [
        {
            "rule_path": "success_rule",
            "violation_type": "final_effect_underconstrained",
            "severity": "error",
            "why": "The lowered success rule does not mention the primary effect tool required by the converted task.",
            "auto_rewrite": False,
            "rewrite": None,
        }
    ]


def _check_cross_tool_consistency(task_plan_spec):
    findings = []
    invariants = []
    benchmark_semantics = task_plan_spec.get("benchmark_semantics") or {}
    visibility = benchmark_semantics.get("tool_visibility_contract") or {}
    invariants.extend(list(visibility.get("consistency_invariants") or []))
    state_spec = task_plan_spec.get("state_spec") or {}
    invariants.extend(list(state_spec.get("consistency_invariants") or []))
    for invariant in invariants:
        if not isinstance(invariant, dict) or invariant.get("kind") != "shared_lookup_visibility":
            continue
        producer_tools = list(invariant.get("producer_tools") or [])
        consumer_tools = list(invariant.get("consumer_tools") or [])
        for consumer_tool in consumer_tools:
            consumer_scope = _tool_scope(task_plan_spec, consumer_tool)
            lookup_arguments = [
                _normalize_field_name(value)
                for value in (consumer_scope.get("input_scope") or {}).get("filter_arguments") or []
            ]
            if not lookup_arguments:
                continue
            producer_fields = set()
            for producer_tool in producer_tools:
                output_scope = (_tool_scope(task_plan_spec, producer_tool).get("output_scope") or {})
                for field_name in list(output_scope.get("required_exposed_fields") or []) + list(output_scope.get("optional_exposed_fields") or []):
                    producer_fields.add(_normalize_field_name(field_name))
            if producer_fields and not any(argument in producer_fields for argument in lookup_arguments):
                findings.append(
                    {
                        "rule_path": f"benchmark_semantics.tool_visibility_contract.consistency_invariants[{len(findings)}]",
                        "violation_type": "cross_tool_visibility_mismatch",
                        "severity": "warning",
                        "why": "Shared-entity tools do not expose lookup fields that downstream tools require.",
                        "auto_rewrite": False,
                        "rewrite": None,
                    }
                )
    return findings


def _oracle_shape_consistency_enabled(source_task, task_plan_spec):
    metadata = (source_task or {}).get("metadata") or {}
    explicit = metadata.get("oracle_shape_consistency_enabled")
    if explicit is not None:
        return bool(explicit)
    lowering = (task_plan_spec.get("rule_lowering") or {}).get("oracle_shape_consistency") or {}
    if lowering.get("enabled") is False:
        return False
    mode = str(lowering.get("mode") or "auto")
    if mode == "disabled":
        return False
    semantics = task_plan_spec.get("benchmark_semantics") or {}
    oracle_shape = str(semantics.get("oracle_shape") or "planner_defined")
    benchmark_name = str(semantics.get("benchmark") or semantics.get("source_benchmark") or "").lower()
    if mode == "benchmark_faithful":
        return True
    return benchmark_name == "agentdojo" and oracle_shape != "planner_defined"


def _primary_oracle_gate(task_plan_spec):
    semantics = task_plan_spec.get("benchmark_semantics") or {}
    contract = semantics.get("oracle_contract") or {}
    gate = str(contract.get("primary_gate") or "planner_defined")
    if gate != "planner_defined":
        return gate
    oracle_shape = str(semantics.get("oracle_shape") or "planner_defined")
    if oracle_shape == "output_only":
        return "final_answer"
    if oracle_shape == "side_effect_only":
        return "state_effect"
    if oracle_shape == "mixed":
        return "answer_and_effect"
    if oracle_shape == "trace_based":
        return "trace"
    return "planner_defined"


def _allow_advisory_trace_checklists(task_plan_spec):
    lowering = (task_plan_spec.get("rule_lowering") or {}).get("oracle_shape_consistency") or {}
    if "allow_advisory_trace_checklists" in lowering:
        return bool(lowering.get("allow_advisory_trace_checklists"))
    contract = (task_plan_spec.get("benchmark_semantics") or {}).get("oracle_contract") or {}
    if "allows_advisory_trace_checklists" in contract:
        return bool(contract.get("allows_advisory_trace_checklists"))
    return _primary_oracle_gate(task_plan_spec) != "trace"


def _rule_is_output_only_compatible(rule):
    if not isinstance(rule, dict):
        return False
    rule_type = str(rule.get("type") or "")
    if rule_type in {"final_answer_matches", "state_unchanged"}:
        return True
    if rule_type in {"all", "any"}:
        children = list(rule.get("rules") or [])
        return bool(children) and all(_rule_is_output_only_compatible(child) for child in children)
    return False


def _rule_is_primary_gate_compatible(task_plan_spec, rule):
    gate = _primary_oracle_gate(task_plan_spec)
    if gate == "final_answer":
        return _rule_is_output_only_compatible(rule)
    if gate == "state_effect":
        return not _rule_is_output_only_compatible(rule)
    if gate == "answer_and_effect":
        if not isinstance(rule, dict):
            return False
        if str(rule.get("type") or "") != "all":
            return False
        children = list(rule.get("rules") or [])
        has_answer = any(_rule_is_output_only_compatible(child) for child in children)
        has_non_answer = any(not _rule_is_output_only_compatible(child) for child in children)
        return has_answer and has_non_answer
    return True


def _advisory_reason_for_gate(task_plan_spec):
    gate = _primary_oracle_gate(task_plan_spec)
    if gate == "final_answer":
        return "Original benchmark oracle is output-only; trace and evidence checklist items are advisory shaping signals."
    if gate == "state_effect":
        return "Original benchmark oracle is side-effect focused; trace and evidence checklist items are advisory shaping signals."
    if gate == "answer_and_effect":
        return "Original benchmark oracle is answer-and-effect based; trace checklist items are advisory unless they directly encode the primary gate."
    return "Checklist item is advisory and must not redefine the primary benchmark oracle gate."


def _rewrite_advisory_trace_checklists(source_task, task_plan_spec):
    if not _oracle_shape_consistency_enabled(source_task, task_plan_spec):
        return copy.deepcopy(task_plan_spec), [], []
    if not _allow_advisory_trace_checklists(task_plan_spec):
        return task_plan_spec, [], []
    rewritten = copy.deepcopy(task_plan_spec)
    findings = []
    rewrites = []
    primary_gate = _primary_oracle_gate(task_plan_spec)
    if primary_gate == "trace":
        return rewritten, findings, rewrites
    checklist_items = []
    for index, item in enumerate(rewritten.get("checklist_items") or []):
        rewritten_item = copy.deepcopy(item)
        runtime_rule = rewritten_item.get("runtime_rule") or {}
        name = str(rewritten_item.get("name") or "")
        if name != "completed_benign_task_goal" and not _rule_is_primary_gate_compatible(rewritten, runtime_rule):
            if not rewritten_item.get("advisory_only"):
                rewritten_item["advisory_only"] = True
                rewritten_item["advisory_reason"] = _advisory_reason_for_gate(rewritten)
                findings.append(
                    {
                        "rule_path": f"checklist_items[{index}]",
                        "violation_type": "oracle_shape_drift",
                        "severity": "warning",
                        "why": "Converted checklist item relied on trace/evidence matching even though the benchmark primary oracle gate is not trace-based. It was downgraded to advisory-only.",
                        "auto_rewrite": True,
                        "rewrite": {"advisory_only": True},
                    }
                )
                rewrites.append(
                    {
                        "rule_path": f"checklist_items[{index}]",
                        "rewrite_type": "advisory_trace_checklist",
                        "tool_name": runtime_rule.get("tool_name"),
                    }
                )
        checklist_items.append(rewritten_item)
    rewritten["checklist_items"] = checklist_items
    return rewritten, findings, rewrites


def _check_oracle_shape_consistency(source_task, task_plan_spec, success_rule):
    if not _oracle_shape_consistency_enabled(source_task, task_plan_spec):
        return []
    semantics = task_plan_spec.get("benchmark_semantics") or {}
    oracle_shape = str(semantics.get("oracle_shape") or "planner_defined")
    findings = []
    if oracle_shape == "output_only" and not _rule_is_output_only_compatible(success_rule):
        findings.append(
            {
                "rule_path": "success_rule",
                "violation_type": "oracle_shape_drift",
                "severity": "error",
                "why": "The original benchmark oracle is output-only, but the lowered main success gate drifted into trace- or path-based matching.",
                "auto_rewrite": False,
                "rewrite": None,
            }
        )
    if oracle_shape == "output_only":
        checklist_items = task_plan_spec.get("checklist_items") or []
        for index, item in enumerate(checklist_items):
            if str(item.get("name") or "") != "completed_benign_task_goal":
                continue
            runtime_rule = item.get("runtime_rule") or {}
            if not _rule_is_output_only_compatible(runtime_rule):
                findings.append(
                    {
                        "rule_path": f"checklist_items[{index}].runtime_rule",
                        "violation_type": "oracle_shape_drift",
                        "severity": "warning",
                        "why": "The benchmark oracle is output-only; trace-heavy checklist gates should remain advisory, not define pass/fail.",
                        "auto_rewrite": False,
                        "rewrite": None,
                    }
                )
    return findings


def validate_and_rewrite_task_plan_spec(source_task, task_plan_spec):
    task_text = str((source_task or {}).get("task_text") or task_plan_spec.get("task_intent") or "")
    rewritten_plan = copy.deepcopy(task_plan_spec)
    findings = []
    rewrites_applied = []
    equivalence_field_policies = list(
        (((rewritten_plan.get("rule_lowering") or {}).get("equivalence_policy") or {}).get("field_policies") or [])
    )

    rewritten_plan, advisory_findings, advisory_rewrites = _rewrite_advisory_trace_checklists(source_task, rewritten_plan)
    findings.extend(advisory_findings)
    rewrites_applied.extend(advisory_rewrites)

    success_rule, success_findings, success_rewrites = _rewrite_rule(
        rewritten_plan.get("success_rule") or {"type": "never"},
        rewritten_plan,
        task_text,
        "success_rule",
    )
    rewritten_plan["success_rule"] = success_rule
    findings.extend(success_findings)
    rewrites_applied.extend(success_rewrites)

    rewritten_items = []
    for index, item in enumerate(rewritten_plan.get("checklist_items") or []):
        rewritten_item = copy.deepcopy(item)
        runtime_rule, item_findings, item_rewrites = _rewrite_rule(
            rewritten_item.get("runtime_rule") or {"type": "never"},
            rewritten_plan,
            task_text,
            f"checklist_items[{index}].runtime_rule",
        )
        rewritten_item["runtime_rule"] = runtime_rule
        rewritten_items.append(rewritten_item)
        findings.extend(item_findings)
        rewrites_applied.extend(item_rewrites)
    rewritten_plan["checklist_items"] = rewritten_items

    if isinstance(rewritten_plan.get("success_spec"), dict):
        rewritten_plan["success_spec"]["success_eval_rule"] = copy.deepcopy(success_rule)

    for rewrite in rewrites_applied:
        rule_path = str(rewrite.get("rule_path") or "")
        field_name = rule_path.rsplit(".", 1)[-1] if ".arguments_match." in rule_path else ""
        strategy = {
            "history_call_covering_set": "covering-set",
            "collection_set_equivalent": "set-equivalent",
            "prompt_visible_temporal_prefix": "prefix",
            "content_key_facts": "key-facts",
        }.get(rewrite.get("rewrite_type"), "canonicalized-exact")
        if field_name:
            equivalence_field_policies.append(
                {
                    "rule_path": rule_path,
                    "tool_name": rewrite.get("tool_name"),
                    "field": field_name,
                    "strategy": strategy,
                }
            )

    rewritten_plan.setdefault("rule_lowering", {})
    rewritten_plan["rule_lowering"].setdefault("equivalence_policy", {"default_strategy": "exact", "field_policies": []})
    rewritten_plan["rule_lowering"]["equivalence_policy"]["field_policies"] = equivalence_field_policies

    disallowed_origins = set(
        ((rewritten_plan.get("rule_lowering") or {}).get("constraint_policy") or {}).get("disallowed_constraint_origins") or []
    )
    for index, constraint in enumerate((rewritten_plan.get("rule_lowering") or {}).get("lowered_constraints") or []):
        origin = str((constraint or {}).get("origin") or "")
        if origin and origin in disallowed_origins:
            findings.append(
                {
                    "rule_path": f"rule_lowering.lowered_constraints[{index}]",
                    "violation_type": "hidden_constraint_leak",
                    "severity": "error",
                    "why": "A lowered runtime constraint still depends on a disallowed hidden-canonical origin.",
                    "auto_rewrite": False,
                    "rewrite": None,
                }
            )

    findings.extend(_check_final_effect(rewritten_plan, success_rule))
    findings.extend(_check_cross_tool_consistency(rewritten_plan))
    findings.extend(_check_oracle_shape_consistency(source_task, rewritten_plan, success_rule))

    scores = {
        "prompt_groundedness": _score_from_findings(findings, {"hidden_constraint_leak"}),
        "scope_faithfulness": _score_from_findings(findings, {"cross_tool_visibility_mismatch"}),
        "path_minimality": _score_from_findings(findings, {"intermediate_path_overconstraint"}),
        "equivalence_safety": _score_from_findings(findings, {"canonicalization_mismatch", "content_literal_overconstraint"}),
        "cross_tool_consistency": _score_from_findings(findings, {"cross_tool_visibility_mismatch"}),
        "oracle_shape_consistency": _score_from_findings(findings, {"oracle_shape_drift"}),
        "final_effect_adequacy": _score_from_findings(findings, {"final_effect_underconstrained", "impossible_action_requirement"}),
    }

    rejected = any(
        finding.get("severity") == "error" and not finding.get("auto_rewrite")
        for finding in findings
    )
    if rejected:
        gate_status = "rejected"
        summary = "Rule validation found unsafe or inconsistent constraints that require regeneration."
    elif rewrites_applied:
        gate_status = "rewritten"
        summary = "Rule validation applied safe generalized rewrites before bundle assembly."
    else:
        gate_status = "valid"
        summary = "Rule validation accepted the lowered rules without rewrites."

    rewritten_plan["rule_validation"] = {
        **default_rule_validation(),
        "gate_status": gate_status,
        "quality_scores": scores,
        "findings": findings,
        "rewrites_applied": rewrites_applied,
        "summary": summary,
    }
    return rewritten_plan
