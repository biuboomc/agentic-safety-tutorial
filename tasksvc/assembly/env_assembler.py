import copy

from tasksvc.assembly.bundle_validator import validate_runtime_bundle
from tasksvc.assembly.bundle_validator import validate_task_draft
from tasksvc.common.contracts import EXECUTOR_SIGNATURE, RESPONSE_CONTRACT, default_risk_success_rule
from tasksvc.generation.generator import build_default_static_task_drafts, build_llm_static_task_drafts
from tasksvc.generation.task_safety_perturber import TaskSafetyPerturber
from tasksvc.rules.evaluation_hints import build_success_eval_rule


def assemble_runtime_bundle(task_draft):
    validate_task_draft(task_draft)
    planned_task = task_draft["planned_task"]
    agent_input_view = task_draft["agent_input_view"]
    state_draft = task_draft["state_draft"]
    utility_checklist_draft = task_draft["utility_checklist_draft"]
    risk_checklist_draft = task_draft["risk_checklist_draft"]
    risk_config = copy.deepcopy(agent_input_view["risk_placeholders"]["risk_config"])
    selected_tools = list(planned_task["selected_tools"])
    success_rule = copy.deepcopy(state_draft.get("success_rule") or build_success_eval_rule(state_draft["success_spec"]))
    risk_success_rule = copy.deepcopy(state_draft.get("risk_success_rule") or default_risk_success_rule())
    scenario_specs = copy.deepcopy(state_draft.get("scenarios") or {})

    dispatch_table = {}
    for tool_name in selected_tools:
        dispatch_table[tool_name] = {
            "source_key": tool_name,
            "entrypoint": "execute"
        }

    bundle = {
        "bundle_version": "v1",
        "task_spec": {
            "task_id": planned_task["task_id"],
            "domain": planned_task["domain"],
            "difficulty_tier": planned_task["difficulty_tier"],
            "user_query": agent_input_view["user_query"],
            "available_scenarios": sorted(scenario_specs.keys()) if scenario_specs else ["clean"],
            "default_scenario": "clean",
            "selected_tools": selected_tools,
            "evaluation_contract": copy.deepcopy(task_draft["task_plan_spec"].get("evaluation_contract", {})),
            "risk_config": risk_config,
            "risk_spec": copy.deepcopy(task_draft["task_plan_spec"].get("risk_spec", {})),
            "safety_perturbation": copy.deepcopy(task_draft.get("safety_perturbation_draft", {})),
            "task_metadata": copy.deepcopy(planned_task["task_metadata"]),
            "planner_trace": copy.deepcopy(planned_task["planner_trace"]),
        },
        "tool_registry_view": {
            "tool_schemas": copy.deepcopy(agent_input_view["tool_schemas"]),
            "allowed_tool_names": selected_tools,
        },
        "execution_bundle": {
            "tool_impl_sources": copy.deepcopy(task_draft["tool_code_drafts"]),
            "tool_entrypoints": copy.deepcopy(dispatch_table),
            "initial_state_template": copy.deepcopy(state_draft["initial_state_template"]),
            "scenarios": copy.deepcopy(scenario_specs),
            "success_spec": copy.deepcopy(state_draft["success_spec"]),
        },
        "evaluation_bundle": {
            "utility_evaluation_spec": copy.deepcopy(task_draft["evaluation_spec_draft"]),
            "utility_checklist": copy.deepcopy(utility_checklist_draft["items"]),
            "checklist_eval_hints": copy.deepcopy(utility_checklist_draft["checklist_eval_hints"]),
            "success_eval_rule": success_rule,
            "risk_evaluation_spec": copy.deepcopy(task_draft["risk_evaluation_spec_draft"]),
            "risk_checklist": copy.deepcopy(risk_checklist_draft["items"]),
            "risk_checklist_eval_hints": copy.deepcopy(risk_checklist_draft["checklist_eval_hints"]),
            "risk_success_eval_rule": risk_success_rule,
        },
        "executor_contract": {
            "signature": EXECUTOR_SIGNATURE
        },
        "response_contract": copy.deepcopy(RESPONSE_CONTRACT),
        "server_adapter_manifest": {
            "state_init_key": "execution_bundle.initial_state_template",
            "tool_dispatch_table": copy.deepcopy(dispatch_table),
            "success_eval_type": state_draft["success_spec"]["type"],
            "checklist_eval_type": state_draft["success_spec"]["type"],
        }
    }
    validate_runtime_bundle(bundle)
    return bundle


def assemble_runtime_catalog(task_drafts, perturbation_config=None, progress=None):
    perturber = TaskSafetyPerturber(config=perturbation_config)
    processed_drafts = perturber.apply_to_drafts(task_drafts)
    bundles = []
    phase = progress.phase("Assemble bundles", len(processed_drafts)) if progress else None
    for task_draft in processed_drafts:
        bundles.append(assemble_runtime_bundle(task_draft))
        if phase:
            phase.advance(detail=task_draft["planned_task"]["task_id"])
    if phase:
        phase.close()
    return {bundle["task_spec"]["task_id"]: bundle for bundle in bundles}


def build_default_runtime_catalog(target_domain=None, perturbation_config=None):
    return assemble_runtime_catalog(
        build_default_static_task_drafts(target_domain=target_domain),
        perturbation_config=perturbation_config,
    )


def build_llm_runtime_catalog(target_domain=None, config=None, perturbation_config=None):
    return assemble_runtime_catalog(
        build_llm_static_task_drafts(target_domain=target_domain, config=config),
        perturbation_config=perturbation_config,
    )
