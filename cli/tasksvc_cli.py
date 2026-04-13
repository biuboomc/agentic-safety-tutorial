import argparse
import copy
import inspect
import json
import os
import re
import socket
import subprocess
from pathlib import Path

from tasksvc.assembly.catalog_loader import export_json
from tasksvc.assembly.env_assembler import assemble_runtime_catalog
from tasksvc.common.progress import ProgressReporter
from tasksvc.generation.benchmark_source_utils import (
    build_benchmark_extraction_payload,
    build_tool_pool_from_source_tasks,
)
from tasksvc.generation.agentdojo_source_tasks import (
    DEFAULT_AGENTDOJO_BENCHMARK_VERSION,
    DEFAULT_AGENTDOJO_ATTACK_VARIANTS,
    extract_agentdojo_attack_specs,
    extract_agentdojo_attacked_source_tasks,
    extract_agentdojo_source_tasks,
    extract_agentdojo_utility_source_tasks,
)
from tasksvc.generation.agentsafetybench_source_tasks import (
    DEFAULT_AGENTSAFETYBENCH_VERSION,
    build_agentdyn_source_tasks_payload,
    build_agentsafetybench_source_tasks_payload,
    extract_agentdyn_source_tasks,
    extract_agentsafetybench_source_tasks,
)
from tasksvc.generation.asb_source_tasks import (
    DEFAULT_ASB_VERSION,
    build_asb_source_inventory,
    extract_asb_source_tasks,
)
from tasksvc.generation.attack_wrapper_expander import (
    expand_paired_bundle_to_wrapper_variant,
    expand_paired_draft_to_wrapper_variant,
    expand_paired_source_task_to_wrapper_variant,
    expand_paired_artifacts_to_attack_wrapped_variants,
)
from tasksvc.generation.attack_materializer import materialize_source_tasks_with_attack_specs
from tasksvc.generation.generator import (
    build_default_static_task_drafts,
    build_llm_static_task_drafts,
    get_default_tool_pool,
    load_tool_pool,
)
from tasksvc.generation.llm_client import OpenAICompatClient
from tasksvc.generation.llm_generator import LLMGenerationConfig
from tasksvc.generation.mtagentrisk_source_tasks import (
    DEFAULT_MTAGENTRISK_VERSION,
    build_mtagentrisk_source_task_payload,
    extract_mtagentrisk_source_tasks,
)
from tasksvc.generation.source_task_converter import load_source_tasks, convert_source_tasks
from tasksvc.runtime.agent_rollout import (
    AgentRolloutConfig,
    BENCHMARK_FAITHFUL_SYSTEM_PROMPT,
    run_agent_episode,
)
from tasksvc.runtime.batch_rollout import BatchRolloutConfig, run_agent_batch


DEFAULT_PJLAB_HOST = "h.pjlab.org.cn"
DEFAULT_PJLAB_USER = "chenguanxu2.chenguanxu.ailab-ai4good1.ws"
DEFAULT_PJLAB_IDENTITY_FILE = str(Path.home() / ".ssh" / "id_ed25519")
DEFAULT_PJLAB_CPU_RLAUNCH = (
    "rlaunch --memory=16000 --cpu=8 --charged-group=ai4good1_cpu_task "
    "--mount=gpfs://gpfs1/chenguanxu:/mnt/shared-storage-user/chenguanxu "
    "--mount=gpfs://gpfs1/ai4good1-share:/mnt/shared-storage-user/ai4good1-share "
    "--mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public "
    "--image=registry.h.pjlab.org.cn/ailab-ai4good1-ai4good1_gpu/chenguanxu_images:vllm_0140_fla -- bash"
)


def _resolve_rollout_prompt_and_profile(system_prompt, protocol_profile, benchmark_faithful_agent):
    default_rollout_prompt = AgentRolloutConfig().system_prompt
    resolved_prompt = system_prompt
    if benchmark_faithful_agent and system_prompt == default_rollout_prompt:
        resolved_prompt = BENCHMARK_FAITHFUL_SYSTEM_PROMPT
    resolved_profile = protocol_profile
    if benchmark_faithful_agent and resolved_profile == "structured_json":
        resolved_profile = "benchmark_faithful"
    return resolved_prompt, resolved_profile


def _build_openai_client(llm_config):
    kwargs = {
        "base_url": llm_config.base_url,
        "model": llm_config.model,
        "api_key": llm_config.api_key,
        "timeout": llm_config.timeout,
    }
    optional_kwargs = {
        "user_agent": getattr(llm_config, "user_agent", ""),
        "proxy_url": getattr(llm_config, "proxy_url", ""),
        "error_log_dir": getattr(llm_config, "error_log_dir", ""),
        "trace_log_dir": getattr(llm_config, "trace_log_dir", ""),
        "max_retries": getattr(llm_config, "max_retries", None),
    }
    try:
        accepted = set(inspect.signature(OpenAICompatClient.__init__).parameters)
    except (TypeError, ValueError):
        accepted = set()
    for key, value in optional_kwargs.items():
        if key in accepted:
            kwargs[key] = value
    return OpenAICompatClient(**kwargs)


def _build_llm_config_from_args(args):
    config = LLMGenerationConfig()
    if getattr(args, "llm_base_url", None):
        config.base_url = args.llm_base_url
    if getattr(args, "llm_model", None):
        config.model = args.llm_model
    if getattr(args, "llm_api_key", None) is not None:
        config.api_key = args.llm_api_key
    if getattr(args, "llm_user_agent", None):
        config.user_agent = args.llm_user_agent
    if getattr(args, "llm_proxy_url", None):
        config.proxy_url = args.llm_proxy_url
    if getattr(args, "planner_llm_base_url", None):
        config.planner_base_url = args.planner_llm_base_url
    if getattr(args, "planner_llm_model", None):
        config.planner_model = args.planner_llm_model
    if getattr(args, "planner_llm_api_key", None) is not None:
        config.planner_api_key = args.planner_llm_api_key
    if getattr(args, "planner_llm_user_agent", None):
        config.planner_user_agent = args.planner_llm_user_agent
    if getattr(args, "planner_llm_proxy_url", None):
        config.planner_proxy_url = args.planner_llm_proxy_url
    if getattr(args, "llm_error_log_dir", None):
        config.error_log_dir = args.llm_error_log_dir
    if getattr(args, "llm_trace_log_dir", None):
        config.trace_log_dir = args.llm_trace_log_dir
    if getattr(args, "llm_timeout", None) is not None:
        config.timeout = args.llm_timeout
    if getattr(args, "planner_llm_timeout", None) is not None:
        config.planner_timeout = args.planner_llm_timeout
    if getattr(args, "llm_max_retries", None) is not None:
        config.max_retries = args.llm_max_retries
    if getattr(args, "planner_llm_max_retries", None) is not None:
        config.planner_max_retries = args.planner_llm_max_retries
    if getattr(args, "require_planner_success", False):
        config.require_planner_success = True
    if getattr(args, "llm_temperature", None) is not None:
        config.temperature = args.llm_temperature
    if getattr(args, "plan_max_tokens", None) is not None:
        config.plan_max_tokens = args.plan_max_tokens
    if getattr(args, "query_max_tokens", None) is not None:
        config.query_max_tokens = args.query_max_tokens
    if getattr(args, "checklist_max_tokens", None) is not None:
        config.checklist_max_tokens = args.checklist_max_tokens
    if getattr(args, "tool_code_max_tokens", None) is not None:
        config.tool_code_max_tokens = args.tool_code_max_tokens
    if getattr(args, "task_parallelism", None) is not None:
        config.task_parallelism = max(1, int(args.task_parallelism))
    if getattr(args, "tool_parallelism", None) is not None:
        config.tool_parallelism = max(1, int(args.tool_parallelism))
    if getattr(args, "query_checklist_parallelism", None) is not None:
        config.query_checklist_parallelism = max(1, int(args.query_checklist_parallelism))
    if getattr(args, "inherit_base_checkpoint_dir", None):
        config.base_reuse_checkpoint_dir = args.inherit_base_checkpoint_dir
    return config


def _load_generation_tool_pool(args):
    if getattr(args, "tool_pool", None):
        return load_tool_pool(args.tool_pool, target_domain=args.domain)
    return get_default_tool_pool(args.domain)


def _build_task_drafts(backend, tool_pool, target_domain=None, llm_config=None, num_tasks=None, seed=7, progress=None):
    if backend == "llm":
        return build_llm_static_task_drafts(
            target_domain=target_domain,
            config=llm_config,
            num_tasks=num_tasks,
            seed=seed,
            tool_pool=tool_pool,
            progress=progress,
        )
    return build_default_static_task_drafts(
        target_domain=target_domain,
        num_tasks=num_tasks,
        seed=seed,
        tool_pool=tool_pool,
        progress=progress,
    )


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _probe_tcp_port(host, port, timeout_seconds):
    timeout_seconds = max(1, int(timeout_seconds))
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_seconds):
            return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _resolve_identity_file(path):
    candidate = str(path or "").strip() or DEFAULT_PJLAB_IDENTITY_FILE
    expanded = Path(os.path.expandvars(os.path.expanduser(candidate)))
    return str(expanded)


def _ssh_base_command(args):
    command = ["ssh"]
    if not getattr(args, "use_ssh_config", False):
        command.extend(["-F", "NUL"])
    command.extend(
        [
            "-T",
            "-o",
            f"ConnectTimeout={max(1, int(args.connect_timeout))}",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=1",
            "-o",
            "UpdateHostKeys=no",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
        ]
    )
    identity_file = _resolve_identity_file(getattr(args, "identity_file", None))
    if identity_file:
        command.extend(["-i", identity_file])
    return command


def _run_ssh_probe(args, target, remote_command="echo ok"):
    command = _ssh_base_command(args) + [target, remote_command]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(5, int(args.ssh_timeout)),
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "timed_out": True,
            "returncode": None,
            "stdout": (exc.stdout or "").strip(),
            "stderr": (exc.stderr or "").strip(),
            "command": command,
        }
    return {
        "ok": completed.returncode == 0,
        "timed_out": False,
        "returncode": completed.returncode,
        "stdout": (completed.stdout or "").strip(),
        "stderr": (completed.stderr or "").strip(),
        "command": command,
    }


def _build_worker_ssh_target(worker_name, worker_owner="chenguanxu", pod_user="root", pod_group="ailab-ai4good1"):
    worker_name = str(worker_name or "").strip()
    if not worker_name:
        return ""
    if "@" in worker_name:
        return worker_name
    if "." in worker_name and worker_name.endswith(".pod"):
        return f"{worker_name}@{DEFAULT_PJLAB_HOST}"
    return f"{worker_name}.{worker_owner}+{pod_user}.{pod_group}.pod@{DEFAULT_PJLAB_HOST}"


def _extract_worker_name(text):
    match = re.search(r"(ws-[A-Za-z0-9-]+)", str(text or ""))
    return match.group(1) if match else None


def _default_rlaunch_template():
    return DEFAULT_PJLAB_CPU_RLAUNCH


def cmd_remote_worker_preflight(args):
    payload = {
        "dev_host": {
            "host": args.host,
            "port": args.port,
            "user": args.user,
            "tcp": _probe_tcp_port(args.host, args.port, args.connect_timeout),
        },
        "worker": None,
        "launch": None,
    }

    dev_target = f"{args.user}@{args.host}"
    payload["dev_host"]["ssh"] = _run_ssh_probe(args, dev_target, args.dev_host_command)

    worker_name = str(getattr(args, "worker_name", "") or "").strip()
    if worker_name:
        worker_target = _build_worker_ssh_target(
            worker_name,
            worker_owner=args.worker_owner,
            pod_user=args.worker_pod_user,
            pod_group=args.worker_pod_group,
        )
        payload["worker"] = {
            "worker_name": worker_name,
            "target": worker_target,
            "ssh": _run_ssh_probe(args, worker_target, args.worker_command),
        }

    should_launch = bool(args.launch_if_missing)
    worker_missing = payload["worker"] is None or not bool((payload["worker"] or {}).get("ssh", {}).get("ok"))
    if should_launch and worker_missing:
        launch_command = args.launch_command or _default_rlaunch_template()
        payload["launch"] = {"requested": True, "command": launch_command}
        if payload["dev_host"]["ssh"]["ok"]:
            launch_result = _run_ssh_probe(args, dev_target, launch_command)
            launched_worker_name = _extract_worker_name(launch_result.get("stdout"))
            payload["launch"]["result"] = launch_result
            if launched_worker_name:
                payload["launch"]["worker_name"] = launched_worker_name
                launched_target = _build_worker_ssh_target(
                    launched_worker_name,
                    worker_owner=args.worker_owner,
                    pod_user=args.worker_pod_user,
                    pod_group=args.worker_pod_group,
                )
                payload["worker"] = {
                    "worker_name": launched_worker_name,
                    "target": launched_target,
                    "ssh": _run_ssh_probe(args, launched_target, args.worker_command),
                }
        else:
            payload["launch"]["skipped_reason"] = "dev_host_ssh_unavailable"
    elif should_launch:
        payload["launch"] = {"requested": True, "skipped_reason": "existing_worker_healthy"}

    if payload["worker"] and payload["worker"]["ssh"]["ok"]:
        payload["recommended_next_action"] = "worker_ready"
    elif payload["dev_host"]["ssh"]["ok"]:
        payload["recommended_next_action"] = "launch_worker_or_fix_worker_target"
    elif payload["dev_host"]["tcp"]["ok"]:
        payload["recommended_next_action"] = "fix_dev_host_ssh"
    else:
        payload["recommended_next_action"] = "fix_network_path_to_dev_host"

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_task_drafts_from_file(path):
    payload = _read_json(path)
    if isinstance(payload, dict) and "task_drafts" in payload:
        drafts = payload["task_drafts"]
    elif isinstance(payload, list):
        drafts = payload
    else:
        raise ValueError("Draft file must be a JSON list or an object with a task_drafts field.")
    if not isinstance(drafts, list) or not drafts:
        raise ValueError("task_drafts must be a non-empty list.")
    return drafts


def _write_payload(output, payload):
    if output:
        export_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _resolve_checkpoint_dir(args):
    explicit = getattr(args, "checkpoint_dir", None)
    if explicit:
        return explicit
    output_dir = getattr(args, "output_dir", None)
    output_prefix = getattr(args, "output_prefix", None)
    backend = getattr(args, "backend", None)
    if output_dir and output_prefix and backend == "llm":
        return str(Path(output_dir) / f"{output_prefix}_checkpoints")
    return None


def _infer_agentdojo_run_root_from_output_dir(output_dir):
    output_dir = Path(output_dir).resolve()
    return output_dir.parent.parent if output_dir.name == "convert" else output_dir.parent


def _infer_paired_convert_dir(args):
    explicit = getattr(args, "paired_convert_dir", None)
    if explicit:
        return Path(explicit).resolve()
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        return None
    return _infer_agentdojo_run_root_from_output_dir(output_dir) / "paired949" / "convert"


def _load_agentdojo_paired_artifacts_for_attack_wrapping(args):
    paired_dir = _infer_paired_convert_dir(args)
    if paired_dir is None:
        return None
    prefix = getattr(args, "paired_output_prefix", None) or "agentdojo_v12_utility_attack_paired"
    source_tasks_path = paired_dir / f"{prefix}_source_tasks.json"
    drafts_path = paired_dir / f"{prefix}_drafts.json"
    bundles_path = paired_dir / f"{prefix}_runtime_catalog.json"
    if not source_tasks_path.exists() or not drafts_path.exists() or not bundles_path.exists():
        return None
    source_payload = _read_json(source_tasks_path)
    drafts_payload = _read_json(drafts_path)
    bundles_payload = _read_json(bundles_path)
    return {
        "paired_convert_dir": str(paired_dir),
        "source_tasks": list(source_payload.get("source_tasks") or []),
        "task_drafts": list(drafts_payload.get("task_drafts") or []),
        "runtime_catalog": dict(bundles_payload.get("runtime_catalog") or {}),
    }


def _json_meta_prefix(meta_items, trailing_field):
    parts = [f"{json.dumps(str(key))}: {json.dumps(value, ensure_ascii=False)}" for key, value in meta_items]
    if trailing_field:
        parts.append(f"{json.dumps(str(trailing_field))}: ")
    return "{%s" % (", ".join(parts))


def _stream_list_payload(output_path, meta, field_name, shard_paths, shard_field):
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.tmp")
    meta_items = [(key, value) for key, value in meta.items() if key != field_name]
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(_json_meta_prefix(meta_items, field_name))
        handle.write("[")
        first = True
        for shard_path in shard_paths:
            payload = _read_json(shard_path)
            for item in payload.get(shard_field) or []:
                if not first:
                    handle.write(",")
                handle.write("\n")
                handle.write(json.dumps(item, ensure_ascii=False, indent=2))
                first = False
        if not first:
            handle.write("\n")
        handle.write("]}\n")
    tmp_path.replace(target)


def _stream_dict_payload(output_path, meta, field_name, shard_paths, shard_field):
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.tmp")
    meta_items = [(key, value) for key, value in meta.items() if key != field_name]
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(_json_meta_prefix(meta_items, field_name))
        handle.write("{")
        first = True
        for shard_path in shard_paths:
            payload = _read_json(shard_path)
            for key, value in (payload.get(shard_field) or {}).items():
                if not first:
                    handle.write(",")
                handle.write("\n")
                handle.write(f"{json.dumps(str(key))}: ")
                handle.write(json.dumps(value, ensure_ascii=False, indent=2))
                first = False
        if not first:
            handle.write("\n")
        handle.write("}}\n")
    tmp_path.replace(target)


def _wrapper_expansion_checkpoint_dir(args):
    resolved = _resolve_checkpoint_dir(args)
    if resolved:
        return Path(resolved)
    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        return Path(output_dir).parent / "wrapper_checkpoints"
    return None


def _load_wrapper_expansion_manifest(manifest_path):
    if not manifest_path.exists():
        return None
    return _read_json(manifest_path)


def _write_wrapper_expansion_manifest(manifest_path, payload):
    export_json(manifest_path, payload)


def _run_checkpointed_attack_wrapper_expansion(args, paired_artifacts, progress):
    checkpoint_dir = _wrapper_expansion_checkpoint_dir(args)
    if checkpoint_dir is None or not args.output_dir:
        return None

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = checkpoint_dir / "wrapper_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / "wrapper_expansion_manifest.json"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"agentdojo_{args.benchmark_version.replace('.', '_')}"
    source_tasks_path = output_dir / f"{prefix}_source_tasks.json"
    drafts_path = output_dir / f"{prefix}_drafts.json"
    bundles_path = output_dir / f"{prefix}_runtime_catalog.json"

    variants = copy.deepcopy(DEFAULT_AGENTDOJO_ATTACK_VARIANTS)
    source_tasks = paired_artifacts["source_tasks"]
    task_drafts = paired_artifacts["task_drafts"]
    runtime_catalog = paired_artifacts["runtime_catalog"]
    draft_by_id = {
        str((draft.get("source_task_input") or {}).get("source_task_id") or ""): draft
        for draft in task_drafts or []
    }
    bundle_by_id = {str(task_id): bundle for task_id, bundle in (runtime_catalog or {}).items()}
    total_count = len(source_tasks) * len(variants)
    if args.limit is not None:
        total_count = min(total_count, int(args.limit))
    # Keep the wrapper-only path memory-friendly: persist each completed expansion
    # immediately instead of accumulating many large copied bundles in memory.
    shard_size = 1

    manifest = _load_wrapper_expansion_manifest(manifest_path) if getattr(args, "resume", False) else None
    completed_count = int((manifest or {}).get("completed_count") or 0)
    shard_count = int((manifest or {}).get("shard_count") or 0)
    source_shards = list((manifest or {}).get("source_task_shards") or [])
    draft_shards = list((manifest or {}).get("draft_shards") or [])
    bundle_shards = list((manifest or {}).get("bundle_shards") or [])

    phase = progress.phase("Expand attacked wrappers", total_count) if progress and total_count else None
    if phase and completed_count:
        phase.completed = min(total_count, completed_count)
        phase._render(detail="resume")

    batch_source_tasks = []
    batch_drafts = []
    batch_bundles = {}

    def flush_batch():
        nonlocal shard_count, batch_source_tasks, batch_drafts, batch_bundles, source_shards, draft_shards, bundle_shards
        if not batch_source_tasks:
            return
        shard_name = f"shard_{shard_count:05d}"
        source_path = shard_dir / f"{shard_name}_source_tasks.json"
        draft_path = shard_dir / f"{shard_name}_drafts.json"
        bundle_path = shard_dir / f"{shard_name}_runtime_catalog.json"
        export_json(source_path, {"source_tasks": batch_source_tasks})
        export_json(draft_path, {"task_drafts": batch_drafts})
        export_json(bundle_path, {"runtime_catalog": batch_bundles})
        source_shards.append(str(source_path))
        draft_shards.append(str(draft_path))
        bundle_shards.append(str(bundle_path))
        shard_count += 1
        batch_source_tasks = []
        batch_drafts = []
        batch_bundles = {}
        _write_wrapper_expansion_manifest(
            manifest_path,
            {
                "version": 1,
                "mode": "paired_wrapper_only",
                "status": "running",
                "total_count": total_count,
                "completed_count": completed_count,
                "shard_count": shard_count,
                "source_task_shards": source_shards,
                "draft_shards": draft_shards,
                "bundle_shards": bundle_shards,
                "output_prefix": prefix,
                "derived_from_paired_convert_dir": paired_artifacts["paired_convert_dir"],
            },
        )

    global_index = 0
    for source_task in source_tasks:
        source_task_id = str(source_task.get("source_task_id") or "")
        base_draft = draft_by_id.get(source_task_id)
        base_bundle = bundle_by_id.get(source_task_id)
        if base_draft is None or base_bundle is None:
            raise ValueError(f"Missing paired draft or bundle for source_task_id={source_task_id}")
        for variant in variants:
            if global_index >= total_count:
                break
            if global_index < completed_count:
                global_index += 1
                continue
            wrapped_source_task = expand_paired_source_task_to_wrapper_variant(source_task, variant)
            wrapped_draft = expand_paired_draft_to_wrapper_variant(base_draft, wrapped_source_task)
            wrapped_bundle = expand_paired_bundle_to_wrapper_variant(base_bundle, wrapped_source_task)
            batch_source_tasks.append(wrapped_source_task)
            batch_drafts.append(wrapped_draft)
            batch_bundles[wrapped_bundle["task_spec"]["task_id"]] = wrapped_bundle
            global_index += 1
            completed_count += 1
            if phase:
                phase.advance(detail=wrapped_bundle["task_spec"]["task_id"])
            if len(batch_source_tasks) >= shard_size:
                flush_batch()
        if global_index >= total_count:
            break

    flush_batch()
    if phase:
        phase.close()

    source_meta = {
        "benchmark": "agentdojo",
        "benchmark_version": args.benchmark_version,
        "extraction_track": args.extraction_track,
        "derived_from_paired_convert_dir": paired_artifacts["paired_convert_dir"],
        "expansion_mode": "paired_wrapper_only",
    }
    drafts_meta = dict(source_meta)
    drafts_meta["source_tasks"] = None
    bundles_meta = {
        "benchmark": "agentdojo",
        "benchmark_version": args.benchmark_version,
        "extraction_track": args.extraction_track,
        "derived_from_paired_convert_dir": paired_artifacts["paired_convert_dir"],
        "expansion_mode": "paired_wrapper_only",
    }
    _stream_list_payload(source_tasks_path, source_meta, "source_tasks", source_shards, "source_tasks")
    _stream_list_payload(
        drafts_path,
        {
            "benchmark": "agentdojo",
            "benchmark_version": args.benchmark_version,
            "extraction_track": args.extraction_track,
            "derived_from_paired_convert_dir": paired_artifacts["paired_convert_dir"],
            "expansion_mode": "paired_wrapper_only",
        },
        "task_drafts",
        draft_shards,
        "task_drafts",
    )
    _stream_dict_payload(bundles_path, bundles_meta, "runtime_catalog", bundle_shards, "runtime_catalog")

    _write_wrapper_expansion_manifest(
        manifest_path,
        {
            "version": 1,
            "mode": "paired_wrapper_only",
            "status": "completed",
            "total_count": total_count,
            "completed_count": completed_count,
            "shard_count": shard_count,
            "source_task_shards": source_shards,
            "draft_shards": draft_shards,
            "bundle_shards": bundle_shards,
            "output_prefix": prefix,
            "derived_from_paired_convert_dir": paired_artifacts["paired_convert_dir"],
            "source_tasks_output": str(source_tasks_path),
            "drafts_output": str(drafts_path),
            "runtime_catalog_output": str(bundles_path),
        },
    )
    return {
        "benchmark": "agentdojo",
        "benchmark_version": args.benchmark_version,
        "extraction_track": args.extraction_track,
        "source_task_count": total_count,
        "derived_from_paired_convert_dir": paired_artifacts["paired_convert_dir"],
        "expansion_mode": "paired_wrapper_only",
        "source_tasks_output": str(source_tasks_path),
        "drafts_output": str(drafts_path),
        "runtime_catalog_output": str(bundles_path),
        "checkpoint_manifest": str(manifest_path),
    }


def _summarize_llm_usage(drafts):
    summary = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    found = False
    for draft in drafts:
        usage = draft.get("llm_generation_usage")
        if not isinstance(usage, dict):
            continue
        found = True
        summary["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        summary["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        summary["total_tokens"] += int(usage.get("total_tokens") or 0)
    return summary if found else None


def _add_common_generation_args(parser):
    parser.add_argument("--backend", choices=["placeholder", "llm"], default="placeholder")
    parser.add_argument("--tool-pool", default=None, help="Path to a JSON file containing tool_pool.")
    parser.add_argument("--domain", default=None, help="Optional domain filter. If omitted, tasks sample from all domains.")
    parser.add_argument("--num-tasks", type=int, default=None, help="Number of tasks to generate. Each task samples a domain.")
    parser.add_argument("--seed", type=int, default=7, help="Sampling seed for placeholder/base task generation.")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-user-agent", default=None)
    parser.add_argument("--llm-proxy-url", default=None)
    parser.add_argument("--planner-llm-base-url", default=None)
    parser.add_argument("--planner-llm-model", default=None)
    parser.add_argument("--planner-llm-api-key", default=None)
    parser.add_argument("--planner-llm-user-agent", default=None)
    parser.add_argument("--planner-llm-proxy-url", default=None)
    parser.add_argument("--llm-error-log-dir", default=None)
    parser.add_argument("--llm-trace-log-dir", default=None)
    parser.add_argument("--llm-timeout", type=int, default=None)
    parser.add_argument("--llm-max-retries", type=int, default=None)
    parser.add_argument("--planner-llm-timeout", type=int, default=None)
    parser.add_argument("--planner-llm-max-retries", type=int, default=None)
    parser.add_argument("--require-planner-success", action="store_true")
    parser.add_argument("--llm-temperature", type=float, default=None)
    parser.add_argument("--plan-max-tokens", type=int, default=None)
    parser.add_argument("--query-max-tokens", type=int, default=None)
    parser.add_argument("--checklist-max-tokens", type=int, default=None)
    parser.add_argument("--tool-code-max-tokens", type=int, default=None)
    parser.add_argument("--task-parallelism", type=int, default=None, help="Max parallel drafts to augment at once.")
    parser.add_argument("--tool-parallelism", type=int, default=None, help="Max parallel tool code generations within one task.")
    parser.add_argument(
        "--query-checklist-parallelism",
        type=int,
        default=None,
        help="Max parallel independent query/checklist generations within one task.",
    )
    parser.add_argument(
        "--inherit-base-checkpoint-dir",
        default=None,
        help="Optional checkpoint dir containing completed base utility drafts to reuse for planner/tool-code/benign checklist outputs.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable stderr progress bars.")


def cmd_dump_tool_pool(args):
    payload = {"tool_pool": _load_generation_tool_pool(args)}
    _write_payload(args.output, payload)


def cmd_generate_drafts(args):
    tool_pool = _load_generation_tool_pool(args)
    llm_config = _build_llm_config_from_args(args) if args.backend == "llm" else None
    progress = ProgressReporter(enabled=not args.no_progress)
    drafts = _build_task_drafts(
        args.backend,
        tool_pool,
        target_domain=args.domain,
        llm_config=llm_config,
        num_tasks=args.num_tasks,
        seed=args.seed,
        progress=progress,
    )
    payload = {
        "tool_pool": tool_pool,
        "task_drafts": drafts,
    }
    usage_summary = _summarize_llm_usage(drafts)
    if usage_summary:
        payload["llm_usage_summary"] = usage_summary
    _write_payload(args.output, payload)


def cmd_assemble_bundles(args):
    progress = ProgressReporter(enabled=not args.no_progress)
    if args.drafts_file:
        drafts = _load_task_drafts_from_file(args.drafts_file)
    else:
        tool_pool = _load_generation_tool_pool(args)
        llm_config = _build_llm_config_from_args(args) if args.backend == "llm" else None
        drafts = _build_task_drafts(
            args.backend,
            tool_pool,
            target_domain=args.domain,
            llm_config=llm_config,
            num_tasks=args.num_tasks,
            seed=args.seed,
            progress=progress,
        )
    payload = {"runtime_catalog": assemble_runtime_catalog(drafts, progress=progress)}
    usage_summary = _summarize_llm_usage(drafts)
    if usage_summary:
        payload["llm_usage_summary"] = usage_summary
    _write_payload(args.output, payload)


def cmd_generate_examples(args):
    tool_pool = _load_generation_tool_pool(args)
    llm_config = _build_llm_config_from_args(args) if args.backend == "llm" else None
    progress = ProgressReporter(enabled=not args.no_progress)
    drafts = _build_task_drafts(
        args.backend,
        tool_pool,
        target_domain=args.domain,
        llm_config=llm_config,
        num_tasks=args.num_tasks,
        seed=args.seed,
        progress=progress,
    )
    runtime_catalog = assemble_runtime_catalog(drafts, progress=progress)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"generated_{args.backend}"
    if args.domain:
        prefix = f"{prefix}_{args.domain}"

    drafts_path = output_dir / f"{prefix}_tool_pool_examples.json"
    bundles_path = output_dir / f"{prefix}_runtime_catalog.json"

    export_json(drafts_path, {"tool_pool": tool_pool, "task_drafts": drafts})
    export_json(bundles_path, {"runtime_catalog": runtime_catalog})

    payload = {
        "ok": True,
        "backend": args.backend,
        "domain": args.domain,
        "tool_count": len(tool_pool),
        "task_count": len(drafts),
        "task_ids": [draft["planned_task"]["task_id"] for draft in drafts],
        "drafts_output": str(drafts_path),
        "runtime_catalog_output": str(bundles_path),
    }
    usage_summary = _summarize_llm_usage(drafts)
    if usage_summary:
        payload["llm_usage_summary"] = usage_summary
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_convert_source_tasks(args):
    source_tasks = load_source_tasks(args.source_tasks_file)
    llm_config = _build_llm_config_from_args(args) if args.backend == "llm" else None
    progress = ProgressReporter(enabled=not args.no_progress)
    drafts = convert_source_tasks(
        source_tasks,
        backend=args.backend,
        config=llm_config,
        progress=progress,
        checkpoint_dir=_resolve_checkpoint_dir(args),
        resume=getattr(args, "resume", False),
    )
    runtime_catalog = assemble_runtime_catalog(drafts, progress=progress)

    payload = {
        "source_tasks": source_tasks,
        "task_drafts": drafts,
        "runtime_catalog": runtime_catalog,
    }
    usage_summary = _summarize_llm_usage(drafts)
    if usage_summary:
        payload["llm_usage_summary"] = usage_summary

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        drafts_path = output_dir / f"{args.output_prefix}_drafts.json"
        bundles_path = output_dir / f"{args.output_prefix}_runtime_catalog.json"
        export_json(drafts_path, {"source_tasks": source_tasks, "task_drafts": drafts, **({"llm_usage_summary": usage_summary} if usage_summary else {})})
        export_json(bundles_path, {"runtime_catalog": runtime_catalog, **({"llm_usage_summary": usage_summary} if usage_summary else {})})
        payload["drafts_output"] = str(drafts_path)
        payload["runtime_catalog_output"] = str(bundles_path)

    _write_payload(args.output, payload)


def cmd_extract_benchmark_source_tasks(args):
    benchmark = str(args.benchmark).strip().lower()
    if benchmark in {"agent-safety-bench", "agentsafetybench"}:
        benchmark_version = args.benchmark_version or DEFAULT_AGENTSAFETYBENCH_VERSION
        benchmark_name = "agent-safety-bench"
        specific_payload = build_agentsafetybench_source_tasks_payload(
            args.repo_path,
            benchmark_name=benchmark_name,
            benchmark_version=benchmark_version,
            track=args.extraction_track,
            limit=args.limit,
        )
    elif benchmark == "agentdyn":
        benchmark_version = args.benchmark_version or DEFAULT_AGENTSAFETYBENCH_VERSION
        benchmark_name = "agentdyn"
        specific_payload = build_agentdyn_source_tasks_payload(
            args.repo_path,
            benchmark_version=benchmark_version,
            track=args.extraction_track,
            limit=args.limit,
        )
    elif benchmark == "asb":
        benchmark_version = args.benchmark_version or DEFAULT_ASB_VERSION
        benchmark_name = "asb"
        specific_payload = build_asb_source_inventory(
            args.repo_path,
            benchmark_version=benchmark_version,
            track=args.extraction_track,
            limit=args.limit,
            attack_method=args.attack_method,
        )
    elif benchmark in {"mtagentrisk", "toolshield"}:
        benchmark_version = args.benchmark_version or DEFAULT_MTAGENTRISK_VERSION
        benchmark_name = "mtagentrisk"
        specific_payload = build_mtagentrisk_source_task_payload(
            args.repo_path,
            benchmark_version=benchmark_version,
            track=args.extraction_track,
            limit=args.limit,
        )
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")
    source_tasks = list(specific_payload.get("source_tasks") or [])
    tool_pool = specific_payload.get("tool_pool") or build_tool_pool_from_source_tasks(source_tasks)
    benchmark_accounting_summary = build_benchmark_extraction_payload(
        benchmark=benchmark_name,
        benchmark_version=benchmark_version,
        extraction_track=args.extraction_track,
        source_tasks=source_tasks,
        tool_pool=tool_pool,
    )["benchmark_accounting_summary"]
    payload = build_benchmark_extraction_payload(
        benchmark=benchmark_name,
        benchmark_version=benchmark_version,
        extraction_track=args.extraction_track,
        source_tasks=source_tasks,
        tool_pool=tool_pool,
        extra_fields={
            "attack_method": args.attack_method,
            "accounting_summary": specific_payload.get("accounting_summary"),
            "benchmark_accounting_summary": benchmark_accounting_summary,
        }
        if benchmark_name == "asb"
        else {
            "accounting_summary": specific_payload.get("accounting_summary"),
            "benchmark_accounting_summary": benchmark_accounting_summary,
        },
    )
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.output_prefix or benchmark_name.replace("-", "_")
        source_tasks_path = output_dir / f"{prefix}_source_tasks.json"
        tool_pool_path = output_dir / f"{prefix}_tool_pool.json"
        export_json(
            source_tasks_path,
            {
                "benchmark": benchmark_name,
                "benchmark_version": benchmark_version,
                "extraction_track": args.extraction_track,
                "source_task_count": len(source_tasks),
                "source_tasks": source_tasks,
                "accounting_summary": payload.get("accounting_summary"),
                "benchmark_accounting_summary": benchmark_accounting_summary,
            },
        )
        export_json(
            tool_pool_path,
            {
                "benchmark": benchmark_name,
                "benchmark_version": benchmark_version,
                "extraction_track": args.extraction_track,
                "tool_pool_count": len(tool_pool),
                "tool_pool": tool_pool,
                "accounting_summary": payload.get("accounting_summary"),
                "benchmark_accounting_summary": benchmark_accounting_summary,
            },
        )
        payload["source_tasks_output"] = str(source_tasks_path)
        payload["tool_pool_output"] = str(tool_pool_path)
    _write_payload(args.output, payload)


def cmd_convert_agentdojo_benchmark(args):
    llm_config = _build_llm_config_from_args(args) if args.backend == "llm" else None
    if (
        llm_config is not None
        and not str(getattr(llm_config, "base_reuse_checkpoint_dir", "") or "").strip()
        and args.extraction_track in {"utility_attack_paired", "attacked_4x"}
        and args.output_dir
    ):
        output_dir = Path(args.output_dir).resolve()
        run_root = output_dir.parent.parent if output_dir.name == "convert" else output_dir.parent
        candidate = run_root / "utility97" / "checkpoints"
        if candidate.exists():
            llm_config.base_reuse_checkpoint_dir = str(candidate)
    progress = ProgressReporter(enabled=not args.no_progress)
    if args.extraction_track == "utility_clean":
        source_tasks = extract_agentdojo_utility_source_tasks(
            repo_path=args.agentdojo_repo,
            benchmark_version=args.benchmark_version,
            suite_names=args.suites,
            limit=args.limit,
            tool_scope_mode=args.tool_scope_mode,
        )
        attack_specs = []
    elif args.extraction_track == "utility_attack_paired":
        clean_source_tasks = extract_agentdojo_utility_source_tasks(
            repo_path=args.agentdojo_repo,
            benchmark_version=args.benchmark_version,
            suite_names=args.suites,
            limit=None,
            include_injection_tasks=False,
            tool_scope_mode=args.tool_scope_mode,
        )
        attack_specs = extract_agentdojo_attack_specs(
            repo_path=args.agentdojo_repo,
            benchmark_version=args.benchmark_version,
            suite_names=args.suites,
            limit=None,
            tool_scope_mode=args.tool_scope_mode,
        )
        source_tasks = materialize_source_tasks_with_attack_specs(
            clean_source_tasks,
            attack_specs,
            limit=args.limit,
            progress=progress,
        )
    elif args.extraction_track == "attacked_4x":
        paired_artifacts = _load_agentdojo_paired_artifacts_for_attack_wrapping(args)
        if paired_artifacts:
            checkpointed_payload = _run_checkpointed_attack_wrapper_expansion(args, paired_artifacts, progress)
            if checkpointed_payload is not None:
                _write_payload(args.output, checkpointed_payload)
                return
            source_tasks, drafts, runtime_catalog = expand_paired_artifacts_to_attack_wrapped_variants(
                paired_artifacts["source_tasks"],
                paired_artifacts["task_drafts"],
                paired_artifacts["runtime_catalog"],
                attack_variants=copy.deepcopy(DEFAULT_AGENTDOJO_ATTACK_VARIANTS),
                limit=args.limit,
                progress=progress,
            )
            attack_specs = []
            payload = {
                "benchmark": "agentdojo",
                "benchmark_version": args.benchmark_version,
                "extraction_track": args.extraction_track,
                "source_task_count": len(source_tasks),
                "source_tasks": source_tasks,
                "task_drafts": drafts,
                "runtime_catalog": runtime_catalog,
                "derived_from_paired_convert_dir": paired_artifacts["paired_convert_dir"],
                "expansion_mode": "paired_wrapper_only",
            }

            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                prefix = args.output_prefix or f"agentdojo_{args.benchmark_version.replace('.', '_')}"
                source_tasks_path = output_dir / f"{prefix}_source_tasks.json"
                drafts_path = output_dir / f"{prefix}_drafts.json"
                bundles_path = output_dir / f"{prefix}_runtime_catalog.json"
                export_json(
                    source_tasks_path,
                    {
                        "benchmark": "agentdojo",
                        "benchmark_version": args.benchmark_version,
                        "extraction_track": args.extraction_track,
                        "source_tasks": source_tasks,
                        "derived_from_paired_convert_dir": paired_artifacts["paired_convert_dir"],
                        "expansion_mode": "paired_wrapper_only",
                    },
                )
                export_json(
                    drafts_path,
                    {
                        "benchmark": "agentdojo",
                        "benchmark_version": args.benchmark_version,
                        "extraction_track": args.extraction_track,
                        "source_tasks": source_tasks,
                        "task_drafts": drafts,
                        "derived_from_paired_convert_dir": paired_artifacts["paired_convert_dir"],
                        "expansion_mode": "paired_wrapper_only",
                    },
                )
                export_json(
                    bundles_path,
                    {
                        "benchmark": "agentdojo",
                        "benchmark_version": args.benchmark_version,
                        "extraction_track": args.extraction_track,
                        "runtime_catalog": runtime_catalog,
                        "derived_from_paired_convert_dir": paired_artifacts["paired_convert_dir"],
                        "expansion_mode": "paired_wrapper_only",
                    },
                )
                payload["source_tasks_output"] = str(source_tasks_path)
                payload["drafts_output"] = str(drafts_path)
                payload["runtime_catalog_output"] = str(bundles_path)

            _write_payload(args.output, payload)
            return

        source_tasks = extract_agentdojo_attacked_source_tasks(
            repo_path=args.agentdojo_repo,
            benchmark_version=args.benchmark_version,
            suite_names=args.suites,
            limit=args.limit,
            tool_scope_mode=args.tool_scope_mode,
        )
        attack_specs = []
    else:
        source_tasks = extract_agentdojo_source_tasks(
            repo_path=args.agentdojo_repo,
            benchmark_version=args.benchmark_version,
            suite_names=args.suites,
            limit=args.limit,
            tool_scope_mode=args.tool_scope_mode,
        )
        attack_specs = []
    drafts = convert_source_tasks(
        source_tasks,
        backend=args.backend,
        config=llm_config,
        progress=progress,
        checkpoint_dir=_resolve_checkpoint_dir(args),
        resume=getattr(args, "resume", False),
    )
    runtime_catalog = assemble_runtime_catalog(drafts, progress=progress)

    payload = {
        "benchmark": "agentdojo",
        "benchmark_version": args.benchmark_version,
        "extraction_track": args.extraction_track,
        "source_task_count": len(source_tasks),
        "source_tasks": source_tasks,
        **({"attack_specs": attack_specs} if attack_specs else {}),
        "task_drafts": drafts,
        "runtime_catalog": runtime_catalog,
    }
    usage_summary = _summarize_llm_usage(drafts)
    if usage_summary:
        payload["llm_usage_summary"] = usage_summary

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.output_prefix or f"agentdojo_{args.benchmark_version.replace('.', '_')}"
        source_tasks_path = output_dir / f"{prefix}_source_tasks.json"
        drafts_path = output_dir / f"{prefix}_drafts.json"
        bundles_path = output_dir / f"{prefix}_runtime_catalog.json"
        export_json(
            source_tasks_path,
            {
                "benchmark": "agentdojo",
                "benchmark_version": args.benchmark_version,
                "extraction_track": args.extraction_track,
                "source_tasks": source_tasks,
                **({"attack_specs": attack_specs} if attack_specs else {}),
            },
        )
        export_json(
            drafts_path,
            {
                "benchmark": "agentdojo",
                "benchmark_version": args.benchmark_version,
                "extraction_track": args.extraction_track,
                "source_tasks": source_tasks,
                **({"attack_specs": attack_specs} if attack_specs else {}),
                "task_drafts": drafts,
                **({"llm_usage_summary": usage_summary} if usage_summary else {}),
            },
        )
        export_json(
            bundles_path,
            {
                "benchmark": "agentdojo",
                "benchmark_version": args.benchmark_version,
                "extraction_track": args.extraction_track,
                "runtime_catalog": runtime_catalog,
                **({"llm_usage_summary": usage_summary} if usage_summary else {}),
            },
        )
        payload["source_tasks_output"] = str(source_tasks_path)
        payload["drafts_output"] = str(drafts_path)
        payload["runtime_catalog_output"] = str(bundles_path)

    _write_payload(args.output, payload)


def _load_catalog_payload_for_agent(args):
    if args.catalog_file:
        return _read_json(args.catalog_file)
    if args.runtime_catalog_file:
        return _read_json(args.runtime_catalog_file)
    if args.task_draft_file:
        return _read_json(args.task_draft_file)
    return None


def cmd_run_agent_episode(args):
    llm_config = _build_llm_config_from_args(args)
    if not llm_config.base_url or not llm_config.model:
        raise ValueError("run-agent-episode requires --llm-base-url and --llm-model (or configured defaults).")

    client = _build_openai_client(llm_config)
    system_prompt, protocol_profile = _resolve_rollout_prompt_and_profile(
        args.system_prompt,
        args.protocol_profile,
        bool(args.benchmark_faithful_agent),
    )

    rollout_config = AgentRolloutConfig(
        max_turns=args.max_turns,
        temperature=llm_config.temperature,
        max_tokens=args.agent_max_tokens,
        system_prompt=system_prompt,
        stop_on_episode_success=bool(args.stop_on_success),
        protocol_profile=protocol_profile,
        max_empty_terminal_retries=0 if args.benchmark_faithful_agent else AgentRolloutConfig().max_empty_terminal_retries,
    )
    payload = run_agent_episode(
        server_url=args.server_url,
        llm_client=client,
        task_id=args.task_id,
        catalog_payload=_load_catalog_payload_for_agent(args),
        rollout_config=rollout_config,
        scenario=args.scenario,
    )
    _write_payload(args.output, payload)


def cmd_run_agent_batch(args):
    llm_config = _build_llm_config_from_args(args)
    if not llm_config.base_url or not llm_config.model:
        raise ValueError("run-agent-batch requires --llm-base-url and --llm-model (or configured defaults).")

    system_prompt, protocol_profile = _resolve_rollout_prompt_and_profile(
        args.system_prompt,
        args.protocol_profile,
        bool(args.benchmark_faithful_agent),
    )

    rollout_config = AgentRolloutConfig(
        max_turns=args.max_turns,
        temperature=llm_config.temperature,
        max_tokens=args.agent_max_tokens,
        system_prompt=system_prompt,
        stop_on_episode_success=bool(args.stop_on_success),
        protocol_profile=protocol_profile,
        max_empty_terminal_retries=0 if args.benchmark_faithful_agent else AgentRolloutConfig().max_empty_terminal_retries,
    )
    batch_config = BatchRolloutConfig(max_workers=args.max_workers)

    def _client_factory():
        return _build_openai_client(llm_config)

    task_ids = None
    if args.task_ids:
        task_ids = [item.strip() for item in args.task_ids.split(",") if item.strip()]

    payload = run_agent_batch(
        server_url=args.server_url,
        llm_client_factory=_client_factory,
        task_ids=task_ids,
        catalog_payload=_load_catalog_payload_for_agent(args),
        rollout_config=rollout_config,
        batch_config=batch_config,
        output_dir=args.output_dir,
        benchmark=args.benchmark,
    )
    _write_payload(args.output, payload)


def build_parser():
    parser = argparse.ArgumentParser(description="CLI for bundle-driven agentic RL task generation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump_tool_pool = subparsers.add_parser("dump-tool-pool", help="Export the current tool pool.")
    dump_tool_pool.add_argument("--tool-pool", default=None)
    dump_tool_pool.add_argument("--domain", default=None)
    dump_tool_pool.add_argument("--output", default=None)
    dump_tool_pool.set_defaults(func=cmd_dump_tool_pool)

    generate_drafts = subparsers.add_parser("generate-drafts", help="Generate task drafts from the tool pool.")
    _add_common_generation_args(generate_drafts)
    generate_drafts.add_argument("--output", default=None)
    generate_drafts.set_defaults(func=cmd_generate_drafts)

    assemble_bundles = subparsers.add_parser("assemble-bundles", help="Assemble runtime bundles from task drafts.")
    _add_common_generation_args(assemble_bundles)
    assemble_bundles.add_argument("--drafts-file", default=None, help="Path to a JSON file containing task_drafts.")
    assemble_bundles.add_argument("--output", default=None)
    assemble_bundles.set_defaults(func=cmd_assemble_bundles)

    generate_examples = subparsers.add_parser(
        "generate-examples",
        help="Generate both task draft examples and runtime bundle examples in one command.",
    )
    _add_common_generation_args(generate_examples)
    generate_examples.add_argument("--output-dir", default="examples")
    generate_examples.set_defaults(func=cmd_generate_examples)

    convert_source_tasks_parser = subparsers.add_parser(
        "convert-source-tasks",
        help="Convert existing source tasks and tools into planner-first drafts and runtime bundles.",
    )
    _add_common_generation_args(convert_source_tasks_parser)
    convert_source_tasks_parser.add_argument("--source-tasks-file", required=True, help="Path to a JSON file containing source_tasks.")
    convert_source_tasks_parser.add_argument("--output", default=None, help="Optional path to write the full combined payload.")
    convert_source_tasks_parser.add_argument("--output-dir", default=None, help="Optional directory to separately write drafts and runtime catalog JSON files.")
    convert_source_tasks_parser.add_argument("--output-prefix", default="converted_source_tasks")
    convert_source_tasks_parser.add_argument("--checkpoint-dir", default=None, help="Optional checkpoint directory for per-task draft persistence.")
    convert_source_tasks_parser.add_argument("--resume", action="store_true", help="Resume from an existing checkpoint directory.")
    convert_source_tasks_parser.set_defaults(func=cmd_convert_source_tasks)

    extract_benchmark_source_tasks_parser = subparsers.add_parser(
        "extract-benchmark-source-tasks",
        help="Extract benchmark tasks into normalized source-task JSON payloads without conversion.",
    )
    extract_benchmark_source_tasks_parser.add_argument(
        "--benchmark",
        required=True,
        choices=["agent-safety-bench", "agentsafetybench", "agentdyn", "asb", "mtagentrisk", "toolshield"],
    )
    extract_benchmark_source_tasks_parser.add_argument("--repo-path", required=True, help="Path to the local benchmark checkout or dataset root.")
    extract_benchmark_source_tasks_parser.add_argument("--benchmark-version", default=None)
    extract_benchmark_source_tasks_parser.add_argument("--extraction-track", default="all")
    extract_benchmark_source_tasks_parser.add_argument("--limit", type=int, default=None)
    extract_benchmark_source_tasks_parser.add_argument("--attack-method", default="dpi")
    extract_benchmark_source_tasks_parser.add_argument("--output", default=None)
    extract_benchmark_source_tasks_parser.add_argument("--output-dir", default=None)
    extract_benchmark_source_tasks_parser.add_argument("--output-prefix", default=None)
    extract_benchmark_source_tasks_parser.set_defaults(func=cmd_extract_benchmark_source_tasks)

    convert_agentdojo_parser = subparsers.add_parser(
        "convert-agentdojo-benchmark",
        help="Extract AgentDojo benchmark cases and convert them into runtime bundles.",
    )
    _add_common_generation_args(convert_agentdojo_parser)
    convert_agentdojo_parser.add_argument("--agentdojo-repo", required=True, help="Path to a local AgentDojo repository checkout.")
    convert_agentdojo_parser.add_argument(
        "--benchmark-version",
        default=DEFAULT_AGENTDOJO_BENCHMARK_VERSION,
        help="AgentDojo benchmark version to extract. v1 yields the paper-aligned 629-case benchmark.",
    )
    convert_agentdojo_parser.add_argument(
        "--tool-scope-mode",
        choices=["source_native_preferred", "heuristic_only"],
        default="source_native_preferred",
        help=(
            "How AgentDojo extractor derives tool visibility scope. "
            "source_native_preferred preserves native return schemas when available and falls back to heuristics; "
            "heuristic_only forces the legacy heuristic scope builder."
        ),
    )
    convert_agentdojo_parser.add_argument(
        "--extraction-track",
        choices=["paired_attacks", "utility_clean", "attacked_4x", "utility_attack_paired"],
        default="paired_attacks",
        help=(
            "How to materialize AgentDojo source tasks: paired_attacks keeps one source per "
            "(user_task, injection_task) pair; utility_clean exports 97 benign + 35 injection-as-utility tasks "
            "for v1.2-style utility accounting; attacked_4x expands each attacked pair across 4 baseline attack variants; "
            "utility_attack_paired exports clean utility tasks plus independent attack specs, then materializes paired clean/attacked scenarios."
        ),
    )
    convert_agentdojo_parser.add_argument(
        "--suites",
        nargs="*",
        default=None,
        help="Optional subset of AgentDojo suites to extract (workspace travel banking slack).",
    )
    convert_agentdojo_parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of extracted cases.")
    convert_agentdojo_parser.add_argument("--output", default=None, help="Optional path to write the full combined payload.")
    convert_agentdojo_parser.add_argument("--output-dir", required=True, help="Directory for source tasks, drafts, and runtime catalog outputs.")
    convert_agentdojo_parser.add_argument("--output-prefix", default="agentdojo")
    convert_agentdojo_parser.add_argument("--checkpoint-dir", default=None, help="Optional checkpoint directory for per-task draft persistence.")
    convert_agentdojo_parser.add_argument(
        "--paired-convert-dir",
        default=None,
        help="Optional paired949 convert directory to expand into attacked_4x without LLM.",
    )
    convert_agentdojo_parser.add_argument(
        "--paired-output-prefix",
        default=None,
        help="Optional output prefix used by the paired949 convert artifacts.",
    )
    convert_agentdojo_parser.add_argument("--resume", action="store_true", help="Resume from an existing checkpoint directory.")
    convert_agentdojo_parser.set_defaults(func=cmd_convert_agentdojo_benchmark)

    run_agent_episode_parser = subparsers.add_parser(
        "run-agent-episode",
        help="Register bundle payloads if needed, start an episode, and let an LLM agent interact with the server.",
    )
    run_agent_episode_parser.add_argument("--server-url", required=True, help="Base URL of the running tasksvc server.")
    run_agent_episode_parser.add_argument("--task-id", default=None, help="Existing task_id already registered on the server.")
    run_agent_episode_parser.add_argument("--scenario", default="clean", choices=["clean", "attacked"])
    run_agent_episode_parser.add_argument("--catalog-file", default=None, help="Path to a bundle/task payload JSON to register before starting.")
    run_agent_episode_parser.add_argument("--runtime-catalog-file", default=None, help="Alias for --catalog-file when the payload is a runtime catalog.")
    run_agent_episode_parser.add_argument("--task-draft-file", default=None, help="Path to a task draft or task_drafts JSON to register before starting.")
    run_agent_episode_parser.add_argument("--llm-base-url", default=None)
    run_agent_episode_parser.add_argument("--llm-model", default=None)
    run_agent_episode_parser.add_argument("--llm-api-key", default=None)
    run_agent_episode_parser.add_argument("--llm-user-agent", default=None)
    run_agent_episode_parser.add_argument("--llm-timeout", type=int, default=None)
    run_agent_episode_parser.add_argument("--llm-max-retries", type=int, default=None)
    run_agent_episode_parser.add_argument("--llm-temperature", type=float, default=0.0)
    run_agent_episode_parser.add_argument("--agent-max-tokens", type=int, default=1024)
    run_agent_episode_parser.add_argument("--max-turns", type=int, default=15)
    run_agent_episode_parser.add_argument(
        "--system-prompt",
        default=AgentRolloutConfig().system_prompt,
        help="System prompt used for runtime agent reasoning.",
    )
    run_agent_episode_parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="Finish the episode right after a tool step that already satisfies success_rule.",
    )
    run_agent_episode_parser.add_argument(
        "--benchmark-faithful-agent",
        action="store_true",
        help="Use the benchmark-faithful neutral system prompt aligned with the original AgentDojo rollout.",
    )
    run_agent_episode_parser.add_argument(
        "--protocol-profile",
        choices=["structured_json", "benchmark_faithful"],
        default="structured_json",
        help="How tool results are surfaced back to the model during rollout.",
    )
    run_agent_episode_parser.add_argument("--output", default=None)
    run_agent_episode_parser.set_defaults(func=cmd_run_agent_episode)

    run_agent_batch_parser = subparsers.add_parser(
        "run-agent-batch",
        help="Register a catalog if needed and run agent rollouts over many task_ids with concurrent workers.",
    )
    run_agent_batch_parser.add_argument("--server-url", required=True, help="Base URL of the running tasksvc server.")
    run_agent_batch_parser.add_argument("--task-ids", default=None, help="Comma-separated task_ids to run. Defaults to all registered tasks.")
    run_agent_batch_parser.add_argument("--catalog-file", default=None, help="Path to a bundle/task payload JSON to register before running.")
    run_agent_batch_parser.add_argument("--runtime-catalog-file", default=None, help="Alias for --catalog-file when the payload is a runtime catalog.")
    run_agent_batch_parser.add_argument("--task-draft-file", default=None, help="Path to a task draft or task_drafts JSON to register before running.")
    run_agent_batch_parser.add_argument("--llm-base-url", default=None)
    run_agent_batch_parser.add_argument("--llm-model", default=None)
    run_agent_batch_parser.add_argument("--llm-api-key", default=None)
    run_agent_batch_parser.add_argument("--llm-user-agent", default=None)
    run_agent_batch_parser.add_argument("--llm-timeout", type=int, default=None)
    run_agent_batch_parser.add_argument("--llm-max-retries", type=int, default=None)
    run_agent_batch_parser.add_argument("--llm-temperature", type=float, default=0.0)
    run_agent_batch_parser.add_argument("--agent-max-tokens", type=int, default=1024)
    run_agent_batch_parser.add_argument("--max-turns", type=int, default=15)
    run_agent_batch_parser.add_argument("--max-workers", type=int, default=8)
    run_agent_batch_parser.add_argument("--benchmark", default=None, help="Optional benchmark label to include in batch outputs.")
    run_agent_batch_parser.add_argument(
        "--system-prompt",
        default=AgentRolloutConfig().system_prompt,
        help="System prompt used for runtime agent reasoning.",
    )
    run_agent_batch_parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="Finish each episode right after a tool step that already satisfies success_rule.",
    )
    run_agent_batch_parser.add_argument(
        "--benchmark-faithful-agent",
        action="store_true",
        help="Use the benchmark-faithful neutral system prompt aligned with the original AgentDojo rollout.",
    )
    run_agent_batch_parser.add_argument(
        "--protocol-profile",
        choices=["structured_json", "benchmark_faithful"],
        default="structured_json",
        help="How tool results are surfaced back to the model during rollout.",
    )
    run_agent_batch_parser.add_argument("--output-dir", default=None, help="Directory for manifest/summary/per-task result files.")
    run_agent_batch_parser.add_argument("--output", default=None)
    run_agent_batch_parser.set_defaults(func=cmd_run_agent_batch)

    remote_worker_preflight = subparsers.add_parser(
        "remote-worker-preflight",
        help="Check PJLab dev-host/worker SSH reachability and optionally run rlaunch if the worker is missing.",
    )
    remote_worker_preflight.add_argument("--host", default=DEFAULT_PJLAB_HOST)
    remote_worker_preflight.add_argument("--port", type=int, default=22)
    remote_worker_preflight.add_argument("--user", default=DEFAULT_PJLAB_USER)
    remote_worker_preflight.add_argument("--identity-file", default=DEFAULT_PJLAB_IDENTITY_FILE)
    remote_worker_preflight.add_argument("--connect-timeout", type=int, default=10)
    remote_worker_preflight.add_argument("--ssh-timeout", type=int, default=20)
    remote_worker_preflight.add_argument("--use-ssh-config", action="store_true")
    remote_worker_preflight.add_argument("--dev-host-command", default="echo ok")
    remote_worker_preflight.add_argument("--worker-name", default=None)
    remote_worker_preflight.add_argument("--worker-owner", default="chenguanxu")
    remote_worker_preflight.add_argument("--worker-pod-user", default="root")
    remote_worker_preflight.add_argument("--worker-pod-group", default="ailab-ai4good1")
    remote_worker_preflight.add_argument("--worker-command", default="hostname && pwd")
    remote_worker_preflight.add_argument("--launch-if-missing", action="store_true")
    remote_worker_preflight.add_argument("--launch-command", default=None)
    remote_worker_preflight.set_defaults(func=cmd_remote_worker_preflight)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

