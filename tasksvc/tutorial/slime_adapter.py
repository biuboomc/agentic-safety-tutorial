import json
from pathlib import Path


TUTORIAL_SLIME_RECORD_VERSION = "tasksvc_tutorial_slime_v1"


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _message_text(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, dict):
            if "content" in item:
                parts.append(str(item.get("content") or ""))
            elif "text" in item:
                parts.append(str(item.get("text") or ""))
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(part for part in parts if part)


def _flatten_tool_calls(transcript):
    flattened = []
    for turn in transcript or []:
        for item in turn.get("tool_calls") or []:
            server_result = item.get("server_result") or {}
            flattened.append(
                {
                    "tool_name": item.get("tool_name"),
                    "arguments": item.get("arguments") or {},
                    "tool_result": server_result.get("tool_result"),
                    "observation": server_result.get("observation"),
                    "reward_info": server_result.get("reward_info") or {},
                    "invalid_call": bool(server_result.get("invalid_call")),
                    "error": server_result.get("error"),
                }
            )
    return flattened


def _reward_trace(transcript):
    trace = []
    for tool_call in _flatten_tool_calls(transcript):
        reward_info = tool_call.get("reward_info") or {}
        if reward_info:
            trace.append(reward_info)
    return trace


def _episode_summary(result):
    episode = result.get("episode") or {}
    risk = episode.get("risk") or {}
    return {
        "episode_id": result.get("episode_id") or episode.get("episode_id"),
        "scenario": result.get("scenario") or episode.get("scenario") or "clean",
        "finished": bool(episode.get("finished")),
        "finish_reason": episode.get("finish_reason"),
        "turn_index": episode.get("turn_index"),
        "history_length": len(episode.get("history") or []),
        "task_success": bool(result.get("task_success")),
        "risk_enabled": bool(result.get("risk_enabled")),
        "risk_success": bool(result.get("risk_success")),
        "final_answer": result.get("final_answer"),
        "risk_checklist_progress": risk.get("checklist_progress") or {},
    }


def _scenario_record(result):
    if not isinstance(result, dict) or not result:
        return None
    messages = list(result.get("messages") or [])
    transcript = list(result.get("transcript") or [])
    episode = result.get("episode") or {}
    return {
        "scenario": result.get("scenario") or episode.get("scenario") or "clean",
        "user_query": episode.get("user_query") or (_message_text(messages[1]) if len(messages) > 1 else ""),
        "messages": messages,
        "transcript": transcript,
        "tool_calls": _flatten_tool_calls(transcript),
        "reward_trace": _reward_trace(transcript),
        "task_success": bool(result.get("task_success")),
        "risk_enabled": bool(result.get("risk_enabled")),
        "risk_success": bool(result.get("risk_success")),
        "final_answer": result.get("final_answer"),
        "episode_summary": _episode_summary(result),
    }


def rollout_result_to_slime_record(payload, *, split="train", source="tasksvc_tutorial"):
    if not isinstance(payload, dict):
        raise ValueError("rollout_result_to_slime_record expects a dict payload.")

    paired = payload.get("scenario_results") or {}
    clean_result = paired.get("clean") if isinstance(paired, dict) else None
    attacked_result = paired.get("attacked") if isinstance(paired, dict) else None
    if clean_result is None:
        clean_result = payload

    task_id = str(payload.get("task_id") or clean_result.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("Rollout payload is missing task_id.")

    clean_success = bool(payload.get("clean_task_success", clean_result.get("task_success")))
    attacked_success = bool(payload.get("attacked_task_success", attacked_result.get("task_success") if attacked_result else False))
    risk_enabled = bool(payload.get("risk_enabled", attacked_result.get("risk_enabled") if attacked_result else False))
    risk_success = bool(payload.get("risk_success", attacked_result.get("risk_success") if attacked_result else False))

    return {
        "record_version": TUTORIAL_SLIME_RECORD_VERSION,
        "task_id": task_id,
        "pair_id": str(payload.get("pair_id") or task_id),
        "split": str(split or "train"),
        "source": str(source or "tasksvc_tutorial"),
        "status": str(payload.get("status") or ("success" if clean_success else "failure")),
        "metrics": {
            "utility": 1 if clean_success else 0,
            "utility_under_attack": 1 if attacked_success else 0 if attacked_result is not None else None,
            "ASR": 1 if risk_success else 0 if risk_enabled or attacked_result is not None else None,
        },
        "clean": _scenario_record(clean_result),
        "attacked": _scenario_record(attacked_result),
        "llm_usage_summary": payload.get("llm_usage_summary") or {},
        "metadata": {
            "risk_enabled": risk_enabled,
            "risk_success": risk_success,
            "clean_task_success": clean_success,
            "attacked_task_success": attacked_success,
        },
    }


def iter_batch_result_payloads(batch_output_dir):
    results_dir = Path(batch_output_dir) / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Batch results directory not found: {results_dir}")
    for path in sorted(results_dir.glob("*.json")):
        yield _read_json(path)


def write_slime_records_jsonl(records, output_path):
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


def export_batch_results_to_slime_jsonl(batch_output_dir, output_path, *, split="train", source="tasksvc_tutorial", split_by_task_id=None):
    records = []
    split_by_task_id = split_by_task_id or {}
    for payload in iter_batch_result_payloads(batch_output_dir):
        task_id = str(payload.get("task_id") or "").strip()
        record_split = split_by_task_id.get(task_id, split)
        records.append(
            rollout_result_to_slime_record(
                payload,
                split=record_split,
                source=source,
            )
        )
    return write_slime_records_jsonl(records, output_path)
