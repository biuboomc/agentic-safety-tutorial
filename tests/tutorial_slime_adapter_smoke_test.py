import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasksvc.tutorial.slime_adapter import (
    export_batch_results_to_slime_jsonl,
    rollout_result_to_slime_record,
)


def _sample_scenario(task_id, scenario, task_success, risk_enabled=False, risk_success=False):
    return {
        "ok": True,
        "task_id": task_id,
        "scenario": scenario,
        "episode_id": f"ep_{scenario}",
        "messages": [
            {"role": "system", "content": [{"type": "text", "content": "system"}]},
            {"role": "user", "content": [{"type": "text", "content": f"user query {scenario}"}]},
            {"role": "assistant", "content": [{"type": "text", "content": "done"}], "tool_calls": []},
        ],
        "transcript": [
            {
                "turn_index": 1,
                "assistant_text": "",
                "tool_calls": [
                    {
                        "tool_name": "read_file",
                        "arguments": {"file_path": "bill.txt"},
                        "server_result": {
                            "tool_result": {"content": "bill"},
                            "observation": "read ok",
                            "reward_info": {"step_reward": 0.1, "success": False},
                        },
                    }
                ],
                "recorded_tool_calls": [],
            }
        ],
        "final_answer": "finished",
        "episode": {
            "episode_id": f"ep_{scenario}",
            "scenario": scenario,
            "user_query": f"user query {scenario}",
            "history": [{"tool_name": "read_file"}],
            "turn_index": 1,
            "finished": True,
            "finish_reason": "assistant_no_tool_calls",
            "risk": {"enabled": risk_enabled, "success": risk_success, "checklist_progress": {}},
        },
        "task_success": task_success,
        "risk_enabled": risk_enabled,
        "risk_success": risk_success,
    }


def main():
    payload = {
        "task_id": "tutorial_banking_bill_pay",
        "pair_id": "tutorial_banking_bill_pay",
        "status": "success",
        "clean_task_success": True,
        "attacked_task_success": True,
        "risk_enabled": True,
        "risk_success": False,
        "scenario_results": {
            "clean": _sample_scenario("tutorial_banking_bill_pay", "clean", True, False, False),
            "attacked": _sample_scenario("tutorial_banking_bill_pay", "attacked", True, True, False),
        },
        "llm_usage_summary": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    record = rollout_result_to_slime_record(payload, split="train", source="tutorial_curriculum")
    assert record["record_version"] == "tasksvc_tutorial_slime_v1"
    assert record["task_id"] == "tutorial_banking_bill_pay"
    assert record["metrics"]["utility"] == 1
    assert record["metrics"]["utility_under_attack"] == 1
    assert record["metrics"]["ASR"] == 0
    assert record["clean"]["tool_calls"][0]["tool_name"] == "read_file"
    assert record["attacked"]["episode_summary"]["scenario"] == "attacked"

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        results_dir = base / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "tutorial_banking_bill_pay.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_path = base / "slime.jsonl"
        export_batch_results_to_slime_jsonl(base, output_path, split="train", source="tutorial_curriculum")
        lines = output_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        exported = json.loads(lines[0])
        assert exported["task_id"] == "tutorial_banking_bill_pay"
        assert exported["metadata"]["risk_enabled"] is True

    print("TUTORIAL_SLIME_ADAPTER_SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
