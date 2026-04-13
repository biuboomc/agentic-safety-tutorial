import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasksvc.generation.llm_client import OpenAICompatClient
from tasksvc.runtime.agent_rollout import AgentRolloutConfig, BENCHMARK_FAITHFUL_SYSTEM_PROMPT, run_agent_episode


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _default_catalog():
    return ROOT / "examples" / "tutorial" / "generated" / "tutorial_runtime_catalog.json"


def main():
    parser = argparse.ArgumentParser(description="Run one tutorial rollout against a live tasksvc server.")
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--scenario", choices=["clean", "attacked"], default="clean")
    parser.add_argument("--runtime-catalog-file", default=None)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-user-agent", default=None)
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--llm-max-retries", type=int, default=2)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--agent-max-tokens", type=int, default=1024)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--benchmark-faithful-agent", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    client = OpenAICompatClient(
        base_url=args.llm_base_url,
        model=args.llm_model,
        api_key=args.llm_api_key,
        timeout=args.llm_timeout,
        user_agent=args.llm_user_agent,
        max_retries=args.llm_max_retries,
    )
    rollout_config = AgentRolloutConfig(
        max_turns=args.max_turns,
        temperature=args.llm_temperature,
        max_tokens=args.agent_max_tokens,
        system_prompt=BENCHMARK_FAITHFUL_SYSTEM_PROMPT if args.benchmark_faithful_agent else AgentRolloutConfig().system_prompt,
        protocol_profile="benchmark_faithful" if args.benchmark_faithful_agent else "structured_json",
    )
    catalog_payload = None
    if args.runtime_catalog_file:
        catalog_payload = _read_json(Path(args.runtime_catalog_file).resolve())
    elif _default_catalog().exists():
        catalog_payload = _read_json(_default_catalog())

    payload = run_agent_episode(
        server_url=args.server_url,
        llm_client=client,
        task_id=args.task_id,
        catalog_payload=catalog_payload,
        rollout_config=rollout_config,
        scenario=args.scenario,
    )

    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        target = Path(args.output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
