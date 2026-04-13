import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasksvc.generation.llm_client import OpenAICompatClient
from tasksvc.runtime.agent_rollout import AgentRolloutConfig
from tasksvc.runtime.batch_rollout import BatchRolloutConfig, run_agent_batch


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _default_catalog():
    return ROOT / "examples" / "tutorial" / "generated" / "tutorial_runtime_catalog.json"


def _default_output_dir():
    return ROOT / "examples" / "tutorial" / "generated" / "rollout"


def main():
    parser = argparse.ArgumentParser(description="Run tutorial curriculum rollouts and write a batch result directory.")
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--runtime-catalog-file", default=str(_default_catalog()))
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-user-agent", default=None)
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--llm-max-retries", type=int, default=2)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--agent-max-tokens", type=int, default=1024)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    args = parser.parse_args()

    def _client_factory():
        return OpenAICompatClient(
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
    )
    batch_config = BatchRolloutConfig(max_workers=args.max_workers)
    payload = run_agent_batch(
        server_url=args.server_url,
        llm_client_factory=_client_factory,
        catalog_payload=_read_json(Path(args.runtime_catalog_file).resolve()),
        rollout_config=rollout_config,
        batch_config=batch_config,
        output_dir=str(Path(args.output_dir).resolve()),
        benchmark="tutorial_curriculum",
    )
    print(json.dumps(payload.get("summary") or {}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
