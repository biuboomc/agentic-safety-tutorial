import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasksvc.assembly.catalog_loader import export_json
from tasksvc.assembly.env_assembler import assemble_runtime_catalog
from tasksvc.generation.llm_generator import LLMGenerationConfig
from tasksvc.generation.source_task_converter import convert_source_tasks, load_source_tasks


def _default_source_tasks():
    return ROOT / "examples" / "tutorial" / "tutorial_source_tasks.json"


def _default_output_dir():
    return ROOT / "examples" / "tutorial" / "generated"


def _build_llm_config(args):
    config = LLMGenerationConfig()
    if args.llm_base_url:
        config.base_url = args.llm_base_url
    if args.llm_model:
        config.model = args.llm_model
    if args.llm_api_key is not None:
        config.api_key = args.llm_api_key
    if args.llm_user_agent:
        config.user_agent = args.llm_user_agent
    if args.llm_timeout is not None:
        config.timeout = args.llm_timeout
    if args.llm_max_retries is not None:
        config.max_retries = args.llm_max_retries
    if args.llm_temperature is not None:
        config.temperature = args.llm_temperature
    return config


def main():
    parser = argparse.ArgumentParser(description="Build the tutorial curriculum into drafts and a runtime catalog.")
    parser.add_argument("--source-tasks-file", default=str(_default_source_tasks()))
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    parser.add_argument("--backend", choices=["placeholder", "llm"], default="placeholder")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-user-agent", default=None)
    parser.add_argument("--llm-timeout", type=int, default=None)
    parser.add_argument("--llm-max-retries", type=int, default=None)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    args = parser.parse_args()

    source_tasks_path = Path(args.source_tasks_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_tasks = load_source_tasks(source_tasks_path)
    llm_config = _build_llm_config(args) if args.backend == "llm" else None
    drafts = convert_source_tasks(source_tasks, backend=args.backend, config=llm_config)
    runtime_catalog = assemble_runtime_catalog(drafts)

    source_payload = {
        "tutorial": "agentic_safety",
        "source_tasks_file": str(source_tasks_path),
        "source_tasks": source_tasks,
    }
    drafts_payload = {
        "tutorial": "agentic_safety",
        "source_tasks_file": str(source_tasks_path),
        "task_drafts": drafts,
    }
    catalog_payload = {
        "tutorial": "agentic_safety",
        "source_tasks_file": str(source_tasks_path),
        "runtime_catalog": runtime_catalog,
    }
    build_manifest = {
        "tutorial": "agentic_safety",
        "backend": args.backend,
        "source_tasks_file": str(source_tasks_path),
        "task_count": len(source_tasks),
        "task_ids": [draft["planned_task"]["task_id"] for draft in drafts],
        "outputs": {
            "source_tasks": str(output_dir / "tutorial_source_tasks.json"),
            "task_drafts": str(output_dir / "tutorial_task_drafts.json"),
            "runtime_catalog": str(output_dir / "tutorial_runtime_catalog.json"),
        },
    }

    export_json(output_dir / "tutorial_source_tasks.json", source_payload)
    export_json(output_dir / "tutorial_task_drafts.json", drafts_payload)
    export_json(output_dir / "tutorial_runtime_catalog.json", catalog_payload)
    export_json(output_dir / "tutorial_build_manifest.json", build_manifest)

    print(f"Wrote tutorial curriculum artifacts to {output_dir}")
    print(f"Task IDs: {', '.join(build_manifest['task_ids'])}")


if __name__ == "__main__":
    main()
