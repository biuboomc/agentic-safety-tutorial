import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasksvc.tutorial.slime_adapter import export_batch_results_to_slime_jsonl


def _default_manifest():
    return ROOT / "examples" / "tutorial" / "tutorial_curriculum_manifest.json"


def _default_dataset():
    return ROOT / "examples" / "tutorial" / "generated" / "slime_train.jsonl"


def _default_output_dir():
    return ROOT / "examples" / "tutorial" / "generated" / "slime_runs" / "default"


def _read_manifest(path):
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _split_mapping(manifest):
    mapping = {}
    for split_key in ("warmup_task_ids", "train_task_ids", "eval_task_ids"):
        split_name = split_key.replace("_task_ids", "")
        for task_id in manifest.get(split_key) or []:
            mapping[str(task_id)] = split_name
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Thin wrapper for launching a tutorial run with slime.")
    parser.add_argument("--batch-output-dir", default=None, help="Optional rollout directory to export before training.")
    parser.add_argument("--dataset-jsonl", default=str(_default_dataset()))
    parser.add_argument("--manifest-file", default=str(_default_manifest()))
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    parser.add_argument(
        "--slime-command-template",
        default="slime train --dataset {dataset_jsonl} --output-dir {output_dir}",
        help=(
            "Command template for your local slime installation. "
            "The placeholders {dataset_jsonl} and {output_dir} will be expanded."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_output_dir:
        manifest = _read_manifest(Path(args.manifest_file).resolve())
        export_batch_results_to_slime_jsonl(
            Path(args.batch_output_dir).resolve(),
            dataset_path,
            split="train",
            source="tutorial_curriculum",
            split_by_task_id=_split_mapping(manifest),
        )

    command = args.slime_command_template.format(
        dataset_jsonl=str(dataset_path),
        output_dir=str(output_dir),
    )
    print(command)
    if args.dry_run:
        return
    raise SystemExit(subprocess.run(shlex.split(command), check=False).returncode)


if __name__ == "__main__":
    main()
