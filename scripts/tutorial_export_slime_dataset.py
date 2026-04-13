import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasksvc.tutorial.slime_adapter import export_batch_results_to_slime_jsonl


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _default_batch_dir():
    return ROOT / "examples" / "tutorial" / "generated" / "rollout"


def _default_output():
    return ROOT / "examples" / "tutorial" / "generated" / "slime_train.jsonl"


def _split_mapping(manifest):
    mapping = {}
    for split_key in ("warmup_task_ids", "train_task_ids", "eval_task_ids"):
        split_name = split_key.replace("_task_ids", "")
        for task_id in manifest.get(split_key) or []:
            mapping[str(task_id)] = split_name
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Export tutorial rollout results into slime-ready JSONL records.")
    parser.add_argument("--batch-output-dir", default=str(_default_batch_dir()))
    parser.add_argument("--output-path", default=str(_default_output()))
    parser.add_argument("--manifest-file", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--source", default="tutorial_curriculum")
    args = parser.parse_args()

    split_by_task_id = None
    if args.manifest_file:
        split_by_task_id = _split_mapping(_read_json(Path(args.manifest_file).resolve()))

    output_path = export_batch_results_to_slime_jsonl(
        Path(args.batch_output_dir).resolve(),
        Path(args.output_path).resolve(),
        split=args.split,
        source=args.source,
        split_by_task_id=split_by_task_id,
    )
    print(f"Wrote slime-ready JSONL to {output_path}")


if __name__ == "__main__":
    main()
