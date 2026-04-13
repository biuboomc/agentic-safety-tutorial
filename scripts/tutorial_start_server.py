import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _default_catalog():
    return ROOT / "examples" / "tutorial" / "generated" / "tutorial_runtime_catalog.json"


def main():
    parser = argparse.ArgumentParser(description="Start tasksvc with tutorial-friendly defaults.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--catalog-file", default=str(_default_catalog()))
    parser.add_argument("--backend", choices=["placeholder", "llm"], default="placeholder")
    parser.add_argument("--episode-max-steps", type=int, default=15)
    parser.add_argument("--tool-exec-timeout", type=int, default=8)
    args = parser.parse_args()

    command = [
        sys.executable,
        str(ROOT / "server.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--backend",
        args.backend,
        "--episode-max-steps",
        str(args.episode_max_steps),
        "--tool-exec-timeout",
        str(args.tool_exec_timeout),
    ]
    if args.catalog_file:
        command.extend(["--catalog-file", str(Path(args.catalog_file).resolve())])
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
