# Agentic Safety Tutorial

This repository is a tutorial companion for building, evaluating, and training **tool-calling LLM agents** with a deliberate focus on **agentic safety**.

The main story is:

1. understand what makes an agent different from a chatbot
2. understand why tool use introduces new safety failures
3. turn those failures into explicit environments, rewards, and evaluation
4. connect the environment to a training stack and run a small safety-oriented capstone

This is **not** primarily a benchmark-conversion tutorial.  
The `tasksvc` environment, benchmark extraction utilities, and batch evaluation code are here to support the later capstone chapters, not to dominate the narrative from the first page.

## Who This Is For

- researchers who know LLM safety but are new to agentic safety
- readers who know about agents but want a concrete safety-oriented system to study
- students, job seekers, and project builders who want a hands-on agent tutorial with a clear engineering payoff

## Reading Order

Start with the tutorial index:

- [Tutorial Index](docs/tutorial/README.md)
- [Chapter 0: Overview](docs/tutorial/00_overview.md)
- [Chapter 1: Why Agents](docs/tutorial/01_why_agents.md)
- [Chapter 2: Tool-Calling Agents](docs/tutorial/02_tool_calling_agents.md)
- [Chapter 3: Agent Fragility](docs/tutorial/03_agent_fragility.md)
- [Chapter 4: Attack Surfaces](docs/tutorial/04_attack_surfaces.md)
- [Chapter 5: Safety Requirements](docs/tutorial/05_safety_requirements.md)
- [Chapter 6: Evaluation](docs/tutorial/06_evaluating_safe_agents.md)
- [Chapter 7: tasksvc as an Environment](docs/tutorial/07_tasksvc_as_environment.md)
- [Chapter 8: Tutorial Curriculum](docs/tutorial/08_tutorial_curriculum.md)
- [Chapter 9: Slime Capstone](docs/tutorial/09_slime_capstone.md)
- [Chapter 10: AgentDojo Showcase](docs/tutorial/10_agentdojo_showcase.md)

## Quick Start

### 1. Build the tutorial curriculum

```bash
python scripts/tutorial_build_curriculum.py --backend placeholder --output-dir examples/tutorial/generated
```

### 2. Start the local server

```bash
python scripts/tutorial_start_server.py --catalog-file examples/tutorial/generated/tutorial_runtime_catalog.json
```

### 3. Run a single rollout

```bash
python scripts/tutorial_run_single_rollout.py ^
  --server-url http://127.0.0.1:8000 ^
  --task-id tutorial_banking_bill_pay ^
  --scenario clean ^
  --llm-base-url http://your-openai-compatible-endpoint/v1 ^
  --llm-model your-model-name
```

### 4. Run the tutorial curriculum as a batch

```bash
python scripts/tutorial_run_curriculum_batch.py ^
  --server-url http://127.0.0.1:8000 ^
  --runtime-catalog-file examples/tutorial/generated/tutorial_runtime_catalog.json ^
  --llm-base-url http://your-openai-compatible-endpoint/v1 ^
  --llm-model your-model-name ^
  --output-dir examples/tutorial/generated/rollout
```

### 5. Export a slime-ready dataset

```bash
python scripts/tutorial_export_slime_dataset.py ^
  --batch-output-dir examples/tutorial/generated/rollout ^
  --manifest-file examples/tutorial/tutorial_curriculum_manifest.json ^
  --output-path examples/tutorial/generated/slime_train.jsonl
```

### 6. Launch the capstone training wrapper

```bash
python scripts/tutorial_train_with_slime.py ^
  --batch-output-dir examples/tutorial/generated/rollout ^
  --manifest-file examples/tutorial/tutorial_curriculum_manifest.json ^
  --slime-command-template "slime train --dataset {dataset_jsonl} --output-dir {output_dir}"
```

The repository does not vendor `slime`; the tutorial supplies the adapter and wrapper layer that turns tasksvc rollouts into a stable training artifact.

## Tutorial Assets

- [Tutorial assets README](examples/tutorial/README.md)
- [Tutorial source tasks](examples/tutorial/tutorial_source_tasks.json)
- [Tutorial curriculum manifest](examples/tutorial/tutorial_curriculum_manifest.json)
- [AgentDojo showcase manifest](examples/tutorial/agentdojo_showcase_manifest.json)

## Repository Layout

- `docs/tutorial/`
  - the main tutorial chapters
- `examples/tutorial/`
  - hand-authored warmup and safety curriculum tasks
- `tasksvc/`
  - environment contracts, source-task conversion, runtime server, rollout, evaluation
- `tasksvc/tutorial/`
  - tutorial-facing helpers such as the slime adapter
- `scripts/`
  - tutorial entrypoints for building the curriculum, running rollouts, exporting records, and launching capstone steps
- `tests/`
  - smoke tests for rollout, conversion, and tutorial-specific assets

## What `tasksvc` Does

`tasksvc` is the environment and evaluation layer behind the capstone:

- converts source tasks into `TaskDraft` artifacts
- assembles drafts into `RuntimeTaskBundle` objects
- serves bundles through a resident HTTP environment server
- executes simulated tools with timeout isolation
- computes benign and risk-track evaluation without runtime LLM scoring

The core runtime loop is:

1. start an episode from a task id and scenario
2. expose `user_query` and tool schemas to the agent
3. accept tool calls through the server
4. update state and reward traces
5. finish the episode and expose task success, risk success, and transcript data

## Advanced / Reference Material

These remain useful, but they are no longer the first-stop story:

- [Progress and roadmap](docs/progress_and_roadmap.md)
- benchmark extraction and conversion CLIs in [tasksvc_cli.py](tasksvc_cli.py)
- source-task converter internals in [source_task_converter.py](tasksvc/generation/source_task_converter.py)
- runtime rollout and batch evaluation in [agent_rollout.py](tasksvc/runtime/agent_rollout.py) and [batch_rollout.py](tasksvc/runtime/batch_rollout.py)
