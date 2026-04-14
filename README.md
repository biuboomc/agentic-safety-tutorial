# Agentic Safety Tutorial

This repository is a tutorial companion for understanding, building, evaluating, and training **tool-calling LLM agents** with a deliberate focus on **agentic safety**.

The central claim of this tutorial is simple:

- the interesting unit is no longer just the model
- it is the **model + harness + tools + environment + evaluator**
- once that system can act, safety becomes a question about **behavior and outcomes**, not only text

That is why this repository does not start from benchmark conversion or infrastructure APIs.  
It starts from the conceptual arc of modern agentic systems, then gradually introduces `tasksvc`, benchmark slices, and `slime` as training-and-evaluation tools.

## Why This Tutorial Exists

The field has moved fast:

- prompt engineering and chain-of-thought made models more useful
- **ReAct** showed that reasoning and acting can be interleaved
- **Toolformer** made tool selection itself part of the modeling problem
- **Voyager** pushed memory, skills, and open-ended interaction forward
- production teams then discovered that the hard part is often not "calling a model", but designing the **harness**, the **environment**, the **context pipeline**, and the **evaluation loop**

In other words, the center of gravity has shifted:

- from prompt engineering
- to **context engineering**
- to **agent engineering**
- and, increasingly, to **harness engineering** 

This tutorial is about that shift, with agentic safety as the main lens.

## Working Definition

Throughout this repo, we use a pragmatic definition:

- an **agent** is an LLM-based system that can **use tools in a loop** to pursue a user goal
- a **workflow** is a more scripted procedure whose control flow is mostly fixed by the developer
- a **harness** is the orchestration layer that packages context, exposes tools, runs the loop, records the trace, and determines when the interaction is done

This is deliberately close to how recent engineering writing from Anthropic and OpenAI talks about real deployed agents.

## What You Will Learn

By the end, you should be able to:

- explain the engineering evolution from prompt-centric systems to agentic systems
- define terms like **agent loop**, **harness**, **sandbox**, **trajectory**, **outcome**, and **evaluation harness**
- distinguish ordinary LLM safety from **agentic safety**
- reason about prompt injection, tool misuse, state corruption, and verification loops
- represent a safety task as an executable environment with explicit benign and risk goals
- connect environment rollouts to a training stack and run a small safety-oriented training pipeline

## Core Concepts You Will See Repeatedly

### Agent loop
The repeated cycle of model inference, tool selection, tool execution, observation, and continuation until the system finishes.

### Harness
The orchestration layer around the model. It prepares context, routes tool calls, records the trace, and determines how the system advances.

### Sandbox
The execution boundary where actions actually run. A sandbox constrains what the agent can do and isolates side effects.

### Transcript / trajectory
The full record of a run: prompts, tool calls, observations, intermediate steps, and messages.

### Outcome
The final state of the environment after the run. This is often more important than the final text output.

### Evaluation harness
The infrastructure that runs many tasks, records outputs and state, grades results, and aggregates metrics.

### Context engineering
The practice of deciding what information is available to the model at each step, and in what form.

## Suggested Reading Paths

### Path A: Concept-first
Best for readers new to agents.

1. [Tutorial Index](docs/tutorial/README.md)
2. [Overview](docs/tutorial/00_overview.md)
3. [Why Agents](docs/tutorial/01_why_agents.md)
4. [Tool-Calling Agents](docs/tutorial/02_tool_calling_agents.md)
5. [Agent Fragility](docs/tutorial/03_agent_fragility.md)
6. [Attack Surfaces](docs/tutorial/04_attack_surfaces.md)
7. [Safety Requirements](docs/tutorial/05_safety_requirements.md)
8. [Evaluating Safe Agents](docs/tutorial/06_evaluating_safe_agents.md)

### Path B: Training-system-first
Best for readers who already know agent basics and want the environment/training path.

1. [tasksvc as an Environment](docs/tutorial/07_tasksvc_as_environment.md)
2. [Tutorial Curriculum](docs/tutorial/08_tutorial_curriculum.md)
3. [Slime Training Pipeline](docs/tutorial/09_slime_training_pipeline.md)
4. [AgentDojo Evaluation Showcase](docs/tutorial/10_agentdojo_evaluation_showcase.md)
5. [Research Map](docs/tutorial/11_research_map.md)

### Path C: Research-first
Best for readers who want the conceptual map before touching code.

1. [Research Map](docs/tutorial/11_research_map.md)
2. [Why Agents](docs/tutorial/01_why_agents.md)
3. [Tool-Calling Agents](docs/tutorial/02_tool_calling_agents.md)
4. [Evaluating Safe Agents](docs/tutorial/06_evaluating_safe_agents.md)

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

### 6. Launch the training wrapper

```bash
python scripts/tutorial_train_with_slime.py ^
  --batch-output-dir examples/tutorial/generated/rollout ^
  --manifest-file examples/tutorial/tutorial_curriculum_manifest.json ^
  --slime-command-template "slime train --dataset {dataset_jsonl} --output-dir {output_dir}"
```

The repository does not vendor `slime`; it provides the adapter layer that makes the training boundary explicit.

## Resource-Based Navigation

Different parts of the tutorial need very different resources.

### CPU-only / no external API
Good for reading and local environment inspection.

- [Tutorial Index](docs/tutorial/README.md)
- [Overview](docs/tutorial/00_overview.md)
- [Why Agents](docs/tutorial/01_why_agents.md)
- [Tool-Calling Agents](docs/tutorial/02_tool_calling_agents.md)
- [Agent Fragility](docs/tutorial/03_agent_fragility.md)
- [Attack Surfaces](docs/tutorial/04_attack_surfaces.md)
- [Safety Requirements](docs/tutorial/05_safety_requirements.md)
- [Evaluating Safe Agents](docs/tutorial/06_evaluating_safe_agents.md)
- [Research Map](docs/tutorial/11_research_map.md)

### CPU + local task runtime
Good for understanding environment packaging and running placeholder examples.

- [tasksvc as an Environment](docs/tutorial/07_tasksvc_as_environment.md)
- [Tutorial Curriculum](docs/tutorial/08_tutorial_curriculum.md)
- [Tutorial source tasks](examples/tutorial/tutorial_source_tasks.json)
- [Tutorial curriculum manifest](examples/tutorial/tutorial_curriculum_manifest.json)
- [Build curriculum script](scripts/tutorial_build_curriculum.py)
- [Start server script](scripts/tutorial_start_server.py)

### API model required
Good for live agent rollouts and batch data collection.

- [Single rollout script](scripts/tutorial_run_single_rollout.py)
- [Curriculum batch rollout script](scripts/tutorial_run_curriculum_batch.py)
- [AgentDojo evaluation showcase](docs/tutorial/10_agentdojo_evaluation_showcase.md)
- [Showcase eval script](scripts/tutorial_run_showcase_eval.py)

### Training resources required
Good for readers who want the full training path.

- [Slime Training Pipeline](docs/tutorial/09_slime_training_pipeline.md)
- [Slime export adapter](tasksvc/tutorial/slime_adapter.py)
- [Slime dataset export script](scripts/tutorial_export_slime_dataset.py)
- [Slime training launcher](scripts/tutorial_train_with_slime.py)

### Suggested resource profile

- **Concept chapters:** CPU only
- **Environment chapters:** CPU only, no external model required if you use `placeholder`
- **Rollout collection:** one model API endpoint or local inference server
- **Training chapter:** trainer resources, ideally GPUs, with the tutorial target kept within roughly 4 A100s / 24h

## Tutorial Assets

- [Tutorial index](docs/tutorial/README.md)
- [Research map](docs/tutorial/11_research_map.md)
- [Tutorial source tasks](examples/tutorial/tutorial_source_tasks.json)
- [Tutorial curriculum manifest](examples/tutorial/tutorial_curriculum_manifest.json)
- [AgentDojo showcase manifest](examples/tutorial/agentdojo_showcase_manifest.json)

## Repository Layout

- `docs/tutorial/`
  - concept chapters, training/evaluation chapters, and research map
- `examples/tutorial/`
  - hand-authored warmup and safety curriculum tasks
- `tasksvc/`
  - environment contracts, source-task conversion, runtime server, rollout, and evaluation logic
- `tasksvc/tutorial/`
  - tutorial-facing adapters, including the slime export layer
- `scripts/`
  - tutorial entrypoints for building curricula, running rollouts, exporting records, and launching training/evaluation steps
- `tests/`
  - smoke tests for rollout, conversion, and tutorial-specific assets

## Where `tasksvc` Fits

`tasksvc` is not the whole tutorial. It is the **training-and-evaluation infrastructure**.

Its role is to make the following explicit:

- what the agent sees
- what tools it can use
- what state changes
- how success is checked
- how attacked scenarios differ from clean scenarios
- how transcripts and outcomes become reward-bearing artifacts

That is why it appears late in the tutorial rather than at the beginning.

## Advanced / Reference Material

These remain useful, but they are no longer the first-stop story:

- [Progress and roadmap](docs/progress_and_roadmap.md)
- benchmark extraction and conversion CLIs in [tasksvc_cli.py](tasksvc_cli.py)
- source-task converter internals in [source_task_converter.py](tasksvc/generation/source_task_converter.py)
- runtime rollout and batch evaluation in [agent_rollout.py](tasksvc/runtime/agent_rollout.py) and [batch_rollout.py](tasksvc/runtime/batch_rollout.py)
