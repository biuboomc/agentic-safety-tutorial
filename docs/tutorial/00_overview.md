# Chapter 0: Overview

## Why This Tutorial Is Structured This Way

A lot of agent tutorials jump straight into frameworks, SDKs, and demos.  
That is useful if you only want to ship something quickly, but it often leaves two major blind spots:

1. readers never build a stable mental model of what the **agent system** actually is
2. safety gets treated like an add-on, instead of something built into the environment and evaluation loop

This tutorial takes the opposite route:

- first, understand the agentic system conceptually
- then, understand the main safety failure modes
- only then, wire those ideas into code, environments, and training

## What You Should Leave With

By the end, you should be able to:

- explain the difference between a model, an agent, a workflow, and a harness
- describe the engineering evolution from prompt-centric systems to agentic systems
- identify the main attack surfaces for tool-calling agents
- explain why transcript-level success can diverge from environment-level success
- package a small set of safety tasks into an executable environment
- connect that environment to a trainer and evaluate outcomes

## A Practical Definition of "Agent"

Different communities define agents differently.  
For this tutorial, we use a deliberately practical definition:

- an agent is an LLM-based system that can **use tools in a loop** to pursue a user goal

That definition is intentionally close to recent engineering writing from Anthropic and OpenAI, because it maps well onto real systems.

## Key Terms

### Agent loop
The repeated cycle of model inference, tool selection, tool execution, observation, and continuation.

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

## A Short Timeline

The broad arc of the field looks like this:

- **2022:** prompt engineering and chain-of-thought dominated applied LLM work
- **2022-2023:** ReAct made reasoning and acting an explicit loop
- **2023:** Toolformer and related work made tool selection part of the modeling problem
- **2023:** Voyager highlighted memory, skills, and open-ended long-horizon behavior
- **2024-2025:** teams shifted toward production agent engineering, including evaluation, observability, and tool/runtime design
- **2025-2026:** harness engineering and context engineering became first-class concerns for long-running, high-autonomy systems

You do not need to memorize this timeline, but it helps explain why so much current discussion is about scaffolds, traces, and environments rather than just prompts.

## Research Anchors

- [ReAct (2022/2023)](https://arxiv.org/abs/2210.03629)
- [Toolformer (2023)](https://arxiv.org/abs/2302.04761)
- [Voyager (2023)](https://arxiv.org/abs/2305.16291)
- [A practical guide to building agents (OpenAI)](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)
- [Building effective agents (Anthropic)](https://www.anthropic.com/research/building-effective-agents)

## What Lives Where In This Repo

- tutorial chapters: [docs/tutorial](../../docs/tutorial)
- tutorial tasks and manifests: [examples/tutorial](../../examples/tutorial/README.md)
- runtime environment and evaluation: [tasksvc](../../tasksvc)
- tutorial-to-slime adapter: [slime_adapter.py](../../tasksvc/tutorial/slime_adapter.py)

Next: [Chapter 1](01_why_agents.md)
