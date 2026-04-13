# Chapter 1: Why Agents

## Why This Chapter Exists

Before talking about safety, we need a stable answer to a simpler question:

**why did the field move from prompts and workflows toward agents at all?**

The short answer is that some tasks require more than one inference and more than one fixed rule path.

## Agents vs Chatbots vs Workflows

### Chatbots
A chatbot mostly maps input text to output text.

### Workflows
A workflow can involve multiple model calls, but the control flow is mostly defined by the developer in advance.

### Agents
An agent uses an LLM to decide what to do next inside a loop:

- what to inspect
- which tool to call
- whether the task is done
- how to recover from partial failure

This is close to the OpenAI practical guide's definition of agents as systems that independently accomplish tasks on a user's behalf, and to Anthropic's newer practical definition of agents as LLMs autonomously using tools in a loop.

## The Engineering Evolution

### Stage 1: Prompt-centric systems
Early LLM apps mostly focused on prompt wording, examples, and one-shot or few-shot patterns.

### Stage 2: Reasoning traces
Work such as **ReAct** made it explicit that reasoning and acting can be interleaved rather than separated.

### Stage 3: Tool-native behavior
**Toolformer** and later tool-use systems highlighted that reliable tool choice, argument choice, and result incorporation are core capabilities rather than side details.

### Stage 4: Memory, skills, and long horizons
Systems like **Voyager** showed that once tasks become longer and more open-ended, memory, skill reuse, and self-directed exploration matter.

### Stage 5: Production agent engineering
As companies moved from demos to deployments, the bottleneck shifted:

- not just model quality
- but orchestration
- context management
- evaluation
- observability
- permissions

### Stage 6: Harness engineering
Recent engineering writing, especially around coding agents, makes the next shift clear: the job is increasingly to build the environment, scaffolding, and feedback loops that let agents work reliably.

## When You Actually Need an Agent

You do **not** always need one.

Agentic systems are most justified when the task involves:

- ambiguous decisions
- many possible tool paths
- open-ended exploration
- partial failure and recovery
- real interaction with external state

If a clean deterministic workflow already solves the task, a workflow is often better.

## Why This Matters for Safety

The moment a system can act, safety changes meaning.

We are no longer asking only:

- "Will the model say something unsafe?"

We are now asking:

- "Will the system inspect the wrong thing?"
- "Will it call the wrong tool?"
- "Will it make an unsafe state change?"

That is why agentic safety is system-level.

## Research Anchors

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)
- [A practical guide to building agents (OpenAI)](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)
- [Effective context engineering for AI agents (Anthropic)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

## Tutorial Bridge

The warmup task in this repo, [tutorial_calendar_summary](../../examples/tutorial/tutorial_source_tasks.json), is intentionally simple: it lets us talk about the agent loop before introducing safety attacks.

Next: [Chapter 2](02_tool_calling_agents.md)
