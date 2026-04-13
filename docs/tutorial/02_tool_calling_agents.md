# Chapter 2: Tool-Calling Agents

## Why This Chapter Exists

Once you decide to build an agent, the next practical question is:

**what exactly is the system that surrounds the model?**

This is where terms like *harness*, *sandbox*, *trajectory*, and *outcome* become useful instead of sounding like jargon.

## The Core Agent Loop

A typical tool-calling loop looks like this:

1. receive the user goal
2. package context for the model
3. let the model choose a tool or produce a final message
4. execute the tool call
5. observe the result
6. update state and continue

This loop ends only when the system decides the task is complete or blocked.

## What Is a Harness?

The most useful practical definition is:

- a **harness** is the orchestration layer that makes a model behave like an agent

In practice, a harness usually does at least six things:

- builds the prompt/context for the next model call
- exposes tool schemas
- parses tool calls
- executes or routes tool calls
- records the transcript
- decides when the loop stops

In recent Anthropic writing, an **agent harness** is the system that enables a model to act as an agent. In recent OpenAI writing, the same general idea appears in explanations of the Codex agent loop and harness engineering.

## Harness vs Environment vs Sandbox

These terms overlap in casual conversation, but it helps to separate them:

### Harness
The orchestration logic around the model.

### Environment
The world the agent can observe and affect: files, messages, balances, documents, tickets, browser state, and so on.

### Sandbox
The execution boundary where actions actually run. A sandbox constrains what the agent can do and isolates side effects.

### Evaluator
The logic that decides whether the task succeeded or the attack succeeded.

## Transcript vs Outcome

This distinction is central to agentic safety.

### Transcript / trajectory
The complete record of the run:

- prompts
- model outputs
- tool calls
- tool results
- intermediate observations

### Outcome
The final environment state.

A transcript may *say* "the booking succeeded" while the outcome shows that no booking exists.  
That is why outcome-grounded evaluation matters.

## Why This Is an Engineering Problem

A tool-calling agent is not only a prompting trick. It is a systems problem involving:

- API design
- permissions
- state design
- termination conditions
- logging and observability
- retry and recovery behavior

That is one reason the field increasingly talks about **agent engineering** and **harness engineering**, not just prompt engineering.

## Research Anchors

- [Unrolling the Codex agent loop (OpenAI)](https://openai.com/index/unrolling-the-codex-agent-loop/)
- [Harness engineering: leveraging Codex in an agent-first world (OpenAI)](https://openai.com/index/harness-engineering/)
- [Demystifying evals for AI agents (Anthropic)](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [Scaling Managed Agents: Decoupling the brain from the hands (Anthropic)](https://www.anthropic.com/engineering/managed-agents)

## Where This Shows Up In Code

- episode loop: [agent_rollout.py](../../tasksvc/runtime/agent_rollout.py)
- environment server: [server.py](../../tasksvc/runtime/server.py)

Next: [Chapter 3](03_agent_fragility.md)
