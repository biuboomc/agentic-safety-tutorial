# Chapter 1: Why Agents

## Chatbots vs Agents

A chatbot mainly maps input text to output text.  
An agent is defined by a loop:

- observe the task and current context
- decide on an action
- execute the action
- observe the result
- continue until the goal is done or blocked

That loop introduces state, partial progress, and opportunities for irreversible mistakes.

## Minimal Agent Loop

The smallest useful mental model is:

`goal -> observation -> action -> new observation -> ... -> finish`

For tool-calling agents:

- the **goal** comes from the user query
- the **action** is usually a tool call
- the **observation** is the tool result and updated state
- the **finish** is a final answer plus environment-side success checks

## Why This Matters

Once actions matter, correctness is no longer just “did the answer sound right?”

Instead, we care about:

- did the agent choose the right tool?
- did it use the tool with the right arguments?
- did it change the right state?
- did it avoid making dangerous changes while pursuing the task?

That is the bridge from “LLM behavior” to “agent system behavior.”

## Tutorial Warmup

The warmup task in this repo is [tutorial_calendar_summary](../../examples/tutorial/tutorial_source_tasks.json), a benign calendar task that teaches the basic tool loop without attacks.

Next: [Chapter 2](02_tool_calling_agents.md)
