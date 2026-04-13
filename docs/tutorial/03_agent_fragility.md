# Chapter 3: Agent Fragility

## Why This Chapter Exists

If tool-calling agents are so useful, why do even strong models behave unreliably in the wild?

Because agent failures are usually not a single bug. They are the product of:

- long horizons
- partial observability
- changing context
- irreversible side effects
- weak boundary management

That combination makes agents fragile in ways that plain chatbots are not.

## Four Main Sources of Fragility

### 1. Instruction confusion
The system stops distinguishing between:

- the user's actual goal
- background content
- tool-return text
- attacker-controlled instructions

### 2. Tool confusion
The system can call tools, but it may still:

- choose the wrong tool
- call the right tool with the wrong arguments
- overuse writes when reads would suffice

### 3. State confusion
The model often reasons from a mental summary of state rather than the real underlying environment.  
That summary can drift from reality.

### 4. Oversight confusion
The system may prematurely conclude the task is done because the transcript feels complete, even when the environment outcome is wrong.

## Why Long-Horizon Tasks Are Worse

The longer the loop, the more these errors compound:

- one bad read contaminates later reasoning
- one wrong tool call changes the state seen by future steps
- one ambiguous tool result can produce multiple bad downstream actions

This is why long-horizon coding agents, browser agents, and enterprise tool agents have pushed evaluation and harness design to the foreground.

## Capability vs Context

A useful modern framing is:

- some failures are **capability failures**
- many others are **context failures**

In practice, teams often discover that the model is not strictly incapable; rather, the harness is surfacing too much, too little, or the wrong kind of information at the wrong time.

That is one reason **context engineering** is now treated as a core agent skill.

## Why Fragility Becomes a Safety Problem

Fragility is not just an inconvenience. It is what turns:

- ambiguous instructions
- polluted content
- overly broad tool access

into concrete unsafe behavior.

That is the bridge from "agent reliability" to "agentic safety."

## Research Anchors

- [Effective context engineering for AI agents (Anthropic)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Building effective agents (Anthropic)](https://www.anthropic.com/research/building-effective-agents)
- [Harness engineering: leveraging Codex in an agent-first world (OpenAI)](https://openai.com/index/harness-engineering/)

Next: [Chapter 4](04_attack_surfaces.md)
