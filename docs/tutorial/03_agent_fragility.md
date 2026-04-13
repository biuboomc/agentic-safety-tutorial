# Chapter 3: Agent Fragility

## Why Agents Become Fragile

Agents are fragile because they combine:

- natural-language instruction following
- tool APIs
- persistent or semi-persistent state
- multi-step reasoning under uncertainty

That creates more ways to go wrong than plain question answering.

## Common Failure Modes

### Over-trusting instructions
The agent treats an untrusted text fragment as if it were part of the user’s goal.

### Tool misuse
The agent picks a tool that is syntactically valid but semantically unsafe.

### State corruption
A bad action changes environment state in a way the user did not intend.

### Unsafe delegation
The agent offloads judgment to another step, tool, or prompt without preserving the original boundary.

## A Useful Shift in Perspective

In ordinary LLM safety, we often ask:

- “Will the model say something unsafe?”

In agentic safety, we ask:

- “Will the system do something unsafe?”

That shift is why environment design and evaluation are central.

## Tutorial View

The rest of Part II turns these vague failure modes into:

- attack surfaces
- concrete safety requirements
- environment-side evaluation metrics

Next: [Chapter 4](04_attack_surfaces.md)
