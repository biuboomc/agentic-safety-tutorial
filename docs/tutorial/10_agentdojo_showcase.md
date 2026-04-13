# Chapter 10: AgentDojo Showcase

The tutorial does not aim to reproduce the full AgentDojo benchmark.  
Instead, it uses AgentDojo-style tasks as a realistic held-out showcase.

## Why Use a Showcase Slice

The curriculum is deliberately small and transparent.  
That is good for teaching, but we still want evidence that the learned behaviors transfer beyond toy tasks.

## What the Showcase Demonstrates

- the agent can still complete benign tasks
- the agent resists prompt-level and content-level attacks better than before
- the capstone did not merely overfit to one hand-authored example

## Showcase Manifest

See:

- [agentdojo_showcase_manifest.json](../../examples/tutorial/agentdojo_showcase_manifest.json)

It defines a held-out extraction slice rather than shipping benchmark data inside the tutorial repo.

## Typical Outcome

The tutorial’s success criterion is not “beat the benchmark.”  
It is:

- understand the full stack
- train a small safety-oriented agent
- show a measurable before/after difference on realistic held-out tasks

That is enough to turn the tutorial into a launchpad for deeper research or system-building work.
